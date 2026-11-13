"""Household-held-out predictive validation of the three production matchers."""

import hashlib
import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from synthesis.population.matching import matched_v1, matched_v2, matched_v3, opening_hours
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.matching.V2.wrappers import DiaryMatcher
from .evaluation import V1Context, domain_diagnostics
from .metrics import diary_outcomes, outcome_scale, score_ensembles
from .validation import _household_sample, _global_rows
from . import predictive_report

logger = logging.getLogger("synpp")


def configure(context):
    matched_v2.configure(context)
    matched_v3.configure(context)
    context.config("analysis_path")
    context.config("matching_predictive_folds", 2)
    context.config("matching_predictive_draws", 10)
    context.config("matching_predictive_max_persons", 12000)
    context.config("matching_predictive_population_sample", 20000)
    context.config("matching_predictive_bootstrap", 500)


def execute(context):

    # Validate the total execution budget before fitting or calling any matcher.

    folds = int(context.config("matching_predictive_folds"))
    draws = int(context.config("matching_predictive_draws"))
    if folds < 2 or draws < 2 or folds * draws > 20:
        raise ValueError("Require >=2 folds/draws and folds * draws <=20 per algorithm")
    seed = int(context.config("random_seed"))
    const = context.stage("data.constants")
    if const.census != "statpop":
        raise ValueError("Predictive validation requires STATPOP exact-age inputs")
    output = Path(context.config("analysis_path")) / "matching" / "predictive_v1_v2_v3"
    output.mkdir(parents=True, exist_ok=True)
    source, population, trips = prepare_inputs(context, const)
    source, population = opening_hours.add_regions(context, source, population)
    # Microcensus person_id is HHNR (one diary respondent per household).
    if "household_id" not in source:
        source["household_id"] = source.mz_id
    source = _household_sample(source, int(context.config("matching_predictive_max_persons")), seed)
    trips = trips.loc[trips.person_id.isin(source.mz_id)].copy()
    population = population.loc[population.age.ge(const.MZ_AGE_THRESHOLD) & ~population.home_only]
    if "is_crossing_the_border" in population:
        from data.utils import coerce_boolean_series
        population = population.loc[~coerce_boolean_series(population.is_crossing_the_border,
                                                          name="is_crossing_the_border")]
    population = population.sample(n=min(len(population), int(context.config(
        "matching_predictive_population_sample"))), random_state=seed).reset_index(drop=True)

    # Save outcomes and overlap diagnostics; target reweighting is only a sensitivity check.

    observed, labels = diary_outcomes(source, trips)
    groups = source.household_id.fillna(source.mz_id)
    ratio, support, shift = domain_diagnostics(source, population, seed, 20.)
    pd.DataFrame(shift.pop("balance")).to_csv(output / "covariate_balance.csv", index=False)
    support.to_csv(output / "population_support.csv", index=False)
    respondents = source[["mz_id", "age", "sex", "canton_id", "population_density",
                          "employment_status", "work_location_type", "education_distance_source"]].copy()
    respondents["household_group"] = groups.to_numpy()
    respondents["survey_weight"] = source.person_weight.to_numpy()
    respondents["target_weight"] = source.person_weight.to_numpy() * ratio
    respondents.to_csv(output / "respondents.csv", index=False)
    identity = {"seed": seed, "folds": folds, "draws": draws,
                "v2_options": asdict(MatchingOptions.from_context(context)),
                "threshold": int(context.config("matching_minimum_observations")),
                "density_bounds": list(context.config("matching_density_bounds")),
                "opening_hours": bool(context.config("consider_opening_hours")),
                "source_sha256": hashlib.sha256(pd.util.hash_pandas_object(source, index=True).to_numpy().tobytes()).hexdigest(),
                "trips_sha256": hashlib.sha256(pd.util.hash_pandas_object(trips, index=True).to_numpy().tobytes()).hexdigest(),
                "algorithms_sha256": hashlib.sha256(b"".join(Path(m.__file__).read_bytes()
                    for m in (matched_v1, matched_v2, matched_v3))).hexdigest()}
    identity_path = output / "experiment_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError(f"Existing experiment differs; preserve it and select a new analysis_path: {output}")
    identity_path.write_text(json.dumps(identity, indent=2))
    raw = context.stage("data.microcensus.persons")
    chains = context.stage("data.microcensus.activity_chains")
    scores, timings, splits = [], [], []

    # Predict every held-out diary using donors from other households only.

    for fold, (train, test) in enumerate(GroupKFold(folds, shuffle=True, random_state=seed).split(
            source, groups=groups)):
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
        donors = source.iloc[train].reset_index(drop=True)
        target = source.iloc[test].reset_index(drop=True).assign(person_id=source.iloc[test].mz_id.to_numpy())
        donor_trips = trips.loc[trips.person_id.isin(donors.mz_id)].copy()
        raw_donors = raw.loc[raw.person_id.isin(donors.mz_id)].copy()
        directory = output / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        fold_seed = seed + 1009 * fold
        _, scaled, scales = outcome_scale(observed, train, source.person_weight.to_numpy())
        splits.append({"fold": fold, "donors": len(train), "test": len(test),
                       "training_households": int(groups.iloc[train].nunique()),
                       "test_households": int(groups.iloc[test].nunique()), "scales": scales.tolist()})
        pd.DataFrame({"mz_id": source.mz_id, "partition": np.where(
            np.isin(np.arange(len(source)), train), "donor", "test")}).to_csv(directory / "split.csv", index=False)
        selections = {}
        for name, module in (("V1", matched_v1), ("V3", matched_v3)):
            selected = np.full((len(target), draws), -1, dtype=int)
            lookup = pd.Series(np.arange(len(donors)), index=donors.mz_id)
            for draw in range(draws):
                logger.info("Held-out %s fold %d/%d draw %d/%d", name, fold + 1, folds, draw + 1, draws)
                draw_path = directory / name / f"draw_{draw}"
                proxy = FoldContext(context, raw_donors, target, donor_trips, chains,
                                    fold_seed + draw, draw_path)
                started = time.perf_counter()
                saved = draw_path / "validation_prediction.npz"
                if saved.exists():
                    checkpoint = np.load(saved)
                    selected[:, draw] = checkpoint["rows"]
                    elapsed = float(checkpoint["seconds"])
                else:
                    mapping, removed = module.execute(proxy)
                    ids = target.person_id.map(mapping.set_index("person_id").mz_person_id)
                    ids = ids.mask(target.person_id.isin(removed))
                    selected[:, draw] = ids.map(lookup).fillna(-1).to_numpy(dtype=int)
                    elapsed = time.perf_counter() - started
                    np.savez_compressed(saved, rows=selected[:, draw], seconds=elapsed)
                timings.append({"algorithm": name, "fold": fold, "draw": draw,
                                "phase": "matching", "seconds": elapsed})
            selections[name] = selected
        started = time.perf_counter()
        matcher = DiaryMatcher(replace(MatchingOptions.from_context(context, random_seed=fold_seed), overwrite=False),
                               cache_directory=directory / "V2").fit(donors, donor_trips)
        timings.append({"algorithm": "V2", "fold": fold, "draw": -1,
                        "phase": "fit", "seconds": time.perf_counter() - started})
        (directory / "training_report.json").write_text(json.dumps(matcher.training_report_, indent=2))
        # Ten independent single-draw calls give an explicit, auditable run count.
        selections["V2"] = np.full((len(target), draws), -1, dtype=int)
        for draw in range(draws):
            started = time.perf_counter()
            saved = directory / "V2" / f"validation_draw_{draw}.npz"
            if saved.exists():
                checkpoint = np.load(saved)
                selections["V2"][:, draw] = checkpoint["rows"]
                elapsed = float(checkpoint["seconds"])
            else:
                selections["V2"][:, draw] = matcher.predict(target, random_seed=fold_seed + draw).donor_rows
                elapsed = time.perf_counter() - started
                np.savez_compressed(saved, rows=selections["V2"][:, draw], seconds=elapsed)
            timings.append({"algorithm": "V2", "fold": fold, "draw": draw,
                            "phase": "matching", "seconds": elapsed})
        for name, selected in selections.items():
            global_rows = _global_rows(train, selected)
            np.savez_compressed(directory / f"{name}_donor_rows.npz", rows=global_rows,
                                test_ids=target.person_id.to_numpy(), source_ids=source.mz_id.to_numpy())
            scored = score_ensembles(observed, labels, test, global_rows, scaled)
            scored["mz_id"], scored["algorithm"], scored["fold"] = target.person_id.to_numpy(), name, fold
            scores.append(scored)
        # Checkpoint completed folds so an interrupted job retains useful evidence.
        pd.concat(scores, ignore_index=True).to_csv(output / "out_of_fold_scores.csv", index=False)
        pd.DataFrame(timings).to_csv(output / "runtime.csv", index=False)

    # Render an independently rebuildable report from saved per-person scores.

    manifest = {"seed": seed, "respondents": len(source), "folds": folds, "draws": draws,
                "matching_executions_per_algorithm": folds * draws, "v2_fits": folds,
                "bootstrap": int(context.config("matching_predictive_bootstrap")),
                "splits": splits, "shift": shift, "implementation_sha256": validate(context),
                "density_area_km2": float(np.pi * (const.POPULATION_DENSITY_RADIUS / 1000.) ** 2),
                "density_bounds": list(context.config("matching_density_bounds")),
                "interpretation": "Household-held-out conditional diary prediction, not proof of population representativeness"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return predictive_report.render(output)


class FoldContext(V1Context):
    """Adapt survey predictors to the actual V1/V3 population-stage contracts."""

    def __init__(self, parent, raw, target, trips, chains, seed, directory):
        super().__init__(parent, raw, target, trips, chains, seed)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        population = self.stages["synthesis.population.sampled"]
        population["household_id"] = target.get("household_id", target.mz_id).to_numpy()
        population["non_private_home_only_households"] = False
        population.drop(columns=[c for c in population if c.startswith("commute_distance") or
                                c.startswith("education_distance")], inplace=True)
        # Supply usual education predictors, rather than synthetic anchors with unrelated IDs.
        education = target[["person_id"]].copy()
        education["commute_distance"] = target.commute_distance_education.fillna(target.commute_distance).fillna(-1)
        self.stages["education_locations"] = education
        commutes = dict(parent.stage("data.microcensus.commute"))
        commutes["education"] = commutes["education"].loc[
            commutes["education"].person_id.isin(raw.person_id)].copy()
        self.stages["data.microcensus.commute"] = commutes

    def path(self, name=None):
        return str(self.directory) if name is None else self.parent.path(name)


def validate(context):
    from . import metrics, evaluation
    from synthesis.population.matching.V2.provenance import implementation_token
    from synthesis.population.matching.V3 import inputs, donors
    modules = (matched_v1, matched_v2, matched_v3, opening_hours, metrics, evaluation,
               predictive_report, inputs, donors)
    content = Path(__file__).read_bytes() + implementation_token().encode()
    return hashlib.sha256(content + b"".join(Path(m.__file__).read_bytes() for m in modules)).hexdigest()
