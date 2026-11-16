"""Shared-model diary reconstruction and repeated full-population matching."""

import hashlib
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from synthesis.population.matching import matched_v1, matched_v3, opening_hours
from synthesis.population.matching.V2 import model as model_stage
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.models.non_private import home_only_mask
from .validate_algorithms import FoldContext
from .metrics import diary_outcomes, outcome_scale, score_ensembles
from .validation import _global_rows
from . import repeated_report

logger = logging.getLogger("synpp")
MODEL = "synthesis.population.matching.V2.model"


def configure(context):
    model_stage.configure(context)
    matched_v3.configure(context)
    context.stage(MODEL)
    context.config("analysis_path")
    context.config("matching_repeat_microcensus", 50)
    context.config("matching_repeat_population", 10)
    context.config("matching_repeat_bootstrap", 500)


def execute(context):

    # Load the sole fitted model and all eligible survey/population recipients.

    matcher = context.stage(MODEL)
    seed = int(context.config("random_seed"))
    micro_draws = int(context.config("matching_repeat_microcensus"))
    population_draws = int(context.config("matching_repeat_population"))
    if not 2 <= micro_draws <= 50 or not 1 <= population_draws <= 10:
        raise ValueError("Require 2..50 Microcensus repetitions and 1..10 population repetitions")
    const = context.stage("data.constants")
    if const.census != "statpop":
        raise ValueError("The repeated experiment currently requires STATPOP")
    source, population, trips = prepare_inputs(context, const)
    source, population = opening_hours.add_regions(context, source, population)
    if not source.mz_id.equals(matcher.donors_.mz_id):
        raise ValueError("Experiment donors differ from shared model donors")
    source["household_id"] = source.get("household_id", source.mz_id)
    output = Path(context.config("analysis_path")) / "matching" / "repeated_shared_v1_v2_v3"
    output.mkdir(parents=True, exist_ok=True)
    identity = {"seed": seed, "micro_draws": micro_draws, "population_draws": population_draws,
                "implementation": validate(context), "source": _frame_hash(source),
                "population": _frame_hash(population), "trips": _frame_hash(trips)}
    path = output / "experiment_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(f"Different existing experiment; preserve it and choose a new analysis_path: {output}")
    path.write_text(json.dumps(identity, indent=2))
    micro = output / "microcensus"
    micro.mkdir(exist_ok=True)
    raw = context.stage("data.microcensus.persons")
    chains = context.stage("data.microcensus.activity_chains")
    outcomes, labels = diary_outcomes(source, trips)
    outcomes.assign(mz_id=source.mz_id.to_numpy()).to_csv(output / "survey_outcomes.csv", index=False)
    source[["mz_id", "age", "canton_id", "population_density", "person_weight", "activity_chain"]].to_csv(output / "survey.csv", index=False)
    respondents = source[["mz_id", "age", "sex", "canton_id", "population_density",
                          "employment_status", "work_location_type", "education_distance_source"]].copy()
    respondents["household_group"] = source.household_id.to_numpy()
    respondents["survey_weight"] = source.person_weight.to_numpy()
    # No domain reweighting is claimed in this reconstruction protocol.
    respondents["target_weight"] = source.person_weight.to_numpy()
    respondents["training_role"] = np.where(source.mz_id.isin(matcher.validation_ids_),
                                             "early_stopping_validation", "fit")
    respondents.to_csv(micro / "respondents.csv", index=False)
    scores, splits = [], []
    times = [dict(algorithm="V2", fold=-1, draw=-1, phase="fit_shared_once",
                  seconds=matcher.training_wall_seconds_)]

    # Repeat whole-Microcensus matching 50 times, restricting donors by three groups.

    groups = source.household_id.fillna(source.mz_id)
    for fold, (train, test) in enumerate(GroupKFold(3, shuffle=True, random_state=seed).split(source, groups=groups)):
        donors = source.iloc[train].reset_index(drop=True)
        target = source.iloc[test].reset_index(drop=True).assign(person_id=source.iloc[test].mz_id.to_numpy())
        assert set(groups.iloc[train]).isdisjoint(groups.iloc[test])
        donor_trips = trips.loc[trips.person_id.isin(donors.mz_id)].copy()
        raw_donors = raw.loc[raw.person_id.isin(donors.mz_id)].copy()
        directory = micro / f"fold_{fold}"
        directory.mkdir(exist_ok=True)
        _, scaled, scales = outcome_scale(outcomes, train, source.person_weight.to_numpy())
        splits.append(dict(fold=fold, donors=len(train), recipients=len(test), scales=scales.tolist()))
        pd.DataFrame({"mz_id": source.mz_id, "partition": np.where(np.isin(np.arange(len(source)), train),
            "donor", "recipient")}).to_csv(directory / "split.csv", index=False)
        for name, module in (("V1", matched_v1), ("V2", None), ("V3", matched_v3)):
            selected = np.full((len(target), micro_draws), -1, dtype=int)
            lookup = pd.Series(np.arange(len(donors)), index=donors.mz_id)
            restricted = matcher.with_donors(donors.mz_id) if name == "V2" else None
            for draw in range(micro_draws):
                draw_path = directory / name / f"draw_{draw}"
                draw_path.mkdir(parents=True, exist_ok=True)
                run_seed = seed + fold * 100003 + draw
                def predict():
                    if restricted is not None:
                        return restricted.match(target, random_seed=run_seed).donor_rows
                    proxy = FoldContext(context, raw_donors, target, donor_trips, chains, run_seed, draw_path)
                    mapping, removed = module.execute(proxy)
                    ids = target.person_id.map(mapping.set_index("person_id").mz_person_id)
                    return ids.mask(target.person_id.isin(removed)).map(lookup).fillna(-1).to_numpy(dtype=int)
                selected[:, draw], elapsed = _checkpoint(draw_path / "prediction.npz", predict)
                times.append(dict(algorithm=name, fold=fold, draw=draw, phase="matching", seconds=elapsed))
                logger.info("Microcensus %s group %d/3 repetition %d/%d", name, fold + 1, draw + 1, micro_draws)
            global_rows = _global_rows(train, selected)
            np.savez_compressed(directory / f"{name}_donor_rows.npz", rows=global_rows,
                                test_ids=target.person_id.to_numpy(), source_ids=source.mz_id.to_numpy())
            scored = score_ensembles(outcomes, labels, test, global_rows, scaled)
            scored["mz_id"], scored["algorithm"], scored["fold"] = target.person_id.to_numpy(), name, fold
            scores.append(scored)
        pd.concat(scores, ignore_index=True).to_csv(micro / "out_of_fold_scores.csv", index=False)
        pd.DataFrame(times).to_csv(micro / "runtime.csv", index=False)
    manifest = dict(seed=seed, respondents=len(source), folds=3, draws=micro_draws,
        matching_executions_per_algorithm=3 * micro_draws, whole_microcensus_repetitions=micro_draws,
        bootstrap=int(context.config("matching_repeat_bootstrap")), shared_model=True,
        v2_fits=1, training_report=matcher.training_report_, splits=splits,
        density_area_km2=float(np.pi * (const.POPULATION_DENSITY_RADIUS / 1000.) ** 2),
        density_bounds=list(context.config("matching_density_bounds")))
    (micro / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Repeat matching of the entire configured synthetic population ten times.

    pop = output / "population"
    pop.mkdir(exist_ok=True)
    roster = context.stage("synthesis.population.sampled")
    eligible = roster.age.ge(const.MZ_AGE_THRESHOLD) & ~home_only_mask(roster)
    roster.loc[eligible, ["person_id", "canton_id", "population_density"]].to_csv(pop / "recipients.csv", index=False)
    target = population.loc[population.person_id.isin(roster.loc[eligible, "person_id"])].reset_index(drop=True)
    for name, module in (("V1", matched_v1), ("V2", None), ("V3", matched_v3)):
        for draw in range(population_draws):
            directory = pop / name / f"draw_{draw}"
            directory.mkdir(parents=True, exist_ok=True)
            run_seed = seed + 10000019 + draw
            def predict():
                if module is None:
                    result = matcher.match(target, random_seed=run_seed)
                    ids = pd.Series(result.donor_ids, index=target.person_id)
                else:
                    mapping, removed = module.execute(PopulationContext(context, run_seed, directory))
                    ids = mapping.loc[~mapping.person_id.isin(removed)].set_index("person_id").mz_person_id
                return roster.loc[eligible, "person_id"].map(ids).fillna(-1).to_numpy(dtype=np.int64)
            _, elapsed = _checkpoint(directory / "prediction.npz", predict)
            times.append(dict(algorithm=name, fold=-1, draw=draw, phase="population_matching", seconds=elapsed))
            pd.DataFrame(times).to_csv(output / "runtime.csv", index=False)
            logger.info("Full synthetic population %s repetition %d/%d", name, draw + 1, population_draws)
    manifest.update(population_repetitions=population_draws, population_eligible=int(eligible.sum()),
                    model_stage=MODEL, interpretation="Shared-model donor-restricted reconstruction, not unseen-diary prediction")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Save summaries and render both report sections without further matching.

    repeated_report.summarize_population(output)
    return repeated_report.render(output)


def _frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def _checkpoint(path, predict):
    if path.exists():
        with np.load(path) as data:
            return data["rows"].copy(), float(data["seconds"])
    started = time.perf_counter()
    rows = predict()
    seconds = time.perf_counter() - started
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, rows=rows, seconds=seconds)
    temporary.replace(path)
    return rows, seconds


class PopulationContext:
    def __init__(self, parent, seed, directory):
        self.parent, self.seed, self.directory = parent, seed, directory

    def config(self, name):
        return self.seed if name == "random_seed" else self.parent.config(name)

    def path(self, name=None):
        return str(self.directory) if name is None else self.parent.path(name)

    def __getattr__(self, name):
        return getattr(self.parent, name)


def validate(context):
    from synthesis.population.matching.V2.provenance import implementation_token
    from synthesis.population.matching.V3 import inputs, donors
    from . import metrics
    content = Path(__file__).read_bytes() + implementation_token().encode()
    return hashlib.sha256(content + b"".join(Path(m.__file__).read_bytes()
        for m in (matched_v1, matched_v3, inputs, donors, metrics))).hexdigest()
