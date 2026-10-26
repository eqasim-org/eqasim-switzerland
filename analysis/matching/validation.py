"""Nested respondent/household-disjoint evaluation shared by both analyses."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from data.utils import coerce_boolean_series

from synthesis.population.matching import matched_v1, matched_v2
from synthesis.population.matching.V2.features import FEATURE_GROUPS, feature_inventory
from synthesis.population.matching.V2.donors import select_donors
from .evaluation import (domain_diagnostics, masked_neighbors, raw_space,
                         random_donors, shuffle_group, v1_donors)
from .metrics import diary_outcomes, energy_score, outcome_scale, score_ensembles
from . import evaluation, metrics
from synthesis.population.matching.V2 import features
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.matching.V2.provenance import implementation_token
from synthesis.population.matching.V2.wrappers import DiaryMatcher
from synthesis.population.matching.V2.runtime import torch

logger = logging.getLogger("synpp")


def configure(context):
    matched_v2.configure(context)
    matched_v1.configure(context)
    context.config("matching_validation_folds", 3)
    context.config("matching_validation_repeats", 2)
    context.config("matching_validation_max_persons", 12000)
    context.config("matching_validation_population_sample", 20000)
    context.config("matching_validation_draws", 20)
    context.config("matching_validation_permutation_repeats", 3)
    context.config("matching_validation_ratio_clip", 20.0)
    context.config("matching_validation_bootstrap", 500)


def execute(context):

    # Prepare shared inputs once, retaining full households while recovering counts.

    seed = int(context.config("random_seed"))
    const = context.stage("data.constants")
    if const.census != "statpop":
        raise ValueError("Matching validation currently requires STATPOP; ARE lacks exact age ground truth")
    source, population, trips = prepare_inputs(context, const)
    source = _household_sample(source, int(context.config("matching_validation_max_persons")), seed)
    trips = trips.loc[trips["person_id"].isin(source["mz_id"])].copy()
    population = population.loc[population["age"] >= const.MZ_AGE_THRESHOLD]
    cross_border_excluded = 0
    if "is_crossing_the_border" in population:
        border = coerce_boolean_series(population["is_crossing_the_border"], name="is_crossing_the_border")
        cross_border_excluded = int(border.sum())
        population = population.loc[~border]
    population = population.sample(n=min(len(population), int(context.config("matching_validation_population_sample"))), random_state=seed).reset_index(drop=True)
    folds = int(context.config("matching_validation_folds"))
    repeats = int(context.config("matching_validation_repeats"))
    draws = int(context.config("matching_validation_draws"))
    permutation_repeats = int(context.config("matching_validation_permutation_repeats"))
    if folds < 2 or repeats < 1 or draws < 2 or permutation_repeats < 1 or len(population) < 3:
        raise ValueError("Validation needs >=2 folds/draws, positive repeats/permutations and >=3 target people")
    output = Path(context.path())
    output.mkdir(parents=True, exist_ok=True)
    feature_inventory(source, population).to_csv(output / "feature_inventory.csv", index=False)
    outcomes, labels = diary_outcomes(source, trips)
    groups = source.get("household_id", source["mz_id"]).fillna(source["mz_id"])
    raw = context.stage("data.microcensus.persons").copy()
    chains = context.stage("data.microcensus.activity_chains").copy()

    # Estimate covariate overlap without using any observed travel outcomes.

    ratio, support, shift_report = domain_diagnostics(
        source, population, seed, float(context.config("matching_validation_ratio_clip")))
    support.to_csv(output / "population_support.csv", index=False)
    pd.DataFrame(shift_report.pop("balance")).to_csv(output / "covariate_balance.csv", index=False)
    source[["mz_id", "age", "income_class", "employed", "is_student", "sex", "sp_region",
            "canton_id", "household_size", "N_children_under_12", "population_density"]].assign(
                household_group=groups.to_numpy(), survey_weight=source["person_weight"].to_numpy(),
                respondent_weight=1.,
                target_weight=source["person_weight"].to_numpy() * ratio).to_csv(output / "respondents.csv", index=False)

    # Fit exclusively inside each outer training partition and score unseen diaries.

    scores, importance, model_reports = [], [], []
    split = list(GroupKFold(folds, shuffle=True, random_state=seed).split(source, groups=groups))
    for repeat in range(repeats):
        for fold, (train, test) in enumerate(split):
            fold_seed = seed + repeat * 10007 + fold * 1009
            logger.info("Matching outer validation: repeat %d/%d, fold %d/%d", repeat + 1, repeats, fold + 1, folds)
            directory = output / f"repeat_{repeat}_fold_{fold}"
            directory.mkdir(exist_ok=True)
            donors = source.iloc[train].reset_index(drop=True)
            target = source.iloc[test].reset_index(drop=True)
            target["person_id"] = target["mz_id"]
            donor_trips = trips.loc[trips["person_id"].isin(donors["mz_id"])]
            matcher = DiaryMatcher(MatchingOptions.from_context(context, random_seed=fold_seed),
                                   cache_directory=directory).fit(donors, donor_trips)
            z_train, z_test = matcher.donor_embeddings_, matcher.transform(target)
            report = matcher.training_report_
            _, scaled, scale = outcome_scale(outcomes, train, source["person_weight"].to_numpy())
            # Raw source rows are deliberately restricted before calling the unchanged V1.
            selected = {
                "V1": v1_donors(context, raw.loc[raw["person_id"].isin(donors["mz_id"])].copy(), donors,
                                 target, donor_trips, chains, fold_seed, draws),
                "V2": matcher.predict(target, random_seed=fold_seed, n_draws=draws).donor_rows,
                "stratified_random": random_donors(donors, target, np.random.RandomState(fold_seed), draws),
            }
            a, b = raw_space(donors, target)
            selected["raw_knn"] = select_donors(donors, target, a, b, rng=np.random.RandomState(fold_seed),
                top_k=int(context.config("matching_embedding_top_k")),
                temperature=float(context.config("matching_embedding_temperature")), balance_strength=0.,
                query_batch_size=int(context.config("matching_v2_query_batch_size")), n_draws=draws)[0]
            base = None
            for algorithm, local in selected.items():
                global_rows = _global_rows(train, local)
                scored = score_ensembles(outcomes, labels, test, global_rows, scaled)
                valid = (local >= 0).all(axis=1)
                for column in ("employed", "is_student"):
                    mismatch = np.full(len(test), np.nan)
                    mismatch[valid] = (donors[column].to_numpy()[local[valid]] !=
                                       target[column].to_numpy()[valid, None]).mean(axis=1)
                    scored["mask_violation_" + column] = mismatch
                scored["mz_id"] = source.iloc[test]["mz_id"].to_numpy()
                scored["algorithm"], scored["repeat"], scored["fold"] = algorithm, repeat, fold
                scores.append(scored)
                if algorithm == "V2":
                    base = scored["energy"].to_numpy()
            importance.extend(_permutation_scores(target, matcher,
                source, train, test, scaled, base, fold_seed, draws, repeat, fold, permutation_repeats))
            _export_geometry(directory, donors, target, z_train, z_test, train,
                             a, b, scaled, test)
            model_reports.append({"repeat": repeat, "fold": fold, "seed": fold_seed,
                                  "training_households": int(groups.iloc[train].nunique()),
                                  "test_households": int(groups.iloc[test].nunique()),
                                  "outcome_scales": scale.tolist(), "training": report})

    # Save reproducible per-person evidence shared by comparison and embedding reports.

    pd.concat(scores, ignore_index=True).to_csv(output / "out_of_fold_scores.csv", index=False)
    pd.DataFrame(importance).to_csv(output / "grouped_permutation.csv", index=False)
    manifest = {"seed": seed, "outer_folds": folds, "model_repeats": repeats, "draws": draws,
                "target_universe": "survey-age domestic residents whose diary is not replaced by cross-border modelling",
                "cross_border_target_persons_excluded": cross_border_excluded,
                "respondents": len(source), "shift": shift_report, "models": model_reports,
                "v1_sha256": hashlib.sha256(Path(matched_v1.__file__).read_bytes()).hexdigest(),
                "test_ids_sha256": hashlib.sha256(source["mz_id"].to_numpy().tobytes()).hexdigest(),
                "implementation_sha256": validate(context),
                "source_features_sha256": hashlib.sha256(pd.util.hash_pandas_object(
                    source[features.CONTINUOUS + features.CATEGORICAL], index=False).to_numpy().tobytes()).hexdigest(),
                "observed_trips_sha256": hashlib.sha256(pd.util.hash_pandas_object(trips[[
                    "person_id", "trip_id", "origin_purpose", "purpose", "mode", "departure_time", "arrival_time",
                    "crowfly_distance"]], index=False).to_numpy().tobytes()).hexdigest(),
                "torch_version": str(torch.__version__),
                "model_settings": {key: context.config(key) for key in (
                    "matching_embedding_epochs", "matching_embedding_batch_size", "matching_embedding_top_k",
                    "matching_embedding_temperature", "matching_v2_raw_feature_weight", "matching_v2_balance_strength",
                    "matching_embedding_validation_fraction", "matching_embedding_patience",
                    "matching_embedding_lr", "matching_embedding_dropout", "matching_embedding_weight_decay")},
                "interpretation": "Household-disjoint survey prediction; target risk assumes covariate shift, not unbiased survey truth"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return str(output)


def validate(context):
    return hashlib.sha256((implementation_token() + features.source_token(
        matched_v2, matched_v1, evaluation, metrics)).encode()).hexdigest()


def _household_sample(source, maximum, seed):
    if maximum < 10:
        raise ValueError("matching_validation_max_persons must be at least 10")
    if len(source) <= maximum:
        return source.reset_index(drop=True)
    group = source.get("household_id", source["mz_id"]).fillna(source["mz_id"])
    ordered = pd.Series(group.unique()).sample(frac=1, random_state=seed).tolist()
    size = group.value_counts()
    selected, count = [], 0
    for key in ordered:
        if count and count + size[key] > maximum:
            break
        selected.append(key)
        count += size[key]
    return source.loc[group.isin(selected)].sort_values("mz_id").reset_index(drop=True)


def _global_rows(train, local):
    return np.where(local >= 0, train[np.maximum(local, 0)], -1)


def _permutation_scores(target, matcher, source,
                         train, test, scaled, base, seed, draws, repeat, fold, count):
    records = []
    for group_number, (name, columns) in enumerate(FEATURE_GROUPS.items()):
        for permutation in range(count):
            altered = shuffle_group(target, columns, np.random.RandomState(seed + group_number * 97 + permutation * 7919))
            local = matcher.predict(altered, random_seed=seed, n_draws=draws).donor_rows
            selected = _global_rows(train, local)
            valid = (selected >= 0).all(axis=1)
            score = np.full(len(test), np.nan)
            score[valid] = energy_score(scaled[test[valid]], scaled[selected[valid]])
            for pid, delta in zip(source.iloc[test]["mz_id"], score - base):
                records.append({"mz_id": pid, "group": name, "energy_increase": delta,
                                "repeat": repeat, "fold": fold, "permutation": permutation})
    return records


def _export_geometry(directory, donors, target, a, b, train, raw_a, raw_b, outcomes, test):
    for name, frame, z, y in (("donors", donors, a, outcomes[train]), ("heldout", target, b, outcomes[test])):
        exported = frame[["mz_id", "employed", "is_student", "age", "income_class", "population_density"]].copy()
        for i in range(z.shape[1]):
            exported[f"emb_{i}"] = z[:, i]
        exported.to_csv(directory / (name + "_embeddings.csv"), index=False)
        np.save(directory / (name + "_outcomes.npy"), y)
    for name, x, y in (("learned", a, b), ("raw", raw_a, raw_b)):
        neighbors = masked_neighbors(donors, target, x, y)
        ids = np.where(neighbors >= 0, donors["mz_id"].to_numpy()[np.maximum(neighbors, 0)], -1)
        np.save(directory / (name + "_neighbors.npy"), ids)
