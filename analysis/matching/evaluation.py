"""Fold-local matching, negative controls and covariate-shift diagnostics."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from synthesis.population.matching import matched_v1
from synthesis.population.matching.V2 import features
from synthesis.population.matching.V2.donors import retrieval_space


def raw_space(source, target):
    zeros = np.zeros((len(source), 1), dtype=np.float32)
    return retrieval_space(source, target, zeros, np.zeros((len(target), 1), dtype=np.float32), 1.)


def random_donors(source, target, rng, n_draws):
    selected = np.full((len(target), n_draws), -1, dtype=int)
    groups = source.groupby(["employed", "is_student"]).indices
    for key, rows in target.groupby(["employed", "is_student"]).indices.items():
        donors = groups.get(key, [])
        if len(donors):
            weights = source.iloc[donors]["person_weight"].to_numpy(dtype=float)
            selected[rows] = rng.choice(donors, size=(len(rows), n_draws), p=weights / weights.sum())
    return selected


class V1Context:
    """Run the actual, unchanged V1 stage on an outer-training donor file only."""

    def __init__(self, parent, raw_source, population, trips, chains, seed):
        self.parent, self.seed = parent, seed
        population = population.copy()
        population["employed"] = np.where(population["employed"].eq(1), 1, 3)
        population["N_children_under_12"] = population["N_children_under_12"].fillna(0).gt(0)
        # V1 deliberately compares against the literal "None" before collapsing
        # PT quality. Undo V2's lowercase normalization for this independent run.
        quality = population["ovgk"].astype("string")
        population["ovgk"] = quality.str.upper().where(~quality.str.lower().eq("none"), "None")
        population["collective_housing_resident"] = False
        population["household_id"] = population["person_id"]
        for col in ("sp_region", "canton_id", "marital_status"):
            if col not in raw_source:
                raw_source = raw_source.assign(**{col: -1})
        # Work inputs use original strings/metres; V1 performs its own preparation.
        work = population[["person_id", "work_location_type", "commute_distance"]].copy()
        work["commute_distance"] = work["commute_distance"].fillna(-1)
        population = population.drop(columns=["work_location_type", "commute_distance", "mz_id"])
        self.stages = {"data.microcensus.persons": raw_source, "data.microcensus.trips": (trips, None),
                       "data.microcensus.activity_chains": chains,
                       "synthesis.population.sampled": population, "work_locations": work}

    def stage(self, name):
        return self.stages[name] if name in self.stages else self.parent.stage(name)

    def config(self, name):
        if name == "random_seed":
            return self.seed
        return self.parent.config(name)

    def progress(self, iterable, *args, **kwargs):
        return iterable


def v1_donors(context, raw_source, source, target, trips, chains, seed, n_draws):
    if context.stage("data.constants").census != "statpop":
        raise ValueError("The faithful V1 holdout adapter currently supports STATPOP exact ages")
    lookup = pd.Series(np.arange(len(source)), index=source["mz_id"])
    selected = np.full((len(target), n_draws), -1, dtype=int)
    for draw in range(n_draws):
        proxy = V1Context(context, raw_source, target, trips, chains, seed + draw)
        mapping, _ = matched_v1.execute(proxy)
        donor_ids = target["person_id"].map(mapping.set_index("person_id")["mz_person_id"])
        selected[:, draw] = donor_ids.map(lookup).fillna(-1).to_numpy(dtype=int)
    return selected


def masked_neighbors(source, target, source_embedding, target_embedding, k=20):
    result = np.full((len(target), k), -1, dtype=int)
    groups = source.groupby(["employed", "is_student"]).indices
    for key, rows in target.groupby(["employed", "is_student"]).indices.items():
        donors = groups.get(key, [])
        if len(donors):
            count = min(k, len(donors))
            model = NearestNeighbors(n_neighbors=count, n_jobs=1).fit(source_embedding[donors])
            result[np.asarray(rows)[:, None], np.arange(count)] = np.asarray(donors)[model.kneighbors(target_embedding[rows], return_distance=False)]
    return result


def shuffle_group(target, columns, rng):
    """Keep joint household features together and preserve employment/student masks."""
    result = target.copy()
    for rows in target.groupby(["employed", "is_student"]).indices.values():
        permutation = rng.permutation(rows)
        result.loc[rows, columns] = target.loc[permutation, columns].to_numpy()
    return result


def domain_diagnostics(source, population, seed, ratio_clip):
    """Cross-fit P(target|X) using covariates only, with survey-weighted source rows.

    Equal prior mass for the two domains makes the posterior odds a density
    ratio relative to the weighted survey. This supports risk transport under
    covariate shift; it does not identify unobserved survey selection bias.
    """
    if not np.isfinite(ratio_clip) or ratio_clip <= 0:
        raise ValueError("matching_validation_ratio_clip must be finite and positive")
    a, b = raw_space(source, population)
    x = np.vstack([a, b])
    labels = np.r_[np.zeros(len(source), dtype=int), np.ones(len(population), dtype=int)]
    weight = np.r_[source["person_weight"] / source["person_weight"].mean(), np.ones(len(population))]
    probability = np.empty(len(x))
    for train, test in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, labels):
        classifier = LogisticRegression(C=.2, max_iter=1000, class_weight="balanced", random_state=seed)
        classifier.fit(x[train], labels[train], sample_weight=weight[train])
        probability[test] = classifier.predict_proba(x[test])[:, 1]
    odds = probability[:len(source)] / np.maximum(1 - probability[:len(source)], 1e-6)
    ratio = np.minimum(odds, ratio_clip)
    support = population[["person_id", "employed", "is_student", "age", "income_class", "population_density"]].copy()
    support["domain_target_probability"] = probability[len(source):]
    if "household_counts_reconstructed" in population:
        support["household_counts_reconstructed"] = population["household_counts_reconstructed"].to_numpy()
    model = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(a)
    support["nearest_raw_covariate_distance"] = model.kneighbors(b)[0][:, 0]
    support["unseen_category_count"] = 0
    for column in features.CATEGORICAL:
        support["unseen_category_count"] += ~population[column].isin(source[column])
    support["unseen_missingness_count"] = 0
    for column in features.CONTINUOUS:
        if not source[column].isna().any():
            support["unseen_missingness_count"] += population[column].isna()
    core = set(map(tuple, source[["employed", "is_student"]].to_numpy()))
    support["core_supported"] = [tuple(key) in core for key in population[["employed", "is_student"]].to_numpy()]
    report = {"crossfit_domain_auc": float(roc_auc_score(labels, probability, sample_weight=weight)),
              "ratio_clip": ratio_clip, "clipped_source_fraction": float((odds > ratio_clip).mean()),
              "source_respondents": len(source), "target_persons": len(population),
              "target_core_support_fraction": float(support["core_supported"].mean())}
    report["balance"] = _covariate_balance(source, population, ratio)
    return ratio, support, report


def _covariate_balance(source, population, ratio):
    rows = []
    weight = source["person_weight"].to_numpy(dtype=float)
    for column in features.CONTINUOUS:
        a, b = source[column].to_numpy(dtype=float), population[column].to_numpy(dtype=float)
        valid_a, valid_b = np.isfinite(a), np.isfinite(b)
        if not valid_a.any() or not valid_b.any():
            continue
        mean_a = np.average(a[valid_a], weights=weight[valid_a])
        mean_b = np.mean(b[valid_b])
        variance_a = np.average((a[valid_a] - mean_a)**2, weights=weight[valid_a])
        pooled_sd = np.sqrt((variance_a + np.var(b[valid_b])) / 2)
        denominator = pooled_sd if pooled_sd > 1e-8 else 1.
        after = np.average(a[valid_a], weights=(weight*ratio)[valid_a])
        rows.append({"feature": column, "type": "continuous",
                     "discrepancy_before": float((mean_a - mean_b) / denominator),
                     "discrepancy_after": float((after - mean_b) / denominator),
                     "survey_missing": float(np.average(~valid_a, weights=weight)),
                     "target_missing": float(np.mean(~valid_b))})
    for column in features.CATEGORICAL:
        categories = pd.Index(pd.concat([source[column], population[column]]).astype("string").unique())
        a = source[column].astype("string")
        b = population[column].astype("string").value_counts(normalize=True).reindex(categories, fill_value=0)
        before = pd.Series(weight, index=a).groupby(level=0).sum().reindex(categories, fill_value=0) / weight.sum()
        after = pd.Series(weight*ratio, index=a).groupby(level=0).sum().reindex(categories, fill_value=0) / np.sum(weight*ratio)
        rows.append({"feature": column, "type": "categorical",
                     "discrepancy_before": float(np.abs(before-b).sum()/2),
                     "discrepancy_after": float(np.abs(after-b).sum()/2),
                     "survey_missing": None, "target_missing": None})
    return rows
