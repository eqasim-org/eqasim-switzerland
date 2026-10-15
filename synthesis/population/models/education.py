"""Joint study-level and education-country-status imputation for students."""

import logging

import numpy as np
import pandas as pd

from data.structural_survey.structural_survey import EDUCATION_LEVELS


logger = logging.getLogger("synpp")
DEFAULT_PRIOR_STRENGTH = 20.0
AGE_EDGES = [15, 18, 20, 25, 30, 40, 50, 65, np.inf]
AGE_CENTERS = np.array([16, 18.5, 22, 27, 34.5, 44.5, 57, 75])
FEATURES = ["age_band", "employed", "sex", "canton_id"]
DESTINATION_STATUSES = ["domestic", "abroad", "unknown"]
# 14 is an INTERNAL outcome index for pooled tertiary, never an SS source code.
# Synthetic tertiary students retain current_education=NA rather than a fabricated
# Bachelor/Master/doctoral programme; study_level identifies the pooled outcome.
TERTIARY_CODE = 14
OUTCOMES = [(code, status) for code in [0] + list(range(2, 9)) + [TERTIARY_CODE]
            for status in DESTINATION_STATUSES]
EDUCATION_GROUPS = {
    2: "compulsory", 3: "compulsory", 4: "bridging",
    5: "secondary", 6: "secondary", 7: "secondary", 8: "secondary",
    9: "higher_vocational", 10: "higher_vocational",
    11: "bachelor_master", 12: "bachelor_master", 13: "doctorate_habilitation",
    TERTIARY_CODE: "tertiary",
}
MODEL_LEVELS = EDUCATION_LEVELS | {TERTIARY_CODE: "tertiary"}


def configure(context):
    context.stage("synthesis.population.models.students")
    context.stage("data.structural_survey.student_training")
    context.config("random_seed")
    context.config("education_level_prior_strength", DEFAULT_PRIOR_STRENGTH)


def prepare_features(frame):
    """Build the same age, employment, sex and canton keys for donors and people."""
    features = frame[["age", "employed", "sex", "canton_id"]].copy()
    for column in features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features["age_band"] = pd.cut(features["age"], AGE_EDGES, right=False, labels=False)
    return features[FEATURES].fillna(-1).astype(int)


def prepare_training(survey):
    """Keep weighted adult student observations and their joint study/country outcome."""
    # Preserve unknown level/country reports among participating students.
    donors = survey[survey["is_student"].eq(1) & survey["age"].ge(15)].copy()
    weight = pd.to_numeric(donors["weight"], errors="coerce")
    donors = donors[np.isfinite(weight) & weight.gt(0) & np.isfinite(donors["age"])].copy()
    donors["_weight"] = pd.to_numeric(donors["weight"])
    donors["_education_code"] = donors["current_education"].where(donors["current_education"].isin(range(2, 14)), 0).astype(int)
    donors.loc[donors["_education_code"].isin(range(9, 14)), "_education_code"] = TERTIARY_CODE
    country = donors["education_country_code"]
    foreign = country.gt(0) & country.ne(8100)
    if "education_zone_level" in donors and "education_zone_id" in donors:
        swiss_destination = donors["education_zone_id"].notna() & donors["education_zone_level"].isin(["municipality", "quarter"])
    elif "education_municipality" in donors:
        # The standalone audit has raw Swiss school-municipality codes, before
        # boundary mapping. An explicit foreign country always takes priority.
        swiss_destination = donors["education_municipality"].gt(0)
    else:
        swiss_destination = pd.Series(False, index=donors.index)
    donors["_destination_status"] = np.select(
        [foreign, country.eq(8100) | swiss_destination], ["abroad", "domestic"], default="unknown")
    outcome_index = {value: index for index, value in enumerate(OUTCOMES)}
    donors["_outcome"] = [outcome_index[value] for value in zip(donors["_education_code"], donors["_destination_status"])]
    return donors


def fit_model(survey, prior_strength=DEFAULT_PRIOR_STRENGTH):
    """Blend sparse subgroup profiles toward broader profiles using effective N.

    Parent profiles include their child observations; this is weighted smoothing,
    not an independent Bayesian prior. Joint outcomes retain study/country links.
    """
    if not np.isfinite(prior_strength) or prior_strength < 0:
        raise ValueError("Education prior strength must be finite and non-negative")
    donors = prepare_training(survey)
    if donors.empty:
        raise ValueError("No age-15+ student observations with positive survey weights")
    features = prepare_features(donors).reset_index(drop=True)
    features["outcome"] = donors["_outcome"].to_numpy()
    features["weight"] = donors["_weight"].to_numpy()
    profiles = []
    for depth in range(1, len(FEATURES) + 1):
        layer = {}
        grouping = FEATURES[:depth]
        for key, group in features.groupby(grouping, sort=True):
            key = key if isinstance(key, tuple) else (key,)
            weights = group["weight"].to_numpy()
            effective_n = weights.sum() ** 2 / np.square(weights).sum()
            probabilities = np.bincount(group["outcome"], weights=weights, minlength=len(OUTCOMES))
            probabilities = probabilities / probabilities.sum()
            if depth > 1:
                parent = profiles[-1][key[:-1]][0]
                empirical_share = effective_n / (effective_n + prior_strength)
                probabilities = empirical_share * probabilities + (1 - empirical_share) * parent
            layer[key] = (probabilities, effective_n)
        profiles.append(layer)
    return {"profiles": profiles, "prior_strength": prior_strength,
            "training_observations": len(donors)}


def predict_probabilities(model, persons):
    """Use the deepest supported profile and report the backoff actually used."""
    features = prepare_features(persons)
    probabilities = np.zeros((len(persons), len(OUTCOMES)))
    method = np.empty(len(persons), dtype=object)
    support = np.zeros(len(persons))
    profiles = model["profiles"]
    available_bands = np.array([key[0] for key in profiles[0]])
    for key, positions in features.groupby(FEATURES, sort=True).indices.items():
        key = tuple(key)
        if key[0] < 0:
            raise ValueError("Education prediction requires age 15 or older")
        if key[:1] not in profiles[0]:
            nearest = available_bands[np.argmin(np.abs(AGE_CENTERS[available_bands] - AGE_CENTERS[key[0]]))]
            chosen, effective_n = profiles[0][(nearest,)]
            label = "nearest_observed_age_band"
        else:
            chosen, effective_n = profiles[0][key[:1]]
            label = "age_band"
            for depth in range(2, len(FEATURES) + 1):
                if key[:depth] in profiles[depth - 1]:
                    chosen, effective_n = profiles[depth - 1][key[:depth]]
                    label = "_".join(FEATURES[:depth]) + "_smoothed"
        probabilities[positions] = chosen
        method[positions] = label
        support[positions] = effective_n
    return probabilities, method, support


def assign_education(persons, survey, seed, prior_strength=DEFAULT_PRIOR_STRENGTH, chunk_size=50000):
    """Impute study attributes for existing students without changing student flags.

    The structural survey covers ages 15+, so children retain an explicit coverage
    marker. One RNG stream across chunks makes draws independent of chunk size.
    """
    if chunk_size <= 0:
        raise ValueError("Education prediction chunk size must be positive")
    result = persons.copy()
    if not result.index.is_unique:
        raise ValueError("Education assignment requires a unique population index")
    result["current_education"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["study_level"] = "unresolved"
    result["education_group"] = "unresolved"
    result["education_destination_status"] = "unknown"
    result["education_imputation_method"] = "missing_age_or_student_status"
    result["education_imputation_effective_n"] = np.nan
    result["education_is_imputed"] = False
    nonstudents = result["is_student"].eq(0)
    result.loc[nonstudents, "current_education"] = 1
    result.loc[nonstudents, "study_level"] = "none"
    result.loc[nonstudents, "education_group"] = "not_in_education"
    result.loc[nonstudents, "education_destination_status"] = "not_applicable"
    result.loc[nonstudents, "education_imputation_method"] = "existing_nonstudent_flag"
    children = result["is_student"].eq(1) & result["age"].lt(15)
    result.loc[children, "study_level"] = "below_survey_age_coverage"
    result.loc[children, "education_group"] = "below_survey_age_coverage"
    result.loc[children, "education_imputation_method"] = "below_survey_age_coverage"
    adult_students = result["is_student"].eq(1) & result["age"].ge(15) & np.isfinite(result["age"])
    if not adult_students.any():
        return result
    model = fit_model(survey, prior_strength)
    rng = np.random.RandomState(seed)
    indices = result.index[adult_students]
    for start in range(0, len(indices), chunk_size):
        selected = indices[start:start + chunk_size]
        probabilities, method, support = predict_probabilities(model, result.loc[selected])
        # Accommodation evidence is descriptive: recovered students draw from
        # the same study/country profiles as otherwise comparable residents.
        cumulative = probabilities.cumsum(axis=1)
        cumulative[:, -1] = 1.0
        picks = (rng.random_sample(len(selected))[:, None] >= cumulative).sum(axis=1)
        codes = np.array([OUTCOMES[pick][0] for pick in picks])
        statuses = [OUTCOMES[pick][1] for pick in picks]
        result.loc[selected, "current_education"] = pd.array(
            np.where((codes > 0) & (codes != TERTIARY_CODE), codes, np.nan), dtype="Int64")
        result.loc[selected, "study_level"] = pd.Series(codes).map(MODEL_LEVELS).fillna("unresolved").to_numpy()
        result.loc[selected, "education_group"] = pd.Series(codes).map(EDUCATION_GROUPS).fillna("unresolved").to_numpy()
        result.loc[selected, "education_destination_status"] = statuses
        result.loc[selected, "education_imputation_method"] = method
        result.loc[selected, "education_imputation_effective_n"] = support
        result.loc[selected, "education_is_imputed"] = True
    logger.info("Imputed joint education attributes for %d students from %d survey observations",
                len(indices), model["training_observations"])
    return result


def execute(context):
    survey = context.stage("data.structural_survey.student_training")
    return assign_education(context.stage("synthesis.population.models.students"), survey,
                            context.config("random_seed"), context.config("education_level_prior_strength"))
