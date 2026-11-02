"""Weighted structural-survey commute distances for secondary students."""

import numpy as np
import pandas as pd


MINIMUM_EFFECTIVE_N = 50.
AGE_EDGES = [15, 18, 24, 30, 40, 50, 60, np.inf]
# Codes 2 and 3 describe the same compulsory-school pool. Other programmes
# remain distinct; sparse age cells borrow all ages of the same programme.
STUDY_GROUPS = {2: "compulsory", 3: "compulsory", 4: "bridging",
                5: "specialised_secondary", 6: "vocational_secondary",
                7: "general_secondary", 8: "vocational_maturity"}


def configure(context):
    context.stage("data.structural_survey.structural_survey")
    context.config("secondary_education_distance_minimum_effective_n", MINIMUM_EFFECTIVE_N)
    context.config("education_distance_ss_unit", "km")


def age_bands(age):
    return pd.cut(age, AGE_EDGES, right=False, labels=False)


def fit_distance_model(survey, minimum_effective_n=MINIMUM_EFFECTIVE_N, unit="km"):
    """Fit empirical distance CDFs without truncating legitimate long commutes.

    Education departure and destination must both be domestic. DISTTOSCHOOL_AIR
    describes that departure origin, not necessarily the registered residence.
    Effective sample size, rather than summed expansion weights, determines
    whether an age/programme cell has enough observations to stand alone.
    """
    if not np.isfinite(minimum_effective_n) or minimum_effective_n <= 0:
        raise ValueError("Secondary distance minimum effective sample size must be positive")
    if unit not in ["km", "m"]:
        raise ValueError("Secondary survey distance unit must be km or m")
    frame = survey.copy()
    distance = pd.to_numeric(frame["crowfly_distance_to_school"], errors="coerce")
    weight = pd.to_numeric(frame["weight"], errors="coerce")
    country = pd.to_numeric(frame["education_country_code"], errors="coerce")
    selected = (frame["is_student"].eq(1) & frame["age"].ge(15)
                & frame["current_education"].isin(STUDY_GROUPS)
                & frame["education_departure_status"].eq("domestic")
                & frame["education_zone_id"].notna()
                & frame["education_zone_level"].isin(["municipality", "quarter"])
                & ~(country.gt(0) & country.ne(8100))
                & np.isfinite(distance) & distance.ge(0) & np.isfinite(weight) & weight.gt(0))
    frame = frame.loc[selected].copy()
    frame["distance_m"] = distance.loc[selected] * (1000. if unit == "km" else 1.)
    frame["weight"] = weight.loc[selected]
    frame["study_group"] = frame["current_education"].map(STUDY_GROUPS)
    frame["age_band"] = age_bands(frame["age"])
    cells, support = {}, []

    def add_cell(group, band, observations):
        weights = observations["weight"]
        effective_n = weights.sum() ** 2 / (weights ** 2).sum()
        totals = observations.groupby("distance_m", sort=True)["weight"].sum()
        cdf = (totals.cumsum() / totals.sum()).to_numpy(copy=True)
        cdf[-1] = 1.
        cells[(group, band)] = {"distances_m": totals.index.to_numpy(), "cdf": cdf,
                                "effective_n": effective_n, "observations": len(observations)}
        support.append({"study_group": group, "age_band": band, "observations": len(observations),
                        "effective_n": effective_n, "mean_km": np.average(observations.distance_m, weights=weights) / 1000.})

    for group, observations in frame.groupby("study_group", sort=True):
        add_cell(group, "all", observations)
        for band, age_observations in observations.groupby("age_band", sort=True):
            add_cell(group, int(band), age_observations)
    return {"cells": cells, "minimum_effective_n": minimum_effective_n,
            "support": pd.DataFrame(support)}


def assign_distance_targets(persons, model, seed):
    """Use stratified survey quantiles while retaining donors' distance ordering.

    A long donor commute supplies rank, not a literal kilometre target. Random
    tie breaking prevents reused donors from receiving one identical distance.
    Missing donor distance gets a random rank; it never becomes a zero commute.
    Only ordering within an age/programme cohort is retained, not across groups.
    """
    result = persons[["person_id"]].copy().set_index("person_id", drop=False)
    result["education_secondary_target_distance_m"] = np.nan
    result["education_secondary_distance_source"] = "unsupported_programme"
    result["education_secondary_distance_effective_n"] = 0.
    frame = persons.copy()
    frame["study_group"] = frame["current_education"].map(STUDY_GROUPS)
    frame["age_band"] = age_bands(frame["age"])
    rng = np.random.RandomState(seed)
    for (group, band), members in frame.groupby(["study_group", "age_band"], sort=True):
        cell = model["cells"].get((group, int(band)))
        source = "structural_survey_age_programme"
        if cell is None or cell["effective_n"] < model["minimum_effective_n"]:
            cell = model["cells"].get((group, "all"))
            source = "structural_survey_programme_pooled_ages"
        if cell is None:
            # No cross-programme fallback: a gymnasium distance distribution
            # cannot stand in for a missing vocational-school distribution.
            continue
        if cell["effective_n"] < model["minimum_effective_n"]:
            source = "structural_survey_programme_low_support"
        donor = members["commute_home_distance"].to_numpy(dtype=float)
        valid = np.isfinite(donor) & (donor >= 0)
        # Give missing donors positions distributed over the valid donor ranks.
        latent = donor.copy()
        latent[~valid] = rng.choice(donor[valid], size=(~valid).sum()) if valid.any() else 0.
        order = np.lexsort((rng.random_sample(len(members)), latent))
        quantiles = (np.arange(len(members)) + .5) / len(members)
        distances = cell["distances_m"][np.searchsorted(cell["cdf"], quantiles)]
        targets = np.empty(len(members))
        targets[order] = distances
        ids = members["person_id"].to_numpy()
        result.loc[ids, "education_secondary_target_distance_m"] = targets
        result.loc[ids, "education_secondary_distance_source"] = source
        result.loc[ids, "education_secondary_distance_effective_n"] = cell["effective_n"]
    return result.reset_index(drop=True)


def execute(context):
    return fit_distance_model(context.stage("data.structural_survey.structural_survey"),
        context.config("secondary_education_distance_minimum_effective_n"),
        context.config("education_distance_ss_unit"))
