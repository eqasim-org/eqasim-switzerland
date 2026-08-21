import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series


def _integer_codes(values, name):
    numeric = pd.to_numeric(values, errors="raise")

    if numeric.isna().any():
        raise ValueError(f"{name} contains missing values")

    non_integer = ~np.equal(numeric, np.floor(numeric))
    if non_integer.any():
        examples = numeric.loc[non_integer].drop_duplicates().head(5).tolist()
        raise ValueError(f"{name} contains non-integer values: {examples!r}")

    return numeric.astype("int64")


def find_truck_drivers(structural_survey, enriched, downsampling):
    se = structural_survey.copy()
    sr = downsampling

    se["weight"]    = (pd.to_numeric(se["weight"], errors="raise") / 3) * sr # Adjust weight as 3 releases of structural survey are combined
    se["age"]       = pd.to_numeric(se["age"], errors="raise")
    se["isco_code"] = pd.to_numeric(se["isco_code"], errors="raise")
    se["canton_id"] = _integer_codes(se["canton_id"], "structural-survey canton_id")
    bins            = [16, 25, 35, 45, 55, 65, np.inf]
    labels          = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    se["age_class"] = pd.cut(se["age"], bins=bins, labels=labels, right=False)
    se["sex"]       = _integer_codes(se["sex"], "structural-survey sex") - 1

    drivers = se[se["isco_code"] == 8332]

    drivers_sociodem = drivers.groupby(["sex", "age_class", "canton_id"], as_index = False, observed = True)["weight"].sum()

    population = enriched.copy()
    population["employment_status"] = _integer_codes(
        population["employment_status"], "population employment_status")
    population["sex"] = _integer_codes(population["sex"], "population sex")
    population["canton_id"] = _integer_codes(
        population["canton_id"], "population canton_id")
    population["age"] = pd.to_numeric(population["age"], errors="raise")

    driving_license = coerce_boolean_series(
        population["driving_license"], name="population driving_license")
    is_student = coerce_boolean_series(
        population["is_student"], name="population is_student")

    candidates = population.loc[
        population["employment_status"].eq(1) & driving_license & ~is_student
    ].copy()

    candidates["age_class"] = pd.cut(candidates["age"], bins = bins, labels = labels, right = False)

    key_cols = ["sex", "age_class", "canton_id"]

    targets = drivers_sociodem.copy()
    target_weight = targets["weight"].sum()

    if not np.isfinite(target_weight) or target_weight < 0:
        raise ValueError(f"Invalid truck-driver target weight: {target_weight!r}")

    total_n = int(round(target_weight))

    if total_n == 0:
        population["is_truck_driver"] = False
        return population

    targets["n"] = (targets["weight"] / target_weight * total_n).round().astype(int)

    diff = total_n - targets["n"].sum()
    if diff != 0:
        adjust_idx = targets["weight"].sort_values(ascending=False).index[:abs(diff)]
        targets.loc[adjust_idx, "n"] += np.sign(diff)

    sampled = []
    for _, row in targets.iterrows():
        mask = np.ones(len(candidates), dtype = bool)

        for col in key_cols:
            mask &= candidates[col].eq(row[col])

        group = candidates[mask]
        n     = int(row["n"])
        if n == 0 or len(group) == 0:
            continue
        sampled.append(group.sample(n = n, replace = len(group) < n, random_state = 42))

    if not sampled:
        target_examples = (
            targets.loc[targets["n"] > 0, key_cols]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            "No eligible population members matched any positive truck-driver "
            f"target stratum ({len(candidates)} eligible candidates; "
            f"target examples: {target_examples!r})"
        )

    sampled_drivers = pd.concat(sampled, ignore_index=True)["person_id"]

    population["is_truck_driver"] = population["person_id"].isin(sampled_drivers)

    return population
