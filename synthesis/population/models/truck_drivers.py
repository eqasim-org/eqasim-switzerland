import numpy as np
import pandas as pd


def find_truck_drivers(structural_survey, enriched, downsampling):
    se = structural_survey.copy()
    sr = downsampling

    se["weight"]    = (se["weight"] / 3) * sr # Adjust weight as 3 releases of structural survey are combined
    bins            = [16, 25, 35, 45, 55, 65, np.inf]
    labels          = ["16-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    se["age_class"] = pd.cut(se["age"], bins=bins, labels=labels, right=False)
    se["sex"]       = (se["sex"] - 1).astype(str)

    drivers = se[se["isco_code"] == 8332]

    drivers_sociodem = drivers.groupby(["sex", "age_class", "canton_id"], as_index = False, observed = False)["weight"].sum()

    population = enriched.copy()
    candidates = population[(population["employment_status"] == "1") & 
                            (population["driving_license"]) &
                            ~(population["is_student"])]

    candidates["age_class"] = pd.cut(candidates["age"], bins = bins, labels = labels, right = False)

    key_cols = ["sex", "age_class", "canton_id"]

    targets = drivers_sociodem.copy()
    total_n = int(round(targets["weight"].sum()))

    targets["n"] = (targets["weight"] / targets["weight"].sum() * total_n).round().astype(int)

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

    sampled_drivers = pd.concat(sampled, ignore_index=True)["person_id"]

    population["is_truck_driver"] = population["person_id"].isin(sampled_drivers)

    return population