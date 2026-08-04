import numpy as np

def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("random_seed")
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.cross_border.sample")
    context.stage("data.microcensus.activity_chains")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")


def execute(context):
    df           = context.stage("data.cross_border.sample")
    mz_actchains = context.stage("data.microcensus.activity_chains")
    mz_persons   = context.stage("data.microcensus.persons")[['person_id', 'age']]
    mz_actchains = mz_actchains.merge(mz_persons, on = "person_id", how = "left")
    cst          = context.stage("data.constants")

    purpose_to_actchain = {
        "work": ["home-work-home"],
        "work_secondary":  ["home-work_secondary-home"],
        "shop": ["home-shop-home", "home-other-home"],
        "leisure": ["home-leisure-home"],
        "education": ["home-education-home"],
        "other": ["home-other-home"]
    }

    day = context.config("specific_day_scenario")
    if day == "weekend":
        mz_actchains = mz_actchains[mz_actchains["weekend"]]
    elif day == "workday":
        mz_actchains = mz_actchains[mz_actchains["workday"]]
    elif day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        mz_actchains = mz_actchains[mz_actchains["day"]==day]

    mode_age_limit = {"car": 18, "pt": cst.MZ_AGE_THRESHOLD, "car_passenger": cst.MZ_AGE_THRESHOLD}
    purpose_age_limit = {"work": 18, "work_secondary": 18, "shop": cst.MZ_AGE_THRESHOLD, "leisure": cst.MZ_AGE_THRESHOLD, 
                         "education": cst.MZ_AGE_THRESHOLD, "other": cst.MZ_AGE_THRESHOLD}
    
    for purpose, chains in purpose_to_actchain.items():
        age_limit = purpose_age_limit.get(purpose, cst.MZ_AGE_THRESHOLD)

        for trip_mode in ["car", "pt", "car_passenger"]:
            age_limit = max(age_limit, mode_age_limit.get(trip_mode, cst.MZ_AGE_THRESHOLD))

            mask_cb    = (df["trip_purpose"] == purpose) & (df["trip_mode"] == trip_mode)
            candidates = mz_actchains[(mz_actchains["activity_chain"].isin(chains)) &
                                       (mz_actchains["mode_chain"]==f"{trip_mode}-{trip_mode}") & 
                                       (mz_actchains["age"] > age_limit)][["person_id", "person_weight"]]

            candidates.columns = ["mz_person_id", "mz_person_weight"]
            N_sample           = np.sum(mask_cb)

            sampled_ids = candidates["mz_person_id"].sample(
                n = N_sample,
                weights = candidates["mz_person_weight"],
                random_state = context.config("random_seed"),
                replace = True
            )

            df.loc[mask_cb, "mz_person_id"] = sampled_ids.values

    assert len(df[df["mz_person_id"].isna()]) == 0

    return df