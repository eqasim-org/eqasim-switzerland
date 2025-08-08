import numpy as np

def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("random_seed")
    context.config("weekend", default = False)

    context.stage("data.cross_border.sample")
    context.stage("data.microcensus.activity_chains")


def execute(context):
    df           = context.stage("data.cross_border.sample")
    mz_actchains = context.stage("data.microcensus.activity_chains")

    purpose_to_actchain = {
        "work": ["home-work-home"],
        "work_secondary":  ["home-work_secondary-home"],
        "shop": ["home-shop-home", "home-other-home"],
        "leisure": ["home-leisure-home"],
        "education": ["home-education-home"],
        "other": ["home-other-home"]
    }

    mz_actchains = mz_actchains[mz_actchains["weekend"] == context.config("weekend")]

    for purpose, chains in purpose_to_actchain.items():
        for trip_mode in ["car", "pt", "car_passenger"]:
            mask_cb    = (df["trip_purpose"] == purpose) & (df["trip_mode"] == trip_mode)
            candidates = mz_actchains[(mz_actchains["activity_chain"].isin(chains)) &
                                       (mz_actchains["mode_chain"]==f"{trip_mode}-{trip_mode}")][["person_id", "person_weight"]]

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