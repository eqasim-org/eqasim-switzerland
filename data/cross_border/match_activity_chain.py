import numpy as np

# Only adult microcensus respondents are matched, whatever the mode or purpose
MINIMUM_AGE = 18

# Require at least 30 minutes at the middle activity of home-X-home chains.
# Do not restrict its clock time: early shifts and late activities remain eligible.
MINIMUM_MAIN_ACTIVITY_DURATION = 30 * 60


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("random_seed")
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.cross_border.sample")
    context.stage("data.microcensus.activity_chains")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):

    # Load the cross-border records and the microcensus chains of adults.

    df           = context.stage("data.cross_border.sample").copy()
    mz_actchains = context.stage("data.microcensus.activity_chains")
    mz_persons   = context.stage("data.microcensus.persons")[['person_id', 'age']]
    mz_actchains = mz_actchains.merge(mz_persons, on = "person_id", how = "left")
    mz_actchains = mz_actchains[mz_actchains["age"] >= MINIMUM_AGE]

    # One shared, seeded RNG for every (purpose, mode) group: re-passing the
    # same random_seed to each .sample() call made groups drawing from the same
    # candidate pool return the very same sequence of microcensus persons.
    rng = np.random.RandomState(context.config("random_seed"))

    purpose_to_actchain = {
        "work": ["home-work-home"],
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

    # Keep the chains whose main activity has a usable timing.

    main_activities = main_activity_times(context.stage("data.microcensus.trips")[0])
    valid_ids = main_activities.index[
        main_activities["end_time"] - main_activities["start_time"] >= MINIMUM_MAIN_ACTIVITY_DURATION
    ]
    mz_actchains = mz_actchains[mz_actchains["person_id"].isin(valid_ids)]

    # Draw a microcensus chain for every cross-border record.

    for purpose, chains in purpose_to_actchain.items():
        for trip_mode in ["car", "pt", "car_passenger"]:
            mask_cb    = (df["trip_purpose"] == purpose) & (df["trip_mode"] == trip_mode)
            candidates = mz_actchains[(mz_actchains["activity_chain"].isin(chains)) &
                                       (mz_actchains["mode_chain"]==f"{trip_mode}-{trip_mode}")][["person_id", "person_weight"]]

            candidates.columns = ["mz_person_id", "mz_person_weight"]
            N_sample           = np.sum(mask_cb)

            if N_sample == 0:
                continue

            assert len(candidates) > 0, (
                f"No microcensus activity chain available for purpose '{purpose}' with mode '{trip_mode}' "
                f"on '{day}' ({N_sample} cross-border agents need one)."
            )

            sampled_ids = candidates["mz_person_id"].sample(
                n = N_sample,
                weights = candidates["mz_person_weight"],
                random_state = rng,
                replace = True
            ).values

            # Shuffle before assigning so the draw order carries no structure
            # from the order of the cross-border records themselves. permutation
            # returns a new array, unlike shuffle, which cannot write into the
            # read-only view .values hands back here.
            sampled_ids = rng.permutation(sampled_ids)

            df.loc[mask_cb, "mz_person_id"] = sampled_ids

    assert len(df[df["mz_person_id"].isna()]) == 0

    # Return the sampled population with its reproducible donor identifiers.

    return df


def main_activity_times(mz_trips):
    """
    Start and end [s] of the activity between the first and the second trip of every
    microcensus person, indexed by person_id. Only used for home-X-home chains, which
    have exactly these two trips.
    """

    trips = mz_trips.sort_values(["person_id", "trip_id"]).groupby("person_id")

    first  = trips.nth(0).set_index("person_id")["arrival_time"]
    second = trips.nth(1).set_index("person_id")["departure_time"]

    return first.to_frame("start_time").join(second.rename("end_time"), how = "inner")
