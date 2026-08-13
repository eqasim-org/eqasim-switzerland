import numpy as np
import pandas as pd

"""
This stage attaches all trip relevant information to the synthetic population.
"""


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.trips")
    context.stage("data.constants")
    context.stage("synthesis.population.spatial.primary.work.work_remotly")

    context.config("random_seed")


BORDER_MATCH_COLUMNS = [
    "cross_border_person_id", "destination_country_raw",
    "interview_geometry_point", "border_crossing_trip_mode", "interview_point_id",
]


def execute(context):
    c      = context.stage("data.constants")

    if c.census == "statpop":
        df_persons = context.stage("synthesis.population.enriched")[[
            "person_id", "mz_person_id", "age", "is_truck_driver", "is_outside_of_switzerland", "is_crossing_the_border", "canton_id"
        ] + BORDER_MATCH_COLUMNS]

    elif c.census == "are_synpop":
        df_persons = context.stage("synthesis.population.enriched")[[
            "person_id", "mz_person_id", "age_class", "is_truck_driver", "is_outside_of_switzerland", "is_crossing_the_border", "canton_id"
        ] + BORDER_MATCH_COLUMNS]

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips")[0], copy=True)[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "origin_purpose"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "following_purpose", "preceding_purpose"]

    df_trips = pd.merge(df_persons, df_trips, on="mz_person_id")

    # Children do not have any trips from the microcensus
    f = np.isnan(df_trips["mz_person_id"])
    if c.census == "statpop":
        assert ((df_trips[f]["age"] > c.MZ_AGE_THRESHOLD).all())
    elif c.census == "are_synpop":
        assert ((df_trips[f]["age_class"] >= 1).all())

    # We deliberately delete them here, since other persons also may not have any
    # trips. May be improved later. TODO
    df_trips = df_trips[~f]

    df_trips.loc[:, "travel_time"] = df_trips.loc[:, "arrival_time"] - df_trips.loc[:, "departure_time"]

    df_trips = df_trips[["person_id", "mz_person_id", "trip_id",
                         "departure_time", "arrival_time",
                         "travel_time", "mode",
                         "preceding_purpose", "following_purpose",
                         "is_truck_driver", "is_outside_of_switzerland",
                         "is_crossing_the_border"]].sort_values(by=["person_id", "trip_id"])

    # Diversify departure times
    counts = df_trips[["person_id", "trip_id"]].groupby("person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["person_id", "departure_time"]]
                .groupby("person_id")
                .min()
                .reset_index()["departure_time"]
                .values)

    # If first departure time is just 5min after midnight, we only add a deviation of 5min
    interval = np.minimum(1800.0, interval)

    # Set up RNG
    rng    = np.random.RandomState(context.config("random_seed"))
    offset = rng.random_sample(size=(len(counts),)) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"]   += offset
    df_trips["departure_time"]  = np.round(df_trips["departure_time"])
    df_trips["arrival_time"]    = np.round(df_trips["arrival_time"])
    df_trips["trip_duration"]   = df_trips["arrival_time"] - df_trips["departure_time"]

    # Define trip index
    df_trips = df_trips.sort_values(by=["person_id", "trip_id"])
    df_count = df_trips.groupby("person_id").size().reset_index(name="count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])

    # Define remote_walk as main mode to go to work for agents working from home
    remote_agents = context.stage("synthesis.population.spatial.primary.work.work_remotly")
    remote_agents = set(remote_agents["person_id"].values)
    f = (df_trips["following_purpose"] == "work") & (df_trips["person_id"].isin(remote_agents))
    df_trips.loc[f, "mode"] = "remote_walk"

    # Delete trips for truck drivers and agents not in Switzerland
    initial_length = len(df_trips)
    df_trips       = df_trips[~df_trips["is_truck_driver"]]
    final_length   = len(df_trips)
    share          = round((final_length - initial_length) / initial_length * 100, 2)

    print(f"Removed {initial_length - final_length} ({share}%) trips (truck drivers)")

    initial_length = len(df_trips)
    df_trips       = df_trips[~df_trips["is_outside_of_switzerland"].astype("boolean").fillna(False).astype(bool)]
    final_length   = len(df_trips)
    share          = round((final_length - initial_length) / initial_length * 100, 2)

    print(f"Removed {initial_length - final_length} ({share}%) trips (people outside of Switzerland)")

    # Adapt trips for agents crossing the border: they get a single trip
    # (home-border or border-home) instead of their regular
    # microcensus-derived chain. The destination country, crossing point, and
    # trip mode are the ones synthesis.population.models.cross_border already
    # matched this person to (from data.cross_border.swiss_residents_od) --
    # reading that match here instead of sampling independently keeps it
    # consistent with matsim/scenario/population.py's crossBorderOD attribute
    # and with the location synthesis.population.spatial.locations assigns.
    df_trips_noncb = df_trips[~df_trips["is_crossing_the_border"].astype("boolean").fillna(False).astype(bool)].copy()
    df_trips_noncb["destination_country_raw"] = None
    df_trips_noncb["interview_geometry_point"] = None
    df_trips_noncb["interview_point_id"]       = None

    is_cb_person = df_persons["is_crossing_the_border"].astype("boolean").fillna(False).astype(bool)
    is_cb_person &= ~df_persons["is_truck_driver"].astype("boolean").fillna(False).astype(bool)
    is_cb_person &= ~df_persons["is_outside_of_switzerland"].astype("boolean").fillna(False).astype(bool)
    is_cb_person &= df_persons["cross_border_person_id"].notna()

    df_cb_persons = df_persons.loc[is_cb_person, ["person_id"] + BORDER_MATCH_COLUMNS].copy()

    final_columns = [
        "person_id", "mz_person_id", "trip_id", "trip_index",
        "departure_time", "arrival_time",
        "preceding_purpose",
        "following_purpose",
        "trip_duration",
        "mode",
        "destination_country_raw", "interview_geometry_point", "interview_point_id",
    ]

    if len(df_cb_persons) == 0:
        return df_trips_noncb[final_columns]

    # Direction: 50% leaving Switzerland (home -> border), 50% entering (border -> home)
    is_leaving = rng.random_sample(size=len(df_cb_persons)) < 0.5

    # Sample departure/arrival times from an already existing (non-border) trip
    sampled_times = df_trips_noncb[["departure_time", "arrival_time"]].sample(
        n=len(df_cb_persons), replace=True, random_state=rng.randint(0, 2**31 - 1)
    ).reset_index(drop=True)

    df_trips_cb = pd.DataFrame({
        "person_id": df_cb_persons["person_id"].values,
        "mz_person_id": df_cb_persons["cross_border_person_id"].values,
        "trip_id": 1,
        "trip_index": 0,
        "departure_time": sampled_times["departure_time"].values,
        "arrival_time": sampled_times["arrival_time"].values,
        "preceding_purpose": np.where(is_leaving, "home", "border"),
        "following_purpose": np.where(is_leaving, "border", "home"),
        "mode": df_cb_persons["border_crossing_trip_mode"].values,
        "destination_country_raw": df_cb_persons["destination_country_raw"].values,
        "interview_geometry_point": df_cb_persons["interview_geometry_point"].values,
        "interview_point_id": df_cb_persons["interview_point_id"].values,
    })
    df_trips_cb["trip_duration"] = df_trips_cb["arrival_time"] - df_trips_cb["departure_time"]

    df_trips = pd.concat([df_trips_noncb, df_trips_cb], ignore_index=True, sort=False)

    return df_trips[final_columns]
