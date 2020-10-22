import pandas as pd

import data.constants as c


def configure(context):
    context.stage("population.sociodemographics")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.commute")

def execute(context):
    df_persons = context.stage("population.sociodemographics")[[
        "person_id", "mz_person_id", "age"
    ]]

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy = True)[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "following_purpose"]

    df_trips = pd.merge(df_persons, df_trips, on = "mz_person_id")

    # Children do not have any trips from the microcensus
    f = np.isnan(df_trips["mz_person_id"])
    assert((df_trips[f]["age"] > c.MZ_AGE_THRESHOLD).all())

    # We deliberately delete them here, since other persons also may not have any
    # trips. May be improved later. TODO
    df_trips = df_trips[~f]

    df_trips.loc[:, "travel_time"] = df_trips.loc[:, "arrival_time"] - df_trips.loc[:, "departure_time"]

    # Impute commuting information
    df_commute = pd.DataFrame(context.stage("data.microcensus.commute"), copy = True)[["person_id", "commute_trip_id"]]
    df_commute.columns = ["mz_person_id", "commute_trip_id"]
    df_trips = pd.merge(df_trips, df_commute, on = "mz_person_id", how = "left")
    df_trips.loc[:, "is_commute"] = df_trips.loc[:, "trip_id"] == df_trips.loc[:, "commute_trip_id"]

    df_trips = df_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "travel_time", "mode", "following_purpose", "is_commute"
    ]].sort_values(by = ["person_id", "trip_id"])

    # Diversify departure times
    counts = df_trips[["person_id", "trip_id"]].groupby("person_id").size().reset_index(name = "count")["count"].values

    interval = df_trips[["person_id", "departure_time"]].groupby("person_id").min().reset_index()["departure_time"].values
    interval = np.minimum(1800.0, interval) # If first departure time is just 5min after midnight, we only add a deviation of 5min

    offset = np.random.random(size = (len(counts), )) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"] += offset
    df_trips["departure_time"] = np.round(df_trips["departure_time"])
    df_trips["arrival_time"] = np.round(df_trips["arrival_time"])

    return df_trips
