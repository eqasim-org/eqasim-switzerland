import numpy as np
import pandas as pd

import data.constants as c

"""
This stage attaches all trip relevant information to the synthetic population.
"""


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.microcensus.trips")


def execute(context):
    df_persons = context.stage("synthesis.population.enriched")[[
        "person_id", "mz_person_id", "age"
    ]]

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy=True)[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "following_purpose"]

    # Assume the preceeding purpose for all trips is home
    df_trips["preceding_purpose"] = df_trips["following_purpose"].shift(1)
    df_trips.loc[df_trips["trip_id"] == 1, "preceding_purpose"] = "home"

    df_trips = pd.merge(df_persons, df_trips, on="mz_person_id")

    # Children do not have any trips from the microcensus
    f = np.isnan(df_trips["mz_person_id"])
    assert ((df_trips[f]["age"] > c.MZ_AGE_THRESHOLD).all())

    # We deliberately delete them here, since other persons also may not have any
    # trips. May be improved later. TODO
    df_trips = df_trips[~f]

    df_trips.loc[:, "travel_time"] = df_trips.loc[:, "arrival_time"] - df_trips.loc[:, "departure_time"]

    df_trips = df_trips[["person_id", "trip_id",
                         "departure_time", "arrival_time",
                         "travel_time", "mode",
                         "preceding_purpose", "following_purpose"]].sort_values(by=["person_id", "trip_id"])

    # Diversify departure times
    counts = df_trips[["person_id", "trip_id"]].groupby("person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["person_id", "departure_time"]]
                .groupby("person_id")
                .min()
                .reset_index()["departure_time"]
                .values)

    # If first departure time is just 5min after midnight, we only add a deviation of 5min
    interval = np.minimum(1800.0, interval)

    offset = np.random.random(size=(len(counts),)) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"] += offset
    df_trips["departure_time"] = np.round(df_trips["departure_time"])
    df_trips["arrival_time"] = np.round(df_trips["arrival_time"])
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]

    # Define trip index
    df_trips = df_trips.sort_values(by=["person_id", "trip_id"])
    df_count = df_trips.groupby("person_id").size().reset_index(name="count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])

    return df_trips[[
        "person_id", "trip_index",
        "departure_time", "arrival_time",
        "preceding_purpose",
        "following_purpose",
        # "is_first_trip", "is_last_trip",
        "trip_duration",
        # "activity_duration",
        "mode"
    ]]
