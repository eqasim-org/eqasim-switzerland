import numpy as np
import pandas as pd

"""
Transforms the synthetic trip table into a synthetic activity table.
"""


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")


def execute(context):
    df_trips = pd.DataFrame(context.stage("synthesis.population.trips"), copy=True)
    df_trips.loc[:, "previous_trip_index"] = df_trips.loc[:, "trip_index"] - 1

    df_activities = pd.merge(
        df_trips, df_trips, left_on=["person_id", "previous_trip_index"], right_on=["person_id", "trip_index"],
        suffixes=["_following_trip", "_previous_trip"], how="left"
    )

    df_activities.loc[:, "start_time"] = df_activities.loc[:, "arrival_time_previous_trip"]
    df_activities.loc[:, "end_time"] = df_activities.loc[:, "departure_time_following_trip"]
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "following_purpose_previous_trip"]
    df_activities.loc[:, "activity_index"] = df_activities.loc[:, "trip_index_following_trip"]
    df_activities.loc[:, "is_last"] = False

    # We assume that the plans start at home
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "purpose"].fillna("home")

    # We're still missing the last activity in the chain.
    df_last = df_activities.sort_values(by=["person_id", "activity_index"]).groupby("person_id").last().reset_index()
    df_last.loc[:, "purpose"] = df_last.loc[:, "following_purpose_following_trip"]
    df_last.loc[:, "start_time"] = df_last.loc[:, "arrival_time_following_trip"]
    df_last.loc[:, "end_time"] = np.nan
    df_last.loc[:, "activity_index"] += 1
    df_last.loc[:, "is_last"] = True

    df_activities = pd.concat([df_activities, df_last])

    # We're still missing activities for people who don't have a any trips
    df_persons = context.stage("synthesis.population.enriched")[["person_id"]]

    missing_ids = set(np.unique(df_persons["person_id"])) - set(np.unique(df_activities["person_id"]))
    print("Found %d persons without activities" % len(missing_ids))

    df_missing = pd.DataFrame.from_records([
        (person_id, 1, "home", True) for person_id in missing_ids
    ], columns=["person_id", "activity_index", "purpose", "is_last"])

    df_activities = pd.concat([df_activities, df_missing], sort=True)
    assert (len(np.unique(df_persons["person_id"])) == len(np.unique(df_activities["person_id"])))

    # Some cleanup
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_activities.loc[:, "duration"] = df_activities.loc[:, "end_time"] - df_activities.loc[:, "start_time"]

    df_activities = df_activities[[
        "person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last"
    ]]

    return df_activities
