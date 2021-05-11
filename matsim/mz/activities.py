import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def execute(context):
    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy=True)
    df_persons = context.stage("data.microcensus.persons")
    df_trips = pd.merge(df_trips, df_persons[["person_id", "home_x", "home_y"]], how="left")

    df_trips.loc[:, "previous_trip_id"] = df_trips.loc[:, "trip_id"] - 1

    df_activities = pd.merge(
        df_trips, df_trips, left_on=["person_id", "previous_trip_id"], right_on=["person_id", "trip_id"],
        suffixes=["_following_trip", "_previous_trip"], how="left"
    )

    df_activities.loc[:, "start_time"] = df_activities.loc[:, "arrival_time_previous_trip"]
    df_activities.loc[:, "end_time"] = df_activities.loc[:, "departure_time_following_trip"]
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "purpose_previous_trip"]
    df_activities.loc[:, "activity_id"] = df_activities.loc[:, "trip_id_following_trip"]
    df_activities.loc[:, "location_x"] = df_activities.loc[:, "destination_x_previous_trip"]
    df_activities.loc[:, "location_y"] = df_activities.loc[:, "destination_y_previous_trip"]
    df_activities.loc[:, "is_last"] = False

    # We assume that the plans start at home
    df_activities.loc[:, "purpose"] = df_activities.loc[:, "purpose"].fillna("home")

    f = df_activities["activity_id"] == 1
    df_activities.loc[f, "location_x"] = df_activities.loc[f, "home_x_following_trip"]
    df_activities.loc[f, "location_y"] = df_activities.loc[f, "home_y_following_trip"]

    # We're still missing the last activity in the chain.
    df_last = df_activities.sort_values(by=["person_id", "activity_id"]).groupby("person_id").last().reset_index()
    df_last.loc[:, "purpose"] = df_last.loc[:, "purpose_following_trip"]
    df_last.loc[:, "start_time"] = df_last.loc[:, "arrival_time_following_trip"]
    df_last.loc[:, "end_time"] = np.nan
    df_last.loc[:, "activity_id"] += 1
    df_last.loc[:, "location_x"] = df_last.loc[:, "destination_x_following_trip"]
    df_last.loc[:, "location_y"] = df_last.loc[:, "destination_y_following_trip"]
    df_last.loc[:, "is_last"] = True

    df_activities = pd.concat([df_activities, df_last])

    # Some cleanup
    df_activities = df_activities.sort_values(by=["person_id", "activity_id"])
    df_activities.loc[:, "duration"] = df_activities.loc[:, "end_time"] - df_activities.loc[:, "start_time"]

    df_activities = df_activities[[
        "person_id", "activity_id", "start_time", "end_time", "duration", "purpose", "is_last",
        "location_x", "location_y"
    ]]

    return df_activities
