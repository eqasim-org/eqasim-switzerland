import pandas as pd
import numpy as np
import data.constants as c

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")

def execute(context):
    df_trips = context.stage("data.microcensus.trips")
    df_persons = context.stage("data.microcensus.persons")

    df_primary_commute = []

    for primary_purpose in ["work", "education"]:
        # Find the maximum activity duration per person
        df_max_time_per_person = df_trips[
            df_trips["purpose"] == primary_purpose
        ][[
            "person_id", "activity_duration"
        ]].groupby("person_id").max().reset_index()

        # Find the trips with the maximum duration
        df_commute = pd.merge(
            df_trips[df_trips["purpose"] == primary_purpose][[
                "person_id", "trip_id", "mode", "activity_duration", "destination_x", "destination_y"
            ]],
            df_max_time_per_person,
            on = ["person_id", "activity_duration"]
        ).groupby("person_id").first().reset_index()[[
            "person_id", "trip_id", "mode", "activity_duration", "destination_x", "destination_y"
        ]]

        df_commute.columns = [
            "person_id", "commute_trip_id", "commute_mode", "commute_activity_duration",
            "destination_x", "destination_y"]

        # Find the commute distance
        df_commute = pd.merge(
            df_commute,
            df_persons[["person_id", "home_x", "home_y"]],
            on = "person_id")

        df_commute["commute_home_distance"] = np.sqrt(
            (df_commute["home_x"] - df_commute["destination_x"])**2 + (df_commute["home_y"] - df_commute["destination_y"])**2
        )

        df_commute["commute_x"] = df_commute["destination_x"]
        df_commute["commute_y"] = df_commute["destination_y"]

        df_commute = df_commute[["person_id", "commute_trip_id", "commute_mode", "commute_home_distance", "commute_activity_duration", "commute_x", "commute_y"]]
        df_commute.loc[:, "commute_purpose"] = primary_purpose
        df_primary_commute.append(df_commute)

    df_commute = pd.concat(df_primary_commute)
    df_commute["commute_purpose"] = df_commute["commute_purpose"].astype("category")

    # Find the one with the longest duration, so we only have one commute purpose
    df_commute = df_commute.sort_values("commute_activity_duration", ascending = False).drop_duplicates("person_id")
    return df_commute
