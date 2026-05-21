import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.21.trips")
    context.stage("data.microcensus.21.persons")


def execute(context):
    df_trips = context.stage("data.microcensus.21.trips")[0].copy()
    df_persons = context.stage("data.microcensus.21.persons").copy()

    commutes = {}

    for primary_purpose in ["work", "education"]:
        # Find the maximum work duration per person
        commuters = list(set(df_trips[(df_trips["purpose"]==primary_purpose) | (df_trips["origin_purpose"]==primary_purpose)]["person_id"].values))

        df_max_time_per_person = (df_trips[(df_trips["purpose"] == primary_purpose)][["person_id", "activity_duration"]]
                                  .groupby("person_id")
                                  .max()
                                  .reset_index())

        # Find the trips with the maximum duration
        df_commute = pd.merge(df_trips[df_trips["purpose"] == primary_purpose][["person_id", "trip_id",
                                                                       "mode", "activity_duration",
                                                                       "destination_x", "destination_y"]],
                              df_max_time_per_person,
                              on=["person_id", "activity_duration"]
                              ).groupby("person_id").first().reset_index()[["person_id", "trip_id",
                                                                            "mode", "activity_duration",
                                                                            "destination_x", "destination_y"]]
        
        # Rename columns
        df_commute.columns = ["person_id", "commute_trip_id",
                              "commute_mode", "commute_activity_duration",
                              "destination_x", "destination_y"]

        # Find the commute distance
        df_commute = pd.merge(df_commute,
                              df_persons[["person_id", "home_x", "home_y"]],
                              on="person_id")

        df_commute["commute_home_distance"] = np.sqrt((df_commute["home_x"] - df_commute["destination_x"]) ** 2 +
                                                      (df_commute["home_y"] - df_commute["destination_y"]) ** 2)

        # Rename and clean columns
        df_commute["commute_x"] = df_commute["destination_x"]
        df_commute["commute_y"] = df_commute["destination_y"]
        df_commute = df_commute[["person_id", "commute_trip_id", "commute_mode", "commute_home_distance",
                                 "commute_activity_duration", "commute_x", "commute_y"]]
        
        # Some people might have primary_purpose only as origin_purpose of the first trip
        # Let's add their info
        remaining_commuters = [x for x in commuters if not x in df_commute["person_id"].values]
        
        df_commute_remaining = df_trips[(df_trips["origin_purpose"] == primary_purpose) & (df_trips["person_id"].isin(remaining_commuters))][
            ["person_id", "trip_id",
             "mode", "departure_time",
             "origin_x", "origin_y"]
            ].groupby("person_id").first().reset_index()[["person_id", "trip_id",
                "mode", "departure_time",
                "origin_x", "origin_y"]]
        
        df_commute_remaining.columns = ["person_id", "commute_trip_id",
                              "commute_mode", "commute_activity_duration",
                              "origin_x", "origin_y"]
        
        df_commute_remaining = pd.merge(df_commute_remaining,
                              df_persons[["person_id", "home_x", "home_y"]],
                              on="person_id")
        
        df_commute_remaining["commute_home_distance"] = np.sqrt((df_commute_remaining["home_x"] - df_commute_remaining["origin_x"]) ** 2 +
                                                      (df_commute_remaining["home_y"] - df_commute_remaining["origin_y"]) ** 2)
        
        df_commute_remaining["commute_x"] = df_commute_remaining["origin_x"]
        df_commute_remaining["commute_y"] = df_commute_remaining["origin_y"]
        df_commute_remaining = df_commute_remaining[["person_id", "commute_trip_id", "commute_mode", "commute_home_distance",
                                 "commute_activity_duration", "commute_x", "commute_y"]]
        
        df_commute = pd.concat([df_commute, df_commute_remaining])

        commutes.update({primary_purpose: df_commute})

    return commutes
