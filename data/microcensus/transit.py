import numpy as np
import pandas as pd


def configure(context):
    context.config("data_path")
    context.stage("data.microcensus.trips")

def execute(context):
    # Load data
    data_path = context.config("data_path")
    df_stages = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")

    # Filter stages in pt trips
    df_trips = context.stage("data.microcensus.trips")
    df_trips = df_trips[df_trips["mode_detailed"] == "pt"]
    df_trips = df_trips[["person_id", "trip_id", "departure_time", "arrival_time"]]
    df_trips = df_trips.rename({ "departure_time" : "trip_departure_time", "arrival_time" : "trip_arrival_time"}, axis = 1)
    df_trips["trip_travel_time"] = df_trips["trip_arrival_time"] - df_trips["trip_departure_time"]

    df_stages["person_id"] = df_stages["HHNR"]
    df_stages["trip_id"] = df_stages["WEGNR"]
    df_stages["stage_id"] = df_stages["ETNR"]

    df_stages = pd.merge(df_stages, df_trips, on = ["person_id", "trip_id"], how = "inner")

    # Cleaning
    df_stages["departure_time"] = df_stages["f51100"] * 60.0
    df_stages["arrival_time"] = df_stages["f51400"] * 60.0
    df_stages["travel_time"] = df_stages["arrival_time"] - df_stages["departure_time"]

    df_stages["mode"] = "other"
    df_stages.loc[df_stages["f51300"] == 1, "mode"] = "walk"
    df_stages.loc[df_stages["f51300"] == 9, "mode"] = "rail"
    df_stages.loc[df_stages["f51300"] == 10, "mode"] = "bus"
    df_stages.loc[df_stages["f51300"] == 11, "mode"] = "bus"
    df_stages.loc[df_stages["f51300"] == 12, "mode"] = "tram"

    df_stages["routed_distance"] = df_stages["rdist"] * 1000.0
    df_stages["euclidean_distance"] = df_stages["ldist"] * 1000.0
    #df_stages.loc[df_stages["euclidean_distance"] < 0, "euclidean_distance"] = np.nan

    # Attention! Euclidean distance may be zero if there is no geo data in MZ.

    # Number of stages
    df_count = df_stages[["person_id", "trip_id"]].groupby(["person_id", "trip_id"]).size().reset_index(name = "number_of_stages")
    df_stages = pd.merge(df_stages, df_count, on = ["person_id", "trip_id"])

    # IDs
    df_ids = df_stages[["person_id", "trip_id", "number_of_stages"]].drop_duplicates()
    df_stages["stage_id"] = np.hstack([np.arange(k) + 1 for k in df_ids["number_of_stages"].values])

    # Flags
    df_stages["is_vehicular"] = df_stages["mode"].isin(["rail", "bus", "tram"])
    df_stages["is_first_stage"] = df_stages["stage_id"] == 1
    df_stages["is_last_stage"] = df_stages["stage_id"] == df_stages["number_of_stages"]

    df_stages = df_stages[[
        "person_id", "trip_id", "stage_id",
        "departure_time", "arrival_time", "travel_time",
        "trip_departure_time", "trip_arrival_time", "mode",
        "is_vehicular", "number_of_stages", "is_first_stage", "is_last_stage",
        "routed_distance", "euclidean_distance"
    ]]

    # Construct times
    f_vehicle = df_stages["is_vehicular"]
    f_access_egress = ~f_vehicle & (df_stages["is_first_stage"] | df_stages["is_last_stage"])
    f_transfer = ~f_vehicle & ~(df_stages["is_first_stage"] | df_stages["is_last_stage"])

    df_stages.loc[f_access_egress, "access_egress_time"] = df_stages.loc[f_access_egress, "travel_time"]
    df_stages.loc[f_transfer, "transfer_time"] = df_stages.loc[f_transfer, "travel_time"]
    df_stages.loc[f_vehicle, "in_vehicle_time"] = df_stages.loc[f_vehicle, "travel_time"]

    # Construct distances
    df_stages.loc[f_access_egress, "access_egress_routed_distance"] = df_stages.loc[f_access_egress, "routed_distance"]
    df_stages.loc[f_transfer, "transfer_routed_distance"] = df_stages.loc[f_transfer, "routed_distance"]
    df_stages.loc[f_vehicle, "in_vehicle_routed_distance"] = df_stages.loc[f_vehicle, "routed_distance"]
    df_stages.loc[f_access_egress, "access_egress_euclidean_distance"] = df_stages.loc[f_access_egress, "euclidean_distance"]
    df_stages.loc[f_transfer, "transfer_euclidean_distance"] = df_stages.loc[f_transfer, "euclidean_distance"]
    df_stages.loc[f_vehicle, "in_vehicle_euclidean_distance"] = df_stages.loc[f_vehicle, "euclidean_distance"]

    # Construct first waiting time
    df_arrival = df_stages[["person_id", "trip_id", "stage_id", "arrival_time", "is_vehicular"]]
    df_arrival = df_arrival[(df_arrival["stage_id"] == 1) & ~df_arrival["is_vehicular"]]
    del df_arrival["stage_id"]
    del df_arrival["is_vehicular"]

    df_departure = df_stages[["person_id", "trip_id", "stage_id", "departure_time", "is_vehicular"]]
    df_departure = df_departure[(df_departure["stage_id"] == 2) & df_departure["is_vehicular"]]
    del df_departure["stage_id"]
    del df_departure["is_vehicular"]

    df_first = pd.merge(df_arrival, df_departure, how = "inner", on = ["person_id", "trip_id"])
    df_first["first_waiting_time"] = df_first["departure_time"] - df_first["arrival_time"]
    df_first = df_first[["person_id", "trip_id", "first_waiting_time"]]

    # Aggregate trips
    df_aggregated = df_stages[[
        "person_id", "trip_id", "access_egress_time", "transfer_time", "in_vehicle_time",
        "access_egress_routed_distance", "transfer_routed_distance", "in_vehicle_routed_distance",
        "access_egress_euclidean_distance", "transfer_euclidean_distance", "in_vehicle_euclidean_distance",
        "is_vehicular"
    ]].groupby(["person_id", "trip_id"]).aggregate({
        "access_egress_time" : "sum",
        "transfer_time" : "sum",
        "in_vehicle_time" : "sum",
        "access_egress_routed_distance" : "sum",
        "transfer_routed_distance" : "sum",
        "in_vehicle_routed_distance" : "sum",
        "access_egress_euclidean_distance" : "sum",
        "transfer_euclidean_distance" : "sum",
        "in_vehicle_euclidean_distance" : "sum",
        "is_vehicular" : "sum"
    }).reset_index()

    df_aggregated["line_switches"] = np.maximum(0, df_aggregated["is_vehicular"])
    del df_aggregated["is_vehicular"]

    df_trips = pd.merge(df_trips, df_aggregated)
    df_trips["aggregated_time"] = df_trips["access_egress_time"] + df_trips["transfer_time"] + df_trips["in_vehicle_time"]
    df_trips["waiting_time"] = df_trips["trip_travel_time"] - df_trips["aggregated_time"]
    del df_trips["aggregated_time"]

    # Filter out remaining information
    del df_trips["trip_travel_time"]
    del df_trips["trip_departure_time"]
    del df_trips["trip_arrival_time"]

    # Implement first waiting time
    df_trips = pd.merge(df_trips, df_first, how = "left", on = ["person_id", "trip_id"])
    df_trips["first_waiting_time"] = df_trips["first_waiting_time"].fillna(0.0)
    df_trips["waiting_time"] -= df_trips["first_waiting_time"]

    return df_trips
