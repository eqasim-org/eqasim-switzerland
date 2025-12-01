import os
import pandas as pd
import numpy as np
from mode_choice.dmc_defaults import Defaults
import matsim.runtime.eqasim as eqasim
import logging

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.trips.get_skim_matrices")

    context.config("walk_speed_m_per_s", default=Defaults.DEFAULT_WALK_SPEED_M_PER_S)
    context.config("walk_distance_factor", default=Defaults.DEFAULT_WALK_DISTANCE_FACTOR)  


def pt_variables(context, df):
    
    matrices = context.stage("mode_choice.trips.get_skim_matrices").copy()[
        ["origin_zone", "destination_zone",
         "in_vehicle_time_min", "access_egress_time_min",
         "waiting_time_min", "number_of_line_switches", "networkDistance"]
    ].rename(columns = {
        "number_of_line_switches": "transfers",
        "networkDistance": "distance_km"
    })
    
    trips = df.merge(matrices, on = ["origin_zone", "destination_zone"], how = "left")

    euclidean_distance = lambda x, y: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    euclidean_distance_trips = lambda df2: euclidean_distance((df2["origin_x"], df2["origin_y"]), (df2["destination_x"], df2["destination_y"]))

    mask_in_study_area = trips["origin_zone"].notna() & trips["destination_zone"].notna()

    # For the trips not in the study area, the travel times is not reported in the matrix
    # Fill the blank entries with 0 or values as if the trip was done by foot
    # TODO how to deal with external traffic? 
    trips.loc[~mask_in_study_area, "transfers"]              = 0
    trips.loc[~mask_in_study_area, "distance_km"]            = trips.apply(euclidean_distance_trips, axis = 1) * 1e-3
    trips.loc[~mask_in_study_area, "in_vehicle_time_min"]    = trips.apply(euclidean_distance_trips, axis = 1) / context.config("walk_speed_m_per_s") / 60 
    trips.loc[~mask_in_study_area, "waiting_time_min"]       = 0
    trips.loc[~mask_in_study_area, "access_egress_time_min"] = 0

    trips = trips[
        ["person_id", "trip_id",
         "in_vehicle_time_min", "access_egress_time_min",
         "waiting_time_min", "transfers", "distance_km"]
    ]

    return trips


# TODO adapt the code to use skim matrices
def execute(context):    
    trips  = context.stage("mode_choice.trips.prepare_trips").copy()[
        ["person_id", "trip_id",
         "origin_x", "origin_y", 
         "destination_x", "destination_y", 
         "departure_time",
         "origin_zone", "destination_zone"]
        ].copy()
    
    matrices = context.stage("mode_choice.trips.get_skim_matrices").copy()[
        ["origin_zone", "destination_zone",
         "in_vehicle_time_min", "access_egress_time_min",
         "waiting_time_min", "number_of_line_switches", "networkDistance"]
    ].rename(columns = {
        "number_of_line_switches": "transfers",
        "networkDistance": "distance_km"
    })
    
    trips = trips.merge(matrices, on = ["origin_zone", "destination_zone"], how = "left")

    euclidean_distance = lambda x, y: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    euclidean_distance_trips = lambda df: euclidean_distance((df["origin_x"], df["origin_y"]), (df["destination_x"], df["destination_y"]))

    mask_in_study_area = trips["origin_zone"].notna() & trips["destination_zone"].notna()

    # For the trips not in the study area, the travel times is not reported in the matrix
    # Fill the blank entries with 0 or values as if the trip was done by foot
    # TODO how to deal with external traffic? 
    trips.loc[~mask_in_study_area, "transfers"]              = 0
    trips.loc[~mask_in_study_area, "distance_km"]            = trips.apply(euclidean_distance_trips, axis = 1) * 1e-3
    trips.loc[~mask_in_study_area, "in_vehicle_time_min"]    = trips.apply(euclidean_distance_trips, axis = 1) / context.config("walk_speed_m_per_s") / 60 
    trips.loc[~mask_in_study_area, "waiting_time_min"]       = 0
    trips.loc[~mask_in_study_area, "access_egress_time_min"] = 0

    trips = trips[
        ["person_id", "trip_id",
         "in_vehicle_time_min", "access_egress_time_min",
         "waiting_time_min", "transfers", "distance_km"]
    ]

    return trips