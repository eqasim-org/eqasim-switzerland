import pandas as pd
import numpy as np
import os
from mode_choice.dmc_defaults import Defaults

def configure(context):
    context.stage("mode_choice.dmc_defaults")
    context.stage("mode_choice.trips.prepare_trips")
    context.config("walk_speed_m_per_s", default=Defaults.DEFAULT_WALK_SPEED_M_PER_S)  # average walking speed ~5 km/h
    context.config("walk_distance_factor", default=Defaults.DEFAULT_WALK_DISTANCE_FACTOR)  # factor to account for indirect walking paths

def walk_travel_time(context, distance_km):
    walk_speed_m_per_s = context.config("walk_speed_m_per_s")
    travel_time_min = (distance_km * 1e3 / walk_speed_m_per_s) / 60
    return travel_time_min

def walk_distance(context, euclidean_distance_km):
    walk_distance_factor = context.config("walk_distance_factor")
    adjusted_distance_km = euclidean_distance_km * walk_distance_factor
    return adjusted_distance_km

def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_id","euclidean_distance_km"]
    ].copy()

    # calculate walking distance
    trips["distance_km"] = walk_distance(context, trips["euclidean_distance_km"])
    
    # calculate walking travel time in seconds    
    trips["travel_time_min"] = walk_travel_time(context, trips["distance_km"])

    return trips[["person_id","trip_id","travel_time_min","distance_km"]]

    
    
    

    