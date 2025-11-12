import pandas as pd
import numpy as np
import os

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.config("walk_speed_m_per_s", default=1.3)  # average walking speed ~5 km/h
    context.config("walk_distance_factor", default=1.3)  # factor to account for indirect walking paths


def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_index","crowfly_distance"]
    ].copy()

    # Euclidean distance in km
    trips["Euclidean_distance_km"] = trips["crowfly_distance"] * 1e-3

    # calculate walking distance
    walk_distance_factor = context.config("walk_distance_factor")
    trips["distance_km"] = trips["Euclidean_distance_km"] * walk_distance_factor
    
    # calculate walking travel time in seconds
    walk_speed = context.config("walk_speed_m_per_s")
    trips["travel_time_min"] = (trips["distance_km"]*1e3 / walk_speed) / 60

    return trips[["person_id","trip_index","travel_time_min","distance_km"]]

    
    
    

    