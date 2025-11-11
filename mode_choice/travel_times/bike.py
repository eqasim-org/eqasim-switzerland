import pandas as pd
import numpy as np
import os

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.config("bike_speed_m_per_s", default=4.0)  # average biking speed ~14.4 km/h
    context.config("bike_distance_factor", default=1.4)  # factor to account for indirect biking paths


def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.prepare_trips")[
        ["person_id","trip_index","crowfly_distance"]
    ].copy()

    # calculate biking distance
    bike_distance_factor = context.config("bike_distance_factor")
    trips["distance_km"] = trips["crowfly_distance"] * bike_distance_factor * 1e-3
    
    # calculate biking travel time in seconds
    bike_speed = context.config("bike_speed_m_per_s")
    trips["travel_time_min"] = (trips["bike_distance_km"]*1e3 / bike_speed) / 60
    
    return trips

    
    
    

    