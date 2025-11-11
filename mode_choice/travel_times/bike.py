import pandas as pd
import numpy as np
import os

def config(context):
    context.stage("mode_choice.prepare_trips")
    context.config("bike_speed_m_per_s", default=4.0)  # average biking speed ~14.4 km/h
    context.config("bike_distance_factor", default=1.4)  # factor to account for indirect biking paths


def excute(context):
    # read prepared trips
    trips = context.stage("mode_choice.prepare_trips")[
        ["person_id","trip_index","crowfly_distance"]
    ].copy()

    # calculate biking distance
    bike_distance_factor = context.config("bike_distance_factor")
    trips["bike_distance"] = trips["crowfly_distance"] * bike_distance_factor
    
    # calculate biking travel time in seconds
    bike_speed = context.config("bike_speed_m_per_s")
    trips["bike_travel_time"] = trips["bike_distance"] / bike_speed
    
    return trips

    
    
    

    