import pandas as pd
import numpy as np
import os

def config(context):
    context.stage("mode_choice.prepare_trips")
    context.config("walk_speed_m_per_s", default=1.3)  # average walking speed ~5 km/h
    context.config("walk_distance_factor", default=1.3)  # factor to account for indirect walking paths


def excute(context):
    # read prepared trips
    trips = context.stage("mode_choice.prepare_trips")[
        ["person_id","trip_index","crowfly_distance"]
    ].copy()

    # calculate walking distance
    walk_distance_factor = context.config("walk_distance_factor")
    trips["walk_distance"] = trips["crowfly_distance"] * walk_distance_factor
    
    # calculate walking travel time in seconds
    walk_speed = context.config("walk_speed_m_per_s")
    trips["walk_travel_time"] = trips["walk_distance"] / walk_speed
    
    return trips

    
    
    

    