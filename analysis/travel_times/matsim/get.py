from matsim.runtime.eqasim import run as run_eqasim
import pandas as pd
import logging 
import numpy as np

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.travel_times.matsim.route")

def execute(context):
    _, output_path = context.stage("analysis.travel_times.matsim.route")
    
    # read the file
    df = pd.read_csv(output_path)

    # convert times to minutes    
    df["travel_time_min"] = df["travel_time"] / 60

    # convert distances to km
    df["access_distance_km"] = df["access_distance"] / 1000
    df["egress_distance_km"] = df["egress_distance"] / 1000
    df["distance_km"] = df["travel_distance"] / 1000

    # Euclidean distance between origin and destination
    dx = df["origin_x"].astype(float) - df["destination_x"].astype(float)
    dy = df["origin_y"].astype(float) - df["destination_y"].astype(float)
    df["euclidean_distance_km"] = np.sqrt((dx**2 + dy**2)) / 1000

    return df[['identifier', 'distance_km', 'travel_time_min', 
               'access_distance_km', 'egress_distance_km', 
               'departure_time', 'euclidean_distance_km']]