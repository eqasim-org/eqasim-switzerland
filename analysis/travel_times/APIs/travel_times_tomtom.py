import json
import pandas as pd
import os
import logging
from data.spatial.utils import convert_crs

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.travel_times.APIs.get_from_tomtom")

def execute(context):
    path_to_travel_times = context.stage("analysis.travel_times.APIs.get_from_tomtom")

    # load the json file with travel times
    with open(path_to_travel_times, 'r') as f:
        travel_times_data = json.load(f)
    
    # get the data into a dataframe
    records = []
    for identifier, data in travel_times_data.items():
        if data["route_info"] is None:
            continue
        record = {
            'identifier': identifier,            
            'distance_km': data["route_info"]['summary']['lengthInMeters'] / 1000,
            'travel_time_min': data["route_info"]['summary']['historicTrafficTravelTimeInSeconds'] / 60,
            'departure_time': data["departure_time"],
            'origin_x': data["origin_x"],
            'origin_y': data["origin_y"],
            'destination_x': data["destination_x"],
            'destination_y': data["destination_y"],
        }
        records.append(record)
    
    # create dataframe
    df = pd.DataFrame.from_records(records)

    # compute euclidean distance
    df["origin_x"], df["origin_y"] = convert_crs( df["origin_x"].values, 
                                                df["origin_y"].values, 
                                                original_crs="EPSG:4326", 
                                                target_crs="EPSG:2056")
    
    df["destination_x"], df["destination_y"] = convert_crs( df["destination_x"].values, 
                                                            df["destination_y"].values, 
                                                            original_crs="EPSG:4326", 
                                                            target_crs="EPSG:2056")
    df["euclidean_distance_km"] = (( (df["destination_x"] - df["origin_x"])**2 + 
                                     (df["destination_y"] - df["origin_y"])**2 ) ** 0.5 ) / 1000
    
    return df[['identifier', 'distance_km', 'travel_time_min', 'departure_time', 'euclidean_distance_km',
               'origin_x', 'origin_y', 'destination_x', 'destination_y']]