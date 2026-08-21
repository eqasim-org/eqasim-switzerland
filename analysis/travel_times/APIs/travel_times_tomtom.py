import pandas as pd
import logging
from data.spatial.utils import convert_crs

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.travel_times.APIs.get_from_tomtom")

def execute(context):
    summary_path = context.stage("analysis.travel_times.APIs.get_from_tomtom")
    df = pd.read_csv(summary_path, dtype={"identifier": str})

    if df.empty:
        df["euclidean_distance_km"] = pd.Series(dtype=float)
        return df[['identifier', 'distance_km', 'travel_time_min', 'departure_time',
                   'euclidean_distance_km', 'origin_x', 'origin_y', 'destination_x',
                   'destination_y']]

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
