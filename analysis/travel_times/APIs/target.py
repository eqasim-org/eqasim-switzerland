
from turtle import pd
import logging
import os

logger = logging.getLogger("synpp")

def configure(context):
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

    context.config("travel_times_from", default="tomtom")
    if not context.config("travel_times_from").lower() in ["google", "tomtom", "mapbox"]:
        logger.warning("Invalid value for 'travel_times_from'. Defaulting to 'tomtom'.")
    

    if context.config("travel_times_from").lower() == "google":
        context.stage("analysis.travel_times.APIs.travel_times_google", alias="travel_times")
    elif context.config("travel_times_from").lower() == "mapbox":
        context.stage("analysis.travel_times.APIs.travel_times_mapbox", alias="travel_times")
    else:
        context.stage("analysis.travel_times.APIs.travel_times_tomtom", alias="travel_times")
    

def execute(context):    
    df = context.stage("travel_times")
    
    # make sure we do not use very low distance trips
    df = df[df["euclidean_distance_km"] > 2].reset_index(drop=True)

    # convert to required format
    df["travel_time"] = (df["travel_time_min"] * 60).astype(int)  # convert to seconds
    df["traveled_distance"] = (df["distance_km"] * 1000).astype(int)  # convert to meters
    df = df[["identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time","travel_time","traveled_distance"]]

    # save file
    api = context.config("travel_times_from").lower().replace("all","tomtom")
    path_to_output = os.path.join(context.config("output_path"), 
                                context.config("output_id"), 
                                context.config("simulation_directory"),
                                "travel_times_"+api,
                                f"calibration_target_travel_times.csv") 
    os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
    
    df.to_csv(path_to_output, index=False, sep=",")
    logger.info(f"Saved calibration target travel times to {path_to_output}")
    
    return path_to_output
    