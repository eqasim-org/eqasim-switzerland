import pandas as pd
import os
import requests
import time
import logging
import json
import datetime
import random
from data.spatial.utils import convert_crs
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.travel_times.trips.car_trips")
    context.config("tomtom_api_key", default = "")
    context.config("data_path")
    context.config("tomtom_travel_times_path", 
                   default = os.path.join(context.config("data_path"),
                                          "travel_times",
                                          "tomtom_travel_times"))
    context.config("num_tomtom_requests", default=int(1e10))

def convert_departure_time_to_iso(departure_time):
    """
    It generates a random weekday date in 2025 and combines it with the given departure time
    (in seconds after midnight) to create an ISO formatted datetime string.
    """
    # Generate a random weekday in 2025
    start_date = datetime.date(2025, 1, 1)
    while True:
        days_to_add = random.randint(0, 364)
        date = start_date + datetime.timedelta(days=days_to_add)
        if date.weekday() < 5:  # Monday to Friday
            break

    # Convert departure_time (seconds after midnight) to hours, minutes, seconds.
    # Some datasets contain values outside [0, 86400) (e.g. 86400 -> hour 24),
    # which is invalid for datetime.time(). Normalize to time-of-day.
    if pd.isna(departure_time):
        raise ValueError("departure_time is NaN")

    departure_seconds = int(round(float(departure_time)))
    departure_seconds = departure_seconds % (24 * 3600)

    hours = min(departure_seconds // 3600, 23)
    minutes = (departure_seconds % 3600) // 60
    seconds = departure_seconds % 60

    # Create datetime object
    dt = datetime.datetime.combine(date, datetime.time(hours, minutes, seconds))

    # Format to ISO string
    departure_time_iso = dt.isoformat()

    return departure_time_iso

def route_one_trip(origin_x, origin_y, destination_x, destination_y, departure_time, tomtom_api_key):
    # TomTom Routing API expects a contentType suffix (e.g. /json or /xml).
    base_url = "https://api.tomtom.com/routing/1/calculateRoute/{},{}:{},{}/json"
    
    # process departure time from seconds after midnight to ISO format
    departure_time_iso = convert_departure_time_to_iso(departure_time)
    
    # set up parameters
    params = {
        "key": tomtom_api_key,
        "departAt": departure_time_iso,
        "travelMode": "car",
        "computeTravelTimeFor": "all",
        "traffic": "true"
    }

    # make request
    url = base_url.format(origin_y, origin_x, destination_y, destination_x)
    response = requests.get(url, params=params) 
    if response.status_code == 200:   
        data = response.json()
        return data["routes"][0]
    else:
        logger.error(f"TomTom API request failed with status code {response.status_code}: {response.text}")
        return None
    
def route_with_tomtom(df_trips, tomtom_api_key, max_requests):
    routed_data = {}
    skipped = 0
    num_requests = 0
    for i,(index, row) in enumerate(df_trips.iterrows()):
        identifier = row['identifier']
        if i % 100 == 0:
            logger.info(f"\t - TomTom API: Routing trip {i}/{len(df_trips)}")
                                
        route_info = route_one_trip(row['origin_x'], row['origin_y'], row['destination_x'], row['destination_y'], row['departure_time'], tomtom_api_key)            
        
        # If no route info, skip
        if route_info is None:
            logger.warning(f"\n\t Failed to route trip {identifier}")
            skipped += 1
            if skipped>20:
                logger.warning("\n\t Too many skipped requests to TomTom API, stopping.")
                break
            continue
        else:
            num_requests += 1
            if num_requests >= max_requests:
                logger.info(f"\n\t Reached maximum number of TomTom requests: {max_requests}. Stopping.")
                break

        # store routed data
        routed_data[identifier] = {
            'origin_x': row['origin_x'],
            'origin_y': row['origin_y'],
            'destination_x': row['destination_x'],
            'destination_y': row['destination_y'],
            'departure_time': row['departure_time'],
            'route_info': route_info
        }

        # To avoid hitting rate limits
        time.sleep(0.1)  
    
    return routed_data

def execute(context):
    # prepare output path
    output_path = context.config("tomtom_travel_times_path")
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "microcensus_routed_trips.json")

    max_requests = context.config("num_tomtom_requests")
    tomtom_api_key = context.config("tomtom_api_key")
    if (tomtom_api_key != "") and (max_requests > 0):
        # read trips
        df_trips = context.stage("analysis.travel_times.trips.car_trips")    
        logger.info(f"Loaded {len(df_trips)} car trips to be routed using TomTom API.")

        # Convert coordinates to fro EPSG:2056 to EPSG:4326
        df_trips["origin_x"], df_trips["origin_y"] = convert_crs( df_trips["origin_x"].values, 
                                                                df_trips["origin_y"].values, 
                                                                original_crs="EPSG:2056", 
                                                                target_crs="EPSG:4326")
        df_trips["destination_x"], df_trips["destination_y"] = convert_crs( df_trips["destination_x"].values, 
                                                                            df_trips["destination_y"].values, 
                                                                            original_crs="EPSG:2056", 
                                                                            target_crs="EPSG:4326")
        logger.info("Converted coordinates from EPSG:2056 to EPSG:4326.")

        # check for existing routed data
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                existing_routed_data = json.load(f)

            routed_ids = set(existing_routed_data.keys())
            logger.info(f"Found {len(routed_ids)} already routed trips.")
        else:
            existing_routed_data = {}
            routed_ids = set()
            logger.info("No existing routed trips found.")

        # filter to remaining trips
        df_remaining = df_trips[~df_trips['identifier'].isin(routed_ids)]
        logger.info(f"Remaining trips to route: {len(df_remaining)}")

        if (not df_remaining.empty):
            # route remaining trips using tomtom
            new_routed_data = route_with_tomtom(df_remaining, tomtom_api_key, max_requests)
            
            # update existing data with new data
            existing_routed_data.update(new_routed_data)

            # save results as json
            with open(output_path.replace(".json", "_updated.json"), 'w') as f:
                json.dump(existing_routed_data, f, indent=4)

            # onces it is successful, replace old file
            os.replace(output_path.replace(".json", "_updated.json"), output_path)
            logger.info(f"Saved routed trips to {output_path}. Total routed trips: {len(existing_routed_data)}")

    return output_path