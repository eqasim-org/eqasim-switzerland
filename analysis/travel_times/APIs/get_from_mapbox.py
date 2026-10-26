import os
import logging
from data.spatial.utils import convert_crs
import pandas as pd
import requests
import time
import datetime
import random
from analysis.travel_times.APIs.mapbox_storage import (
    ensure_mapbox_summary_csv,
    persist_mapbox_batch,
)
from analysis.travel_times.APIs.tomtom_storage import read_routed_ids
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("analysis.travel_times.trips.car_trips")
    context.config("mapbox_api_key", default = "")
    context.config("data_path")
    context.config("mapbox_travel_times_path", 
                   default = os.path.join(context.config("data_path"),
                                          "travel_times", 
                                          "mapbox_travel_times"))
    context.config(
        "mapbox_travel_times_backup_path",
        default=os.path.join(
            os.path.dirname(context.config("data_path")),
            "travel_times_backup",
            "mapbox",
        ),
    )
    context.config("num_mapbox_requests", default=0)
    context.config("mapbox_json_batch_size", default=100)

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
    departure_time_iso = dt.strftime('%Y-%m-%dT%H:%M')

    return departure_time_iso

def route_one_trip(origin_x, origin_y, destination_x, destination_y, departure_time, mapbox_api_key):
    """
    Call Mapbox Directions (driving-traffic) for a single trip.
    """
    base_url = f"https://api.mapbox.com/directions/v5/mapbox/driving-traffic/{origin_x},{origin_y};{destination_x},{destination_y}"
    
    departure_time_iso = convert_departure_time_to_iso(departure_time)

    params = {
        # Top-level distance and duration remain available without geometry.
        "overview": "false",
        "depart_at": departure_time_iso,
        "access_token": mapbox_api_key,
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
    except requests.RequestException as exc:
        logger.error("Mapbox API request failed: %s", exc)
        return None

    if response.status_code != 200:
        logger.error(
            "Mapbox API request failed with status code %s: %s", response.status_code, response.text
        )
        return None

    data = response.json()
    if data.get("code") != "Ok":
        logger.error("Mapbox API returned non-Ok code: %s", data.get("code"))
        return None

    routes = data.get("routes", [])
    if not routes:
        logger.error("Mapbox API returned no routes for request.")
        return None

    return {
        "route": routes[0],
        "waypoints": data.get("waypoints", []),
        "uuid": data.get("uuid"),
    }
    
def iter_routes_with_mapbox(df_trips, mapbox_api_key, max_requests):
    skipped = 0
    num_requests = 0
    for i,(index, row) in enumerate(df_trips.iterrows()):
        if num_requests >= max_requests:
            logger.info(f"\n\t Reached maximum number of Mapbox requests: {max_requests}. Stopping.")
            break

        identifier = row['identifier']
        if i % 100 == 0:
            logger.info(f"\t - Mapbox API: Routing trip {i}/{len(df_trips)}")
                                        
        route_info = route_one_trip(
            row['origin_x'],
            row['origin_y'],
            row['destination_x'],
            row['destination_y'],
            row['departure_time'],
            mapbox_api_key,
        )
        
        # If no route info, skip
        if route_info is None:
            logger.warning(f"\n\t Failed to route trip {identifier}")
            skipped += 1
            if skipped>20:
                logger.warning("\n\t Too many skipped requests to Mapbox API, stopping.")
                break
            continue
        else:
            num_requests += 1

        yield identifier, {
            'origin_x': row['origin_x'],
            'origin_y': row['origin_y'],
            'destination_x': row['destination_x'],
            'destination_y': row['destination_y'],
            'departure_time': row['departure_time'],
            'route_info': route_info
        }

        # To avoid hitting rate limits
        time.sleep(0.05)  


def route_with_mapbox(df_trips, mapbox_api_key, max_requests):
    routed_data = {}
    for identifier, data in iter_routes_with_mapbox(df_trips, mapbox_api_key, max_requests):
        routed_data[identifier] = data
    return routed_data


def execute(context):
    output_directory = context.config("mapbox_travel_times_path")
    backup_directory = context.config("mapbox_travel_times_backup_path")
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(backup_directory, exist_ok=True)

    summary_path = ensure_mapbox_summary_csv(
        output_directory,
        raw_directories=[output_directory, backup_directory],
        metadata_directory=backup_directory,
    )

    mapbox_api_key = context.config("mapbox_api_key")
    max_requests = context.config("num_mapbox_requests")
    if (mapbox_api_key == "") or (max_requests <= 0):
        logger.info("No Mapbox routing required; returning the compact CSV index.")
        return str(summary_path)

    # read trips
    df_trips = context.stage("analysis.travel_times.trips.car_trips")    
    logger.info(f"Loaded {len(df_trips)} car trips to be routed using Mapbox API.")

    # Convert coordinates from EPSG:2056 to EPSG:4326
    df_trips["origin_x"], df_trips["origin_y"] = convert_crs( df_trips["origin_x"].values, 
                                                              df_trips["origin_y"].values, 
                                                              original_crs="EPSG:2056", 
                                                              target_crs="EPSG:4326")
    df_trips["destination_x"], df_trips["destination_y"] = convert_crs( df_trips["destination_x"].values, 
                                                                        df_trips["destination_y"].values, 
                                                                        original_crs="EPSG:2056", 
                                                                        target_crs="EPSG:4326")
    logger.info("Converted coordinates from EPSG:2056 to EPSG:4326.")

    routed_ids = read_routed_ids(summary_path)
    logger.info(f"Found {len(routed_ids)} already routed trips.")

    # filter to remaining trips
    df_remaining = df_trips[~df_trips['identifier'].isin(routed_ids)]
    logger.info(f"Remaining trips to route: {len(df_remaining)}")

    if (not df_remaining.empty) and (mapbox_api_key != "") and (max_requests > 0):
        batch_size = max(1, int(context.config("mapbox_json_batch_size")))
        batch = {}
        for identifier, data in iter_routes_with_mapbox(
            df_remaining, mapbox_api_key, max_requests
        ):
            batch[identifier] = data
            if len(batch) >= batch_size:
                persist_mapbox_batch(output_directory, backup_directory, batch)
                batch = {}
        if batch:
            persist_mapbox_batch(output_directory, backup_directory, batch)

    return str(summary_path)
