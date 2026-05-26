import os
import pandas as pd
import logging 
from shapely import vectorized

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.swiss_border")
    context.stage("analysis.travel_times.trips.build_highway_trips")
    context.stage("analysis.travel_times.trips.build_urban_trips")
    context.stage("analysis.travel_times.trips.build_geneva_trips")

def execute(context):
    logger.info("\t - Loading microcensus trip data")
    df_trips,filterout_ids = context.stage("data.microcensus.trips")
    
    ### filter weekend trips - currently disabled
    # df_persons = context.stage("data.microcensus.persons")    
    # # remove weekend trips    
    # filterout_ids = set(filterout_ids)
    # wekk_end_persons = df_persons[~df_persons["weekend"]]['person_id'].unique()
    # filterout_ids.update(wekk_end_persons)
    # df_trips = df_trips[~df_trips['person_id'].isin(filterout_ids)]

    # remove loop trips
    df_trips = df_trips[df_trips['crowfly_distance'] > 2000]  # keep trips longer than 2000m

    # keep only car trips
    car_trips = df_trips[df_trips['mode'] == 'car']

    # keep only within switzerland
    df_switzerland = context.stage("data.spatial.swiss_border").geometry.simplify(2000).iloc[0]
    ch_polygon = df_switzerland.buffer(-10_000)  # inward buffer of 10km
    inside_origin = vectorized.contains(ch_polygon, df_trips["origin_x"].values, df_trips["origin_y"].values)
    inside_destination = vectorized.contains(ch_polygon, df_trips["destination_x"].values, df_trips["destination_y"].values)
    df_trips = df_trips[inside_origin&inside_destination]

    # keep departure time between [0, 23]
    df_trips["departure_time"] = df_trips["departure_time"]%86400

    # prepare for eqasim
    car_trips["identifier"] = car_trips["person_id"].astype(str) + "_" + car_trips["trip_id"].astype(str)
    car_trips = car_trips[['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']]

    assert car_trips["identifier"].is_unique, "Identifiers are not unique in car trips!"
    assert car_trips[["origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]].notna().all().all(), "There are NaNs in car trips!"

    # load highway, urban, and geneva trips
    highway_trips = context.stage("analysis.travel_times.trips.build_highway_trips")
    urban_trips = context.stage("analysis.travel_times.trips.build_urban_trips")
    geneva_trips = context.stage("analysis.travel_times.trips.build_geneva_trips")
    car_trips = pd.concat([highway_trips, urban_trips, geneva_trips, car_trips], ignore_index=True)
    logger.info(f"\t - Total car trips after adding highway, urban, and geneva trips: {len(car_trips)}")

    # save to csv    
    # logger.info("\t - Saving car trips to temporary file")   
    # path_to_trips = os.path.join(context.path(), "car_trips.csv")
    # car_trips.to_csv(path_to_trips, index=False)
    # logger.info(f"\t - Car trips saved to {path_to_trips}")
    
    return car_trips