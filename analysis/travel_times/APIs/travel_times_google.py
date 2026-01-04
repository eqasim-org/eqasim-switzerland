import logging

logger = logging.getLogger("synpp")

def configure(context):
    # Path to the csv file containing the dataset of routed trips  
    context.stage("dmc.data.clean_routed_data")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")

def execute(context):
    # load data
    df = context.stage("dmc.data.clean_routed_data")

    # keep only car trips
    df_car = df[df["mode"]=="car"].reset_index(drop=True)
    logger.info(f"Filtered to {len(df_car)} car trips.")

    # remove weekend trips
    df_persons = context.stage("data.microcensus.persons")    
    weekend_persons = df_persons[~df_persons["weekend"]]['person_id'].unique()    
    df_car = df_car[~df_car['person_id'].isin(weekend_persons)]

    # identifier
    df_car["identifier"] = df_car["person_id"].astype(str) + "_" + df_car["trip_id"].astype(str)

    # travel times
    df_car["travel_time_min"] = df_car["travelTime_car"]/60.0
    df_car["distance_km"] = df_car["distance_car"]/1000.0

    # keep only relevant columns
    df_car_trips = df_car[["identifier", "travel_time_min", "distance_km"]]

    # to keep consistency with tomtom
    mz_trips , _ = context.stage("data.microcensus.trips")

    mz_trips["identifier"] = mz_trips["person_id"].astype(str) + "_" + mz_trips["trip_id"].astype(str)
    mz_trips["euclidean_distance_km"] = mz_trips["crowfly_distance"] / 1000.0

    mz_trips = mz_trips[['identifier', 'departure_time', 'euclidean_distance_km','origin_x', 'origin_y', 'destination_x', 'destination_y']]
    df_car_trips = df_car_trips.merge(mz_trips, on="identifier", how="left")
    
    # remove all nans
    df_car_trips = df_car_trips[df_car_trips.isna().sum(axis=1)==0].reset_index(drop=True)

    return df_car_trips