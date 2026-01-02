import logging

logger = logging.getLogger("synpp")

def configure(context):
    # Path to the csv file containing the dataset of routed trips  
    context.stage("dmc.data.clean_routed_data")
    context.stage("data.microcensus.persons")

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
    df["identifier"] = df["person_id"].astype(str) + "_" + df["trip_id"].astype(str)

    # travel times
    df["car_travel_time_min"] = df["travelTime_car"]/60.0
    df["car_distance_km"] = df["distance_car"]/1000.0

    # keep only relevant columns
    df_car_trips = df_car[["identifier", "car_travel_time_min", "car_distance_km"]]

    return df_car_trips