import pandas as pd
import numpy as np


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure(context):
    context.stage("mode_choice.estimate_model.data.variables")
    context.stage("mode_choice.estimate_model.data.survey_data")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.constants")
    context.config("only_from_home_trips", default=False)

def execute(context):
    df_variables = context.stage("mode_choice.estimate_model.data.variables")
    df_survey = context.stage("mode_choice.estimate_model.data.survey_data") [[
        "person_id", "trip_id", "person_weight", "mode", "income", "region", "age", "sex", "driving_license",
        "origin_home", "destination_work", "departure_time", 
        'home_municipality', 'origin_municipality', 'destination_municipality',
        'is_car_passenger'
    ]]      

    # merge the two dataframes on person_id and trip_id
    df = pd.merge(df_variables, df_survey, on=["person_id", "trip_id"], how="inner")    
    logger.info(f"There are {len(df)} trips after merging variables and survey data.")
    
    # LAST CHECKS AND FILTERS
    ### if only first trips and from home trips are considered
    if context.config("only_from_home_trips"):
        df = df[df.origin_home=="home"].reset_index(drop=True)
        logger.info(f"There are {len(df)} trips after keeping only from home trips.")

    ### fill nans when modes are not available, because anyway it won't be considered
    for mode in ['car', 'car_passenger', 'pt', 'walk', 'bike']:    
        not_available_mode = ~ df[f"{mode}_availability"]

        mode_columns = [col for col in df.columns if mode in col]
        is_nan = df[mode_columns].isna().sum(axis=1) > 0
        rows_to_fill = is_nan & not_available_mode
        df.loc[rows_to_fill, mode_columns] = df.loc[rows_to_fill, mode_columns].fillna(0)

    ### fill nans and deactivate availability when modes are not selected 
    for mode in ['car', 'car_passenger', 'pt', 'walk', 'bike']:    
        not_selected = df["mode"]!=mode
        mode_columns = [col for col in df.columns if mode in col]
        is_nan = df[mode_columns].isna().sum(axis=1) > 0

        df.loc[is_nan & not_selected, mode_columns] = df.loc[is_nan & not_selected, mode_columns].fillna(0)
        df.loc[is_nan & not_selected, f"{mode}_availability"] = False
    
    ### remove trips with only one availability
    cols_availabilities = ['car_availability', 'pt_availability', 'bike_availability', 'walk_availability', 'car_passenger_availability']
    df = df[(df[cols_availabilities].sum(axis=1)>1)]
    logger.info(f"There are {len(df)} trips after removing trips with only one available mode.")

    ### if only car_passenger and car are available, remove the trip
    f_only_car = (df["car_availability"]) & (df["car_passenger_availability"]) & (~df["pt_availability"]) & (~df["bike_availability"]) & (~df["walk_availability"])
    df = df[~f_only_car]
    logger.info(f"There are {len(df)} trips after removing trips with only car and car_passenger available.")

    ### remove modes that are selected but not available
    f_remove = np.zeros((len(df),), dtype = bool)
    for mode in df["mode"].unique():
        f_mode = (~df["{}_availability".format(mode)]) & (df["mode"] == mode)        
        f_remove |= f_mode
    df = df[~f_remove]
    logger.info(f"There are {len(df)} trips after removing trips with selected but not available modes.")

    ### remove very short and very long trips
    out_of_range_distance = ((df.euclidean_distance_km < 0.1) | (df.euclidean_distance_km > 200))
    df = df[~out_of_range_distance]
    logger.info(f"There are {len(df)} trips after removing very short and very long trips.")

    ### remove nan rows
    df = df[df.isna().sum(axis=1)==0].reset_index(drop=True)
    logger.info(f"There are {len(df)} trips after removing trips with nan values.")

    ########################### RETURN ################################
    columns = [
        "person_id", "trip_id", "person_weight", "mode", "euclidean_distance_km",
        "home_municipality", "origin_municipality", "destination_municipality", 
        "destination_work", "origin_home",

        # person
        "age", "sex", "income", "region", "is_car_passenger", "driving_license",
        
        # Availabilities
        'walk_availability', 'bike_availability', 'car_availability', 'pt_availability', 'car_passenger_availability',
        
        # Walk variables
        'walk_distance_km', 'walk_travel_time_min',
        
        # Bike variables
        'bike_distance_km', 'bike_travel_time_min',
        
        # PT variables
        'pt_access_egress_time_min', 'pt_waiting_time_min', 'pt_transfers', 'pt_in_vehicle_time_min', 'pt_distance_km', 'pt_cost_CHF',
        
        # Car variables
        'car_travel_time_min', 'car_access_egress_time_min', 'car_distance_km', 'car_cost_CHF', 'parking_cost_CHF', 'parking_searching_duration_min',
        
        # Car passenger variables
        'car_passenger_travel_time_min', 'car_passenger_access_egress_time_min', 'car_passenger_distance_km'
    ]
    
    return df[columns].reset_index(drop=True)
