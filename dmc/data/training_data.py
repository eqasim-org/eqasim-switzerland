import pandas as pd
import numpy as np
import dmc.cost.car as car_cost
import dmc.cost.pt as pt_cost
import dmc.cost.parking as parking_cost
import dmc.penalties.parking as parking_penalty

from dmc.data.utils import merge_same_trips, adjust_weights
from dmc.constants import constants

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def configure(context):
    context.stage("dmc.data.prepare_routed_data")
    context.stage("dmc.data.prepare_survey_data")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.constants")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("data.spatial.swiss_border")

    context.config("car_cost_per_km", constants.CAR_COST_PER_KM) #CHF per km
    context.config("parking_cost_per_hour_CHF_urban", constants.PARKING_COST_PER_HOUR_CHF_URBAN)
    context.config("parking_cost_per_hour_CHF_urbancore", constants.PARKING_COST_PER_HOUR_CHF_URBANCORE)
    context.config("parking_cost_per_hour_CHF_suburban", constants.PARKING_COST_PER_HOUR_CHF_SUBURBAN)
    context.config("parking_price_reduction_for_work", constants.PARKING_PRICE_REDUCTION_FOR_WORK) 
    
    context.config("urbancore_parking_search_min", constants.URBANCORE_PARKING_SEARCH_MIN)
    context.config("urban_parking_search_min", constants.URBAN_PARKING_SEARCH_MIN) 
    context.config("suburban_parking_search_min", constants.SUBURBAN_PARKING_SEARCH_MIN)

    context.config("car_cost_model", constants.CAR_COST_MODEL)
    context.config("pt_regional_radius_km", constants.PT_REGIONAL_RADIUS_KM)
    context.config("only_from_home_trips", False)

def execute(context):
    df_routed = context.stage("dmc.data.prepare_routed_data")
    df_survey = context.stage("dmc.data.prepare_survey_data")    
    c         = context.stage("data.constants")

    # merge the two dataframes on person_id and trip_id
    df = pd.merge(df_routed, df_survey, on=["person_id", "trip_id"], how="inner")
    df = df.drop(columns="mode_x").rename(columns={"mode_y":"mode"}).reset_index(drop=True)

    # here I merge the trips that are supposed to be the same trip
    # If the arrival time of trip i is equal to the departure time of trip i+1, and the mode is the same, merge them
    # df = merge_same_trips(context, df)

    # Availabilities
    persons = context.stage("data.microcensus.persons")
    persons = persons[["person_id","car_availability","number_of_cars","number_of_bikes_class",
                       "driving_license","is_car_passenger", "age"]].reset_index(drop=True)

    persons["car_passenger_availability"] = True # car passenger available for all persons
    persons["walk_availability"] = True # walk available for all persons
    persons["pt_availability"] = True # pt available for all persons
    persons["car_availability"] = persons["car_availability"]!= c.CAR_AVAILABILITY_NEVER         
    persons["car_availability"] = ((persons["car_availability"])&
                                   (persons["driving_license"]==True)&
                                   (persons["age"]>=18))
    persons["bike_availability"] = persons["number_of_bikes_class"] != c.BIKE_AVAILABILITY_NEVER    

    persons = persons[["person_id","car_availability","car_passenger_availability","bike_availability","walk_availability","pt_availability"]]
    df = df.merge(persons, on="person_id", how="left")

    # Manage availabilities based on routing results
    """
    availabilities are removed if :
        - no route is generated (in routed data)
        - travel time < 1 min for car and pt
        - pt_transfers > 5 or pt_transfer_time_min > 50min or in_vehicle_time_min > 300min for pt
        - distance higher than 300km for car
        - distance higher than 20km for bike
        - distance higher than 5km for walk
    """
    for mode in ['car', 'car_passenger', 'pt', 'walk', 'bike']:   
        not_routed = df[f"{mode}_travel_time_min"].isna()     
        expectedModeUsed = f"expectedModeUsed_{mode.replace('_passenger','')}"
        df[expectedModeUsed] = df[expectedModeUsed].astype(bool)        
        not_routed|= ~(df[expectedModeUsed])        
        df.loc[not_routed, f"{mode}_availability"] = False        
        logger.info(f"{mode} : removing {not_routed.sum()} availabilities")

    pt_unavailability = ((df["pt_in_vehicle_time_min"]<0.5) | 
                         (df["pt_in_vehicle_time_min"]>300) |
                         (df["pt_transfers"]>6) |
                         (df["pt_transfer_time_min"]>60) 
                         )
    df.loc[pt_unavailability, "pt_availability"] = False

    car_unavailability = (df["car_travel_time_min"]<0.5) | (df["car_travel_time_min"]>300)
    df.loc[car_unavailability, "car_availability"] = False        
    df.loc[car_unavailability, "car_passenger_availability"] = False    

    small_distance = df.euclidean_distance_km<0.05 #less than 50m
    df.loc[small_distance,["pt_availability","car_availability","car_passenger_availability"]] = False

    bike_unavailability = df.euclidean_distance_km>=12
    df.loc[bike_unavailability, "bike_availability"] = False    

    walk_unavailability = df.euclidean_distance_km>=6
    df.loc[walk_unavailability, "walk_availability"] = False    

    # compute costs
    ## car cost      
    df["car_cost_CHF"] = car_cost.get_cost(df, context)
    
    ## parking cost    
    parking_cost_CHF, parking_duration_min = parking_cost.get_cost(df, context)
    df["parking_cost_CHF"] = parking_cost_CHF
    df["actual_parking_duration_min"] = parking_duration_min

    ##public transport cost
    df["pt_cost_CHF"] = pt_cost.get_cost(df, context, pt_regional_radius_km = context.config("pt_regional_radius_km"))

    # Set parking searching time
    df['parking_searching_duration_min'] = parking_penalty.get_parking_search_min(df, context)


    # LAST CHECKS AND FILTERS
    ### if only first trips and from home trips are considered
    if context.config("only_from_home_trips"):
        origin_activity = df["purpose"].shift(1).fillna("home")
        origin_activity[df.is_first] = "home"
        df = df[origin_activity=="home"]        

    ### remove nans when modes are not available, because anyway it won't be considered
    for mode in ['car', 'car_passenger', 'pt', 'walk', 'bike']:    
        not_available_mode = ~ df[f"{mode}_availability"]

        mode_columns = [col for col in df.columns if mode in col]
        is_nan = df[mode_columns].isna().sum(axis=1) > 0

        df.loc[is_nan & not_available_mode, mode_columns] = df.loc[is_nan & not_available_mode, mode_columns].fillna(0)

    ### remove nans and deactivate availability when modes are not selected 
    for mode in ['car', 'car_passenger', 'pt', 'walk', 'bike']:    
        not_selected = df[f"mode"]!=mode
        mode_columns = [col for col in df.columns if mode in col]
        is_nan = df[mode_columns].isna().sum(axis=1) > 0

        df.loc[is_nan & not_selected, mode_columns] = df.loc[is_nan & not_selected, mode_columns].fillna(0)
        df.loc[is_nan & not_selected, f"{mode}_availability"] = False
    
    ### remove trips with only one availability
    cols_availabilities = ['car_availability', 'pt_availability', 'bike_availability', 'walk_availability', 'car_passenger_availability']
    df = df[(df[cols_availabilities].sum(axis=1)>1)]
    
    ### if only car_passenger and car are available, remove the trip
    f_only_car = (df["car_availability"]) & (~df["pt_availability"]) & (~df["bike_availability"]) & (~df["walk_availability"]) & (df["car_passenger_availability"])
    df = df[~f_only_car]

    ### remove nan rows
    df = df[df.isna().sum(axis=1)==0].reset_index(drop=True)

    ### remove modes that are selected but not available
    f_remove = np.zeros((len(df),), dtype = bool)
    for mode in df["mode"].unique():
        f_mode = (~df["{}_availability".format(mode)]) & (df["mode"] == mode)        
        f_remove |= f_mode
    df = df[~f_remove]

    ### remove very short and very long trips
    out_of_range_distance = ((df.euclidean_distance_km < 0.01) | (df.euclidean_distance_km > 100))
    df = df[~out_of_range_distance].reset_index(drop=True)

    ### adjust weights to match target mode shares
    df["person_weight"] = adjust_weights(context, df)

    ########################### RETURN ################################
    columns = [
        "person_id", "trip_id", "person_weight", "mode", "euclidean_distance_km",
        "home_municipality", "origin_municipality", "destination_municipality", 
        "destination_work", "origin_home", "destination_home", "destination_education",
        "destination_shopping", "destination_leisure", "destination_other",        
        "elevation_difference", "purpose",

        # person
        "age", "sex", "income", "income_class", "sp_region", "ms_region", "is_car_passenger", "ovgk", 
        "good_pt_service", "medium_pt_service", "car_ownership_ratio", "is_retired","is_junior","low_income",

        # car
        'car_availability' ,'car_travel_time_min', 'car_cost_CHF', 'driving_license', "car_distance_km",

        # car passenger
        'car_passenger_availability', 'car_passenger_travel_time_min', 'car_passenger_distance_km',
        
        # pt
        'pt_availability', 'pt_in_vehicle_time_min', 'pt_egress_time_min','pt_in_vehicle_distance_km',
        'pt_access_time_min', 'pt_travel_time_min', 'pt_transfer_time_min',
        'pt_transfers', 'pt_access_egress_time_min', 'pt_cost_CHF',

        # bicycle
        'bike_availability', "bike_travel_time_min", "bike_distance_km",

        # walking
        'walk_availability', "walk_travel_time_min", "walk_distance_km",

        # parking 
        "parking_cost_CHF", "parking_searching_duration_min", "actual_parking_duration_min",
    ]
    
    return df[columns].reset_index(drop=True)
