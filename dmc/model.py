import os

from dmc.constants import constants
from dmc.writer import writer
from dmc.vot.functions import vot_utils

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioMax, bioMin, log
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ['car', 'pt', 'bike', 'walk', 'car_passenger']
TIME_SCALE_MIN = constants.TIME_SCALE_MIN
DISTANCE_SCALE_KM = constants.DISTANCE_SCALE_KM
PT_REGIONAL_RADIUS_KM = constants.PT_REGIONAL_RADIUS_KM

def configure(context):
    context.stage("dmc.data.training_data")
    context.stage("data.constants")
    context.config("ignore_car_passenger", False)
    context.config("distance_cost_interaction", True)
    context.config("income_cost_interaction", True)
    context.config("use_exponents", True)
    context.config("use_income_in_dmc", True)
    
    # these are used in the writer
    context.config("urbancore_parking_search_min", constants.URBANCORE_PARKING_SEARCH_MIN)
    context.config("urban_parking_search_min", constants.URBAN_PARKING_SEARCH_MIN) 
    context.config("suburban_parking_search_min", constants.SUBURBAN_PARKING_SEARCH_MIN)
    context.config("car_cost_per_km", constants.CAR_COST_PER_KM) #CHF per km
    context.config("parking_cost_per_hour_CHF_urban", constants.PARKING_COST_PER_HOUR_CHF_URBAN)
    context.config("parking_cost_per_hour_CHF_urbancore", constants.PARKING_COST_PER_HOUR_CHF_URBANCORE)
    context.config("parking_cost_per_hour_CHF_suburban", constants.PARKING_COST_PER_HOUR_CHF_SUBURBAN)
    context.config("parking_price_reduction_for_work", constants.PARKING_PRICE_REDUCTION_FOR_WORK)
    context.config("car_cost_model", constants.CAR_COST_MODEL)
    context.config("reference_euclidean_distance_km", constants.REF_EUCLIDEAN_DISTANCE_KM)
    context.config("reference_income_chf", constants.REF_INCOME_CHF)
    context.config("pt_regional_radius_km", constants.PT_REGIONAL_RADIUS_KM)

def preprocess_data(df, ignore_car_passenger):
    modes = MODES.copy()
    if ignore_car_passenger:
        df = df[df["mode"] != "car_passenger"].reset_index(drop=True)
        df = df[[col for col in df.columns if "car_passenger" not in col]]
        modes.remove("car_passenger")
    
    df = df[df["mode"].isin(modes)]
    for mode in modes:
        df[f"{mode}_availability"] = df[f"{mode}_availability"].astype(int)
    
    df["driving_license"] = df["driving_license"].astype(int)    
    # purpose dummies
    df["destination_work"] = df["destination_work"].astype(int)
    df["origin_home"] = df["origin_home"].astype(int)
    df["destination_home"] = df["destination_home"].astype(int)
    df["destination_education"] = df["destination_education"].astype(int)
    df["destination_shopping"] = df["destination_shopping"].astype(int)
    df["destination_leisure"] = df["destination_leisure"].astype(int)
    df["destination_other"] = df["destination_other"].astype(int)    
    # mode encoding
    df["mode"] = df["mode"].apply(modes.index)
    # normalize weights
    df["person_weight"] = len(df) * df["person_weight"] / df["person_weight"].sum()
    # other variables
    if not ignore_car_passenger:
        df["is_car_passenger"] = df["is_car_passenger"].astype(int)
    # municipality encoding
    municipalities = ['rural', 'suburban', 'urban', 'urbancore']
    df['home_municipality'] = df['home_municipality'].apply(municipalities.index).astype(int)
    df['origin_municipality'] = df['origin_municipality'].apply(municipalities.index).astype(int)
    df['destination_municipality'] = df['destination_municipality'].apply(municipalities.index).astype(int)
    # dummies for municipality types
    df["urban_destination"] = (df["destination_municipality"] == municipalities.index("urban")).astype(int)
    df["urbancore_destination"] = (df["destination_municipality"] == municipalities.index("urbancore")).astype(int)
    # regions (clusters based on mode shares)
    df["region_0"] = (df["ms_region"]==0).astype(int)
    df["region_1"] = (df["ms_region"]==1).astype(int)
    df["region_2"] = (df["ms_region"]==2).astype(int)
    # distance categories
    df["short_distance"] = df["short_distance"].astype(float)
    df["long_distance"]  = df["long_distance"].astype(float)
    df["very_long_distance"]  = df["very_long_distance"].astype(float)
    # quality of pt service
    df["good_pt_service"] = df["good_pt_service"].astype(int)
    df["medium_pt_service"] = df["medium_pt_service"].astype(int)
    df["destination_good_pt_service"] = df["destination_good_pt_service"].astype(int)
    df["destination_medium_pt_service"] = df["destination_medium_pt_service"].astype(int)
    # low/high income
    df["low_income"] = df["low_income"].astype(int)    
    df["high_income"] = df["high_income"].astype(int)
    # car ownership
    df["car_ownership_ratio"] = df["car_ownership_ratio"].astype(float)
    df["has_car"] = df["has_car"].astype(int)
    # 5 biggest muicipalities
    df["destination_zurich"] = df["destination_zurich"].astype(int)
    df["destination_geneva"] = df["destination_geneva"].astype(int)
    df["destination_basel"] = df["destination_basel"].astype(int)
    df["destination_lausanne"] = df["destination_lausanne"].astype(int)
    df["destination_luzern"] = df["destination_luzern"].astype(int)
    df['destination_bern'] = df['destination_bern'].astype(int)

    # drop columns not used in the model
    columns_to_drop = [
        "person_id", "trip_id", "home_municipality", "origin_municipality", 
        "sp_region", "ms_region", "ovgk", "pt_egress_time_min", "income_class",
        "pt_access_time_min", "actual_parking_duration_min", "region_0", "purpose"
    ]
    df = df.drop(columns=columns_to_drop)
    return df, modes

def define_variables(database, ignore_car_passenger):
    variables = {
        "mode": db.Variable("mode"),
        "weight": db.Variable("person_weight"),
        "euclidean_distance_km": db.Variable("euclidean_distance_km"),
        "age": db.Variable("age"),
        "income": db.Variable("income"),
        "low_income": db.Variable("low_income"),
        "high_income": db.Variable("high_income"),
        "sex": db.Variable("sex"),
        "driving_license": db.Variable("driving_license"),
        "parking_cost_CHF": db.Variable("parking_cost_CHF"),        
        "car_ownership_ratio": db.Variable("car_ownership_ratio"),
        "has_car": db.Variable("has_car"),
        "num_adults": db.Variable("num_adults"),
        "is_retired": db.Variable("is_retired"),
        "is_junior": db.Variable("is_junior"),
        # densities
        "destination_employee_density": db.Variable("destination_employee_density"),
        "destination_population_density": db.Variable("destination_population_density"),
        "destination_companies_density": db.Variable("destination_companies_density"),
        # cities destinations
        "destination_zurich": db.Variable("destination_zurich"),
        "destination_geneva": db.Variable("destination_geneva"),
        "destination_basel": db.Variable("destination_basel"),
        "destination_lausanne": db.Variable("destination_lausanne"),
        "destination_luzern": db.Variable("destination_luzern"),
        "destination_bern": db.Variable("destination_bern"),
        # regions       
        "region_1": db.Variable("region_1"),
        "region_2": db.Variable("region_2"),
        # distances dummies
        "elevation": db.Variable("elevation_difference"),
        "short_distance": db.Variable("short_distance"),
        "long_distance": db.Variable("long_distance"),
        "very_long_distance": db.Variable("very_long_distance"),
        # destination urbanisation
        "urban_destination": db.Variable("urban_destination"),
        "urbancore_destination": db.Variable("urbancore_destination"),
        "destination_municipality": db.Variable("destination_municipality"),
        # purposes
        "destination_work": db.Variable("destination_work"),
        "destination_home": db.Variable("destination_home"),
        "destination_education": db.Variable("destination_education"),
        "destination_shopping": db.Variable("destination_shopping"),
        "destination_leisure": db.Variable("destination_leisure"),
        "destination_other": db.Variable("destination_other"),
        "origin_home": db.Variable("origin_home"),
        # car
        "car_availability": db.Variable("car_availability"),
        "car_travel_time_min": db.Variable("car_travel_time_min"),
        "car_distance_km": db.Variable("car_distance_km"),
        "car_cost_CHF": db.Variable("car_cost_CHF"),
        "parking_searching_duration_min": db.Variable("parking_searching_duration_min"),
        # pt
        "pt_availability": db.Variable("pt_availability"),
        "pt_access_egress_time_min": db.Variable("pt_access_egress_time_min"),
        "pt_in_vehicle_time_min": db.Variable("pt_in_vehicle_time_min"),        
        "pt_transfers": db.Variable("pt_transfers"),
        "pt_transfer_time_min": db.Variable("pt_transfer_time_min"),
        "pt_in_vehicle_distance_km": db.Variable("pt_in_vehicle_distance_km"),
        "pt_cost_CHF": db.Variable("pt_cost_CHF"),
        "good_pt_service": db.Variable("good_pt_service"),
        "medium_pt_service": db.Variable("medium_pt_service"),
        "destination_good_pt_service": db.Variable("destination_good_pt_service"),
        "destination_medium_pt_service": db.Variable("destination_medium_pt_service"),
        # bike
        "bike_availability": db.Variable("bike_availability"),
        "bike_travel_time_min": db.Variable("bike_travel_time_min"),
        "bike_distance_km": db.Variable("bike_distance_km"),
        # walk
        "walk_availability": db.Variable("walk_availability"),
        "walk_travel_time_min": db.Variable("walk_travel_time_min"),
        "walk_distance_km": db.Variable("walk_distance_km"),
    }
    if not ignore_car_passenger:
        variables.update({
            "car_passenger_availability": db.Variable("car_passenger_availability"),
            "car_passenger_travel_time_min": db.Variable("car_passenger_travel_time_min"),
            "is_car_passenger": db.Variable("is_car_passenger"),
            "car_passenger_distance_km": db.Variable("car_passenger_distance_km"),
        })
    return variables

def define_betas(ignore_car_passenger, use_exponents, use_income):
    """
        Note for model developers:
        To ensure model identifiability and convergence in discrete choice estimation, certain parameters must be fixed and non-estimable. 
        Failure to do so may result in non-convergence or indeterminate solutions due to parameter redundancy. Specifically, two exponents 
        are constrained to 1.0 (implying linear utility functions). Additionally, alternative-specific constants (ASCs) are normalized 
        relative to public transport (PT) as the reference mode. Furthermore, one trip purpose category is fixed as the baseline to 
        establish relative effects for other purposes.
    """
    trainable = 0 if use_exponents else 1
    max_disutility = -2e-3
    min_lambda = 1e-2
    max_lambda = 2.0
    betas = {
        # lambdas
        "lambda_cost_distance": Beta("lambda_cost_distance", -0.08, None, max_disutility, 0),
        "lambda_cost_income": Beta("lambda_cost_income", 0.0, None, 0.0, 0 if use_income else 1),

        "lambda_car_travel_time": Beta("lambda_car_travel_time", 0.72 if trainable else 1.0, min_lambda, max_lambda, trainable),
        "lambda_pt_in_vehicle_time": Beta("lambda_pt_in_vehicle_time", 1.5 if trainable else 1.0, min_lambda, max_lambda, trainable),
        "lambda_pt_access_egress_time": Beta("lambda_pt_access_egress_time", 0.593 if trainable else 1.0, min_lambda, max_lambda, trainable),
        "lambda_pt_transfers": Beta("lambda_pt_transfers", 1.187 if trainable else 1.0, min_lambda, max_lambda, trainable),
        "lambda_pt_transfer_time": Beta("lambda_pt_transfer_time", 1.0 if trainable else 1.0, min_lambda, max_lambda, 1), # doesn't converge
        "lambda_pt_distance": Beta("lambda_pt_distance", 0.521, min_lambda, max_lambda, 0),
        "lambda_car_passenger_travel_time": Beta("lambda_car_passenger_travel_time", 0.832 if trainable else 1.0, min_lambda, max_lambda, trainable),        
        "lambda_bike": Beta("lambda_bike", 0.561 if trainable else 1.0, min_lambda, max_lambda, trainable),
        "lambda_walk": Beta("lambda_walk", 0.28 if trainable else 1.0, min_lambda, max_lambda, trainable),

        # cost & other
        "beta_cost_CHF": Beta("beta_cost_CHF", -0.12, None, max_disutility, 0), 
        "beta_destination_employee_density": Beta("beta_destination_employee_density", 0.0, None, None, 1),
        "beta_destination_population_density": Beta("beta_destination_population_density", 0.2, None, None, 0),
        "beta_destination_companies_density": Beta("beta_destination_companies_density", 0.3, None, None, 0),

        # car
        "beta_car_asc": Beta("beta_car_asc", 3.48, None, None, 0),
        "beta_car_travel_time_min": Beta("beta_car_travel_time_min", -0.964, None, max_disutility, 0),        

        "beta_car_destination_work": Beta("beta_car_destination_work", 0.447, None, None, 0),
        "beta_car_destination_home": Beta("beta_car_destination_home", 0, None, None, 1),
        "beta_car_destination_education": Beta("beta_car_destination_education", -0.596, None, None, 0),
        "beta_car_destination_shopping": Beta("beta_car_destination_shopping", 0.608, None, None, 0),
        "beta_car_destination_leisure": Beta("beta_car_destination_leisure", 0.251, None, None, 0),
        "beta_car_destination_other": Beta("beta_car_destination_other", 0.9, None, None, 0),
        "beta_car_origin_home": Beta("beta_car_origin_home", 0, None, None, 1), # not significant

        "beta_car_destination_urban": Beta("beta_car_destination_urban", -0.072, None, 0.0, 0),
        "beta_car_destination_urbancore": Beta("beta_car_destination_urbancore", -1.039, None, 0.0, 0),
        
        "beta_car_sex": Beta("beta_car_sex", -0.574, None, None, 0),
        "beta_car_age": Beta("beta_car_age", 0.015, None, None, 0),
        "beta_car_retired": Beta("beta_car_retired", -0.521, None, None, 0),
        "beta_car_junior": Beta("beta_car_junior", 0.0, None, None, 1),
        "beta_car_ownership_ratio": Beta("beta_car_ownership_ratio", -2.25, None, None, 0),        
        "beta_car_low_income": Beta("beta_car_low_income", 0.0, None, None, 0 if use_income else 1),
        "beta_car_high_income": Beta("beta_car_high_income", 0.0, None, None, 0 if use_income else 1),

        "beta_car_region_1": Beta("beta_car_region_1", 0.223, None, None, 0),
        "beta_car_region_2": Beta("beta_car_region_2", -0.467, None, None, 0),        
                
        "beta_car_short_distance": Beta("beta_car_short_distance", 0.321, None, None, 0),
        "beta_car_long_distance": Beta("beta_car_long_distance", 0.086, None, None, 0),

        "beta_car_densities": Beta("beta_car_densities", -0.1, None, None, 0),
        "beta_car_destination_zurich": Beta("beta_car_destination_zurich", 0.0, None, None, 0.0),
        "beta_car_destination_geneva": Beta("beta_car_destination_geneva", 0.0, None, None, 0.0),
        "beta_car_destination_basel": Beta("beta_car_destination_basel", 0.0, None, None, 0.0),
        "beta_car_destination_lausanne": Beta("beta_car_destination_lausanne", 0.0, None, None, 0.0),
        "beta_car_destination_luzern": Beta("beta_car_destination_luzern", 0.0, None, None, 0.0),
        "beta_car_destination_bern": Beta("beta_car_destination_bern", 0.0, None, None, 0.0),

        # pt
        "beta_pt_asc": Beta("beta_pt_asc", 0, None, None, 1),
        "beta_pt_access_egress_time_min": Beta("beta_pt_access_egress_time_min", -0.716, None, max_disutility, 0),
        "beta_pt_in_vehicle_time_min": Beta("beta_pt_in_vehicle_time_min", -0.044, None, max_disutility, 0),        
        "beta_pt_transfers": Beta("beta_pt_transfers", -0.477, None, max_disutility, 0),
        "beta_pt_transfer_time_min": Beta("beta_pt_transfer_time_min", -0.0234, None, -0.02, 0),
        "beta_pt_distance_km": Beta("beta_pt_distance_km", -0.4, -2.0, 2.0, 0),
        
        "beta_pt_sex": Beta("beta_pt_sex", 0, None, None, 1),
        "beta_pt_age": Beta("beta_pt_age", 0, None, None, 1),
        "beta_pt_retired": Beta("beta_pt_retired", 0, None, None, 1),
        "beta_pt_junior": Beta("beta_pt_junior", 0, None, None, 0),
        "beta_pt_low_income": Beta("beta_pt_low_income", 0.0, None, None, 0 if use_income else 1),
        "beta_pt_high_income": Beta("beta_pt_high_income", 0.0, None, None, 0 if use_income else 1),

        "beta_pt_destination_work": Beta("beta_pt_destination_work", 0, None, None, 1),
        "beta_pt_destination_home": Beta("beta_pt_destination_home", 0, None, None, 1),
        "beta_pt_destination_education": Beta("beta_pt_destination_education", 0, None, None, 1),
        "beta_pt_destination_shopping": Beta("beta_pt_destination_shopping", 0, None, None, 1),
        "beta_pt_destination_leisure": Beta("beta_pt_destination_leisure", 0, None, None, 1),
        "beta_pt_destination_other": Beta("beta_pt_destination_other", 0, None, None, 1),        
        "beta_pt_origin_home": Beta("beta_pt_origin_home", 0, None, None, 1), 
        
        "beta_pt_destination_urban": Beta("beta_pt_destination_urban", 0, None, None, 1),
        "beta_pt_destination_urbancore": Beta("beta_pt_destination_urbancore", 0, None, None, 1),

        "beta_pt_region_1": Beta("beta_pt_region_1", 0, None, None, 1),
        "beta_pt_region_2": Beta("beta_pt_region_2", 0, None, None, 1),               
        
        "beta_pt_short_distance": Beta("beta_pt_short_distance", 0, None, None, 1),
        "beta_pt_long_distance": Beta("beta_pt_long_distance", 0, None, None, 1),
        
        "beta_pt_good_service": Beta("beta_pt_good_service", 0.685, None, None, 0),
        "beta_pt_medium_service": Beta("beta_pt_medium_service", 0.196, None, None, 0),
        "beta_pt_destination_good_service": Beta("beta_pt_destination_good_service", 0.0, None, None, 0),
        "beta_pt_destination_medium_service": Beta("beta_pt_destination_medium_service", 0.0, None, None, 0),
        
        "beta_pt_densities": Beta("beta_pt_densities", 0.08, None, None, 0),

        #bike        
        "beta_bike_asc": Beta("beta_bike_asc", 3.667, None, None, 0),
        "beta_bike_travel_time_min": Beta("beta_bike_travel_time_min", -2.873, None, max_disutility, 0),        

        "beta_bike_age": Beta("beta_bike_age", 0.02, None, None, 0),
        "beta_bike_sex": Beta("beta_bike_sex", -0.441, None, None, 0),
        "beta_bike_retired": Beta("beta_bike_retired", -0.848, None, None, 0),
        "beta_bike_junior": Beta("beta_bike_junior", 0, None, None, 1),
        "beta_bike_low_income": Beta("beta_bike_low_income", 0.0, None, None, 0 if use_income else 1),
        "beta_bike_high_income": Beta("beta_bike_high_income", 0.0, None, None, 0 if use_income else 1),

        "beta_bike_destination_work": Beta("beta_bike_destination_work", 0, None, None, 1), # not significant
        "beta_bike_destination_home": Beta("beta_bike_destination_home", 0, None, None, 1), # not significant
        "beta_bike_destination_education": Beta("beta_bike_destination_education", -0.45, None, None, 0),
        "beta_bike_destination_shopping": Beta("beta_bike_destination_shopping", -0.353, None, None, 0),
        "beta_bike_destination_leisure": Beta("beta_bike_destination_leisure", -0.093, None, None, 0),
        "beta_bike_destination_other": Beta("beta_bike_destination_other", -0.515, None, None, 0),
        "beta_bike_origin_home": Beta("beta_bike_origin_home", 0.228, None, None, 0),

        "beta_bike_destination_urban": Beta("beta_bike_destination_urban", -0.32, None, None, 0),
        "beta_bike_destination_urbancore": Beta("beta_bike_destination_urbancore", -0.655, None, None, 0),

        "beta_bike_region_1": Beta("beta_bike_region_1", -0.929, None, None, 0),
        "beta_bike_region_2": Beta("beta_bike_region_2", -0.413, None, None, 0),

        "beta_bike_short_distance": Beta("beta_bike_short_distance", 0.429, None, None, 0),
        "beta_bike_long_distance": Beta("beta_bike_long_distance", 0.0, None, None, 1),
        
        "beta_bike_densities": Beta("beta_bike_densities", 0.0, None, None, 0),

        # walk
        "beta_walk_asc": Beta("beta_walk_asc", 10.58, None, None, 0),
        "beta_walk_travel_time_min": Beta("beta_walk_travel_time_min", -8.164, None, max_disutility, 0),

        "beta_walk_age": Beta("beta_walk_age", 0.007, None, None, 0),
        "beta_walk_sex": Beta("beta_walk_sex", -0.211, None, None, 0),
        "beta_walk_retired": Beta("beta_walk_retired", -0.285, None, None, 0),
        "beta_walk_junior": Beta("beta_walk_junior", 0, None, None, 0),
        "beta_walk_low_income": Beta("beta_walk_low_income", 0, None, None, 1),
        "beta_walk_high_income": Beta("beta_walk_high_income", 0.0, None, None, 1),

        "beta_walk_destination_work": Beta("beta_walk_destination_work", 0, None, None, 1), # not significant
        "beta_walk_destination_home": Beta("beta_walk_destination_home", 0, None, None, 1), # not significant
        "beta_walk_destination_education": Beta("beta_walk_destination_education", -0.181, None, None, 0),
        "beta_walk_destination_shopping": Beta("beta_walk_destination_shopping", 0.022, None, None, 1), # not significant
        "beta_walk_destination_leisure": Beta("beta_walk_destination_leisure", 0.354, None, None, 0),
        "beta_walk_destination_other": Beta("beta_walk_destination_other", 0.111, None, None, 0),
        "beta_walk_origin_home": Beta("beta_walk_origin_home", 0.198, None, None, 0),

        "beta_walk_destination_urban": Beta("beta_walk_destination_urban", -0.18, None, None, 0),
        "beta_walk_destination_urbancore": Beta("beta_walk_destination_urbancore", -0.469, None, None, 0),

        "beta_walk_region_1": Beta("beta_walk_region_1", 0.241, None, None, 0),
        "beta_walk_region_2": Beta("beta_walk_region_2", -0.155, None, None, 0),
        
        "beta_walk_short_distance": Beta("beta_walk_short_distance", 0.606, None, None, 0),
        "beta_walk_long_distance": Beta("beta_walk_long_distance", 0, None, None, 1),   # not significant  
        "beta_walk_densities": Beta("beta_walk_densities", 0.0, None, None, 1)
    }
    if not ignore_car_passenger:
        betas.update({
            "beta_car_passenger_asc": Beta("beta_car_passenger_asc", 0.46, None, None, 0),
            "beta_car_passenger_travel_time_min": Beta("beta_car_passenger_travel_time_min", -1.27, None, max_disutility, 0),
            "beta_car_passenger_distance_km": Beta("beta_car_passenger_distance_km", -0.1, None, None, 0),

            "beta_car_passenger_driving_permit": Beta("beta_car_passenger_driving_permit", -0.339, None, None, 0),
            "beta_car_passenger_age": Beta("beta_car_passenger_age", -0.003, None, None, 0),
            "beta_car_passenger_sex": Beta("beta_car_passenger_sex", 0.141, None, None, 0),
            "beta_car_passenger_retired": Beta("beta_car_passenger_retired", 0.249, None, None, 0),
            "beta_car_passenger_junior": Beta("beta_car_passenger_junior", 0.0, None, None, 0),
            "beta_car_passenger_low_income": Beta("beta_car_passenger_low_income", 0, None, None, 0 if use_income else 1),
            "beta_car_passenger_high_income": Beta("beta_car_passenger_high_income", 0, None, None, 0 if use_income else 1),

            "beta_car_passenger_destination_work": Beta("beta_car_passenger_destination_work", 0.135, None, None, 0),
            "beta_car_passenger_destination_home": Beta("beta_car_passenger_destination_home", 0, None, None, 1), # not significant
            "beta_car_passenger_destination_education": Beta("beta_car_passenger_destination_education", -0.558, None, None, 0),
            "beta_car_passenger_destination_shopping": Beta("beta_car_passenger_destination_shopping", 0.999, None, None, 0),
            "beta_car_passenger_destination_leisure": Beta("beta_car_passenger_destination_leisure", 1.276, None, None, 0),
            "beta_car_passenger_destination_other": Beta("beta_car_passenger_destination_other", 1.01, None, None, 0),
            "beta_car_passenger_origin_home": Beta("beta_car_passenger_origin_home", 0.0, None, None, 1), # not significant

            "beta_car_passenger_destination_urban": Beta("beta_car_passenger_destination_urban", -0.146, None, None, 0),
            "beta_car_passenger_destination_urbancore": Beta("beta_car_passenger_destination_urbancore", -1.097, None, None, 0),

            "beta_car_passenger_region_1": Beta("beta_car_passenger_region_1", 0.291, None, None, 0),
            "beta_car_passenger_region_2": Beta("beta_car_passenger_region_2", -0.515, None, None, 0),

            "beta_car_passenger_short_distance": Beta("beta_car_passenger_short_distance", 0.284, None, None, 0),
            "beta_car_passenger_long_distance": Beta("beta_car_passenger_long_distance", 0.148, None, None, 0),     
            "beta_car_passenger_very_long_distance": Beta("beta_car_passenger_very_long_distance", 0.0, None, None, 0),

            "beta_car_passenger_ownership_ratio": Beta("beta_car_passenger_ownership_ratio", 0.0, None, None, 0),
            "beta_car_passenger_has_car": Beta("beta_car_passenger_has_car", 0.0, None, None, 0), 
            "beta_car_passenger_densities": Beta("beta_car_passenger_densities", -0.15, None, None, 0),
        })
    return betas

def build_utilities(context, vars, betas, modes, ignore_car_passenger):
    # cost
    if context.config("distance_cost_interaction"):
        ref_euclidean_distance_km = context.config("reference_euclidean_distance_km")
        euclidean_interaction_cost = (vars["euclidean_distance_km"] / ref_euclidean_distance_km) ** betas["lambda_cost_distance"]
    else:
        euclidean_interaction_cost = 1

    if context.config("income_cost_interaction") & context.config("use_income_in_dmc"):
        ref_income_chf = context.config("reference_income_chf")
        income_interaction_cost = (vars["income"] / ref_income_chf) ** betas["lambda_cost_income"]
    else:
        income_interaction_cost = 1

    cost_interaction = euclidean_interaction_cost * income_interaction_cost    
    
    # density
    aggregated_density = betas["beta_destination_employee_density"]   * (vars["destination_employee_density"] ** constants.EMPLOYEES_DENSITY_EXPONENT)/constants.EMPLOYEES_DENSITY_SCALE + \
                         betas["beta_destination_population_density"] * (vars["destination_population_density"] ** constants.POPULATION_DENSITY_EXPONENT)/constants.POPULATION_DENSITY_SCALE + \
                         betas["beta_destination_companies_density"]  * (vars["destination_companies_density"] ** constants.COMPANIES_DENSITY_EXPONENT)/constants.COMPANIES_DENSITY_SCALE

    # utilities
    car_cost = (vars["car_cost_CHF"] + vars["parking_cost_CHF"])
    car_time = (vars["car_travel_time_min"] + vars["parking_searching_duration_min"]) / TIME_SCALE_MIN
    transformed_car_time = car_time ** betas["lambda_car_travel_time"]    
    car_utility = (
        betas["beta_car_asc"]
        + betas["beta_car_travel_time_min"] * transformed_car_time        
        + betas["beta_cost_CHF"] * car_cost * cost_interaction

        + betas["beta_car_destination_work"] * vars["destination_work"]
        + betas["beta_car_destination_home"] * vars["destination_home"]
        + betas["beta_car_destination_education"] * vars["destination_education"]
        + betas["beta_car_destination_shopping"] * vars["destination_shopping"]
        + betas["beta_car_destination_leisure"] * vars["destination_leisure"]
        + betas["beta_car_destination_other"] * vars["destination_other"]
        + betas["beta_car_destination_urban"] * vars["urban_destination"]
        + betas["beta_car_destination_urbancore"] * vars["urbancore_destination"]
        + betas["beta_car_sex"] * vars["sex"]
        + betas["beta_car_age"] * bioMax(0, vars["age"] - 17)
        + betas["beta_car_retired"] * vars["is_retired"]
        + betas["beta_car_junior"] * vars["is_junior"]
        + betas["beta_car_ownership_ratio"] * vars["car_ownership_ratio"]
        + betas["beta_car_low_income"] * vars["low_income"]
        + betas["beta_car_high_income"] * vars["high_income"]
        + betas["beta_car_region_1"] * vars["region_1"]
        + betas["beta_car_region_2"] * vars["region_2"]
        + betas["beta_car_origin_home"] * vars["origin_home"]    
        + betas["beta_car_short_distance"] * vars["short_distance"]
        + betas["beta_car_long_distance"] * vars["long_distance"]
        + betas["beta_car_densities"] * aggregated_density
        + betas["beta_car_destination_zurich"] * vars["destination_zurich"]
        + betas["beta_car_destination_geneva"] * vars["destination_geneva"]
        + betas["beta_car_destination_basel"] * vars["destination_basel"]
        + betas["beta_car_destination_lausanne"] * vars["destination_lausanne"]
        + betas["beta_car_destination_luzern"] * vars["destination_luzern"]
        + betas["beta_car_destination_bern"] * vars["destination_bern"]
    )

    transformed_pt_in_vehicle_time = (vars["pt_in_vehicle_time_min"] / TIME_SCALE_MIN) ** betas["lambda_pt_in_vehicle_time"]
    transformed_pt_transfers      = vars["pt_transfers"] ** betas["lambda_pt_transfers"]
    transformed_access_egress_time = (vars["pt_access_egress_time_min"] / TIME_SCALE_MIN) ** betas["lambda_pt_access_egress_time"]
    transformed_pt_transfer_time = (vars["pt_transfer_time_min"] / TIME_SCALE_MIN) ** betas["lambda_pt_transfer_time"]
    
    pt_distance = (vars["euclidean_distance_km"] / DISTANCE_SCALE_KM)
    distance_correction_limit = PT_REGIONAL_RADIUS_KM / DISTANCE_SCALE_KM
    cost_correction = betas["beta_pt_distance_km"] * bioMax(distance_correction_limit-pt_distance, 0.0)**betas["lambda_pt_distance"]
    pt_cost = vars["pt_cost_CHF"] + cost_correction

    pt_utility = (
        betas["beta_pt_asc"]
        + betas["beta_pt_access_egress_time_min"] * transformed_access_egress_time        
        + betas["beta_pt_in_vehicle_time_min"] * transformed_pt_in_vehicle_time
        + betas["beta_pt_transfer_time_min"] * transformed_pt_transfer_time
        + betas["beta_pt_transfers"] * transformed_pt_transfers                
        + betas["beta_cost_CHF"] * pt_cost * cost_interaction

        + betas["beta_pt_sex"] * vars["sex"]
        + betas["beta_pt_age"] * bioMax(0, vars["age"] - 17)
        + betas["beta_pt_retired"] * vars["is_retired"]
        + betas["beta_pt_junior"] * vars["is_junior"]
        + betas["beta_pt_low_income"] * vars["low_income"]
        + betas["beta_pt_high_income"] * vars["high_income"]

        + betas["beta_pt_destination_work"] * vars["destination_work"]
        + betas["beta_pt_destination_home"] * vars["destination_home"]
        + betas["beta_pt_destination_education"] * vars["destination_education"]
        + betas["beta_pt_destination_shopping"] * vars["destination_shopping"]
        + betas["beta_pt_destination_leisure"] * vars["destination_leisure"]
        + betas["beta_pt_destination_other"] * vars["destination_other"]
        + betas["beta_pt_destination_urban"] * vars["urban_destination"]        
        + betas["beta_pt_destination_urbancore"] * vars["urbancore_destination"]
        + betas["beta_pt_region_1"] * vars["region_1"]
        + betas["beta_pt_region_2"] * vars["region_2"]
        + betas["beta_pt_origin_home"] * vars["origin_home"]
        + betas["beta_pt_short_distance"] * vars["short_distance"]
        + betas["beta_pt_long_distance"] * vars["long_distance"]
        + betas["beta_pt_good_service"] * vars["good_pt_service"]
        + betas["beta_pt_medium_service"] * vars["medium_pt_service"]
        + betas["beta_pt_destination_good_service"] * vars["destination_good_pt_service"]
        + betas["beta_pt_destination_medium_service"] * vars["destination_medium_pt_service"]
        + betas["beta_pt_densities"] * aggregated_density
    )

    bike_travel_time = (vars["bike_travel_time_min"] / TIME_SCALE_MIN)
    bike_utility = (
        betas["beta_bike_asc"]
        + betas["beta_bike_travel_time_min"] * (bike_travel_time**betas["lambda_bike"])

        + betas["beta_bike_age"] * bioMax(0, vars["age"] - 17)
        + betas["beta_bike_sex"] * vars["sex"]
        + betas["beta_bike_retired"] * vars["is_retired"]
        + betas["beta_bike_junior"] * vars["is_junior"]
        + betas["beta_bike_low_income"] * vars["low_income"]
        + betas["beta_bike_high_income"] * vars["high_income"]

        + betas["beta_bike_destination_work"] * vars["destination_work"]
        + betas["beta_bike_destination_home"] * vars["destination_home"]
        + betas["beta_bike_destination_education"] * vars["destination_education"]
        + betas["beta_bike_destination_shopping"] * vars["destination_shopping"]
        + betas["beta_bike_destination_leisure"] * vars["destination_leisure"]
        + betas["beta_bike_destination_other"] * vars["destination_other"]
        + betas["beta_bike_destination_urban"] * vars["urban_destination"] 
        + betas["beta_bike_destination_urbancore"] * vars["urbancore_destination"]
        + betas["beta_bike_region_1"] * vars["region_1"]
        + betas["beta_bike_region_2"] * vars["region_2"]
        + betas["beta_bike_origin_home"] * vars["origin_home"]        
        + betas["beta_bike_short_distance"] * vars["short_distance"]
        + betas["beta_bike_long_distance"] * vars["long_distance"]
        + betas["beta_bike_densities"] * aggregated_density
    )

    walk_travel_time = (vars["walk_travel_time_min"] / TIME_SCALE_MIN)
    walk_utility = (
        betas["beta_walk_asc"]
        + betas["beta_walk_travel_time_min"] * (walk_travel_time**betas["lambda_walk"])

        + betas["beta_walk_age"] * bioMax(0, vars["age"] - 17)
        + betas["beta_walk_sex"] * vars["sex"]
        + betas["beta_walk_retired"] * vars["is_retired"]
        + betas["beta_walk_junior"] * vars["is_junior"]
        + betas["beta_walk_low_income"] * vars["low_income"]
        + betas["beta_walk_high_income"] * vars["high_income"]

        + betas["beta_walk_destination_work"] * vars["destination_work"]
        + betas["beta_walk_destination_home"] * vars["destination_home"]
        + betas["beta_walk_destination_education"] * vars["destination_education"]
        + betas["beta_walk_destination_shopping"] * vars["destination_shopping"]
        + betas["beta_walk_destination_leisure"] * vars["destination_leisure"]
        + betas["beta_walk_destination_other"] * vars["destination_other"]
        + betas["beta_walk_destination_urban"] * vars["urban_destination"]
        + betas["beta_walk_destination_urbancore"] * vars["urbancore_destination"]
        + betas["beta_walk_region_1"] * vars["region_1"]
        + betas["beta_walk_region_2"] * vars["region_2"]        
        + betas["beta_walk_short_distance"] * vars["short_distance"]
        + betas["beta_walk_long_distance"] * vars["long_distance"]
        + betas["beta_walk_origin_home"] * vars["origin_home"]
        + betas["beta_walk_densities"] * aggregated_density
    )

    if not ignore_car_passenger:
        car_passenger_travel_time = (vars["car_passenger_travel_time_min"] / TIME_SCALE_MIN)
        car_passenger_utility = (
            betas["beta_car_passenger_asc"]            
            + betas["beta_car_passenger_travel_time_min"] * (car_passenger_travel_time**betas["lambda_car_passenger_travel_time"])            
            + betas["beta_car_passenger_distance_km"] * bioMax(0,(vars["car_passenger_distance_km"]-50.0)/DISTANCE_SCALE_KM)
            + betas["beta_car_passenger_driving_permit"] * vars["driving_license"]
            + betas["beta_car_passenger_age"] * bioMax(0, vars["age"] - 17)
            + betas["beta_car_passenger_sex"] * vars["sex"]
            + betas["beta_car_passenger_retired"] * vars["is_retired"]
            + betas["beta_car_passenger_junior"] * vars["is_junior"]
            + betas["beta_car_passenger_low_income"] * vars["low_income"]
            + betas["beta_car_passenger_high_income"] * vars["high_income"]

            + betas["beta_car_passenger_destination_work"] * vars["destination_work"]
            + betas["beta_car_passenger_destination_home"] * vars["destination_home"]
            + betas["beta_car_passenger_destination_education"] * vars["destination_education"]
            + betas["beta_car_passenger_destination_shopping"] * vars["destination_shopping"]
            + betas["beta_car_passenger_destination_leisure"] * vars["destination_leisure"]
            + betas["beta_car_passenger_destination_other"] * vars["destination_other"]
            + betas["beta_car_passenger_destination_urban"] * vars["urban_destination"]
            + betas["beta_car_passenger_destination_urbancore"] * vars["urbancore_destination"]
            + betas["beta_car_passenger_region_1"] * vars["region_1"]
            + betas["beta_car_passenger_region_2"] * vars["region_2"]
            + betas["beta_car_passenger_short_distance"] * vars["short_distance"]
            + betas["beta_car_passenger_long_distance"] * vars["long_distance"]
            + betas["beta_car_passenger_very_long_distance"] * vars["very_long_distance"]
            + betas["beta_car_passenger_origin_home"] * vars["origin_home"]
            + betas["beta_car_passenger_ownership_ratio"] * vars["car_ownership_ratio"]
            + betas["beta_car_passenger_has_car"] * vars["has_car"]
            + betas["beta_car_passenger_densities"] * aggregated_density
        )

    utilities = {
        modes.index("car"): car_utility,
        modes.index("pt"): pt_utility,
        modes.index("bike"): bike_utility,
        modes.index("walk"): walk_utility,
    }
    if not ignore_car_passenger:
        utilities[modes.index("car_passenger")] = car_passenger_utility

    availability = {
        modes.index("car"): vars["car_availability"],
        modes.index("pt"): vars["pt_availability"],
        modes.index("bike"): vars["bike_availability"],
        modes.index("walk"): vars["walk_availability"],
    }
    if not ignore_car_passenger:
        availability[modes.index("car_passenger")] = vars["car_passenger_availability"]

    return utilities, availability

def log_trip_stats(df, modes):
    for m in modes:
        sel = df["mode"].isin([modes.index(m)])
        logger.info("%s : number of trips is %d and average distance is %.2f km", m, sel.sum(), df[sel].euclidean_distance_km.mean())
    pt_or_car = df["mode"].isin([modes.index("car"), modes.index("pt")])
    logger.info("The average euclidean distance is: %.2f km, for pt and car is %.2f km", df.euclidean_distance_km.mean(), df[pt_or_car].euclidean_distance_km.mean())
    logger.info("The average income is: %.2f CHF", df.income.mean())

def execute(context):
    df = context.stage("dmc.data.training_data")
    ignore_car_passenger = context.config("ignore_car_passenger")
    use_exponents = context.config("use_exponents")
    use_income = context.config("use_income_in_dmc")

    df, modes = preprocess_data(df, ignore_car_passenger)
    log_trip_stats(df, modes)

    database = db.Database("data", df)
    vars = define_variables(database, ignore_car_passenger)
    betas = define_betas(ignore_car_passenger, use_exponents, use_income)
    utilities, availability = build_utilities(context, vars, betas, modes, ignore_car_passenger)

    # Training the model
    logprob = models.loglogit(utilities, availability, vars["mode"])
    cwd = os.getcwd()
    os.chdir(context.working_directory)
    biogeme = bio.BIOGEME(database, {"loglike": logprob, "weight": vars["weight"]},
                          numberOfThreads= 8,
                          number_of_jobs = 8)
    biogeme.modelName = "DMC_model"
    biogeme.generate_html = True
    biogeme.generate_pickle = True
    
    null_loglikelihood = biogeme.calculateNullLoglikelihood(availability)
    result = biogeme.estimate()
    os.chdir(cwd)
    
    # Print summary of the results
    logger.info(result.shortSummary())

    # write the optimal parameters to a yaml file in MATSim input format    
    mode_params_path, cost_params_path = writer(context, result).write()

    # write the optimal parameters to a csv file
    csv_params_path = os.path.join(context.path(), "dmc_model_parameters.csv")
    result.getEstimatedParameters().to_csv(csv_params_path)
    logger.info("The estimated parameters are saved to %s", csv_params_path)

    # Compute the VOT for car users
    vot_car, mean_vot_car = vot_utils.get_car_vot(context, df, result, MODES)
    vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer = vot_utils.get_pt_vot(context, df, result, MODES)    

    logger.info("The average VOT for car users is %.2f CHF/hour", mean_vot_car)
    logger.info("The average VOT for pt users is %.2f CHF/hour", mean_vot_pt)

    path_to_figure = os.path.join(context.path(),"vot_distribution.png")
    vot_utils.plot_vot(vot_car, vot_pt, figure_path = path_to_figure)
    logger.info("The VOT distribution figure is saved to %s", path_to_figure)
    
    return (result, 
            df, 
            (mode_params_path, cost_params_path), 
            path_to_figure,
            (vot_car, vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer))