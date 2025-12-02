import os
from mode_choice.dmc_defaults import Defaults
from .constants import constants
from .writer import writer
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, bioMax, bioMin, log
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = constants.POSSIBLE_MODES
MEAN_EUCLIDEAN_DISTANCE_KM = constants.MEAN_EUCLIDEAN_DISTANCE_KM
MEAN_INCOME_CHF = constants.MEAN_INCOME_CHF
SHORT_DISTANCE_LIMIT_KM = Defaults.SHORT_DISTANCE_LIMIT_KM
LONG_DISTANCE_LIMIT_KM = Defaults.LONG_DISTANCE_LIMIT_KM

def configure(context):
    context.stage("mode_choice.estimate_model.data.training_data")
    context.config("ignore_car_passenger", default = False)
    context.config("distance_cost_interaction", default = Defaults.DISTANCE_COST_INTERACTION)
    context.config("income_cost_interaction", default = Defaults.INCOME_COST_INTERACTION)
    context.config("urban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_URBAN) #used in the writer
    context.config("suburban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_SUBURBAN) #used in the writer
    context.config("use_exponents", default = Defaults.USE_EXPONENTS_IN_MODE_CHOICE)
    context.stage("mode_choice.dmc_defaults")

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
    df["destination_work"] = df["destination_work"].astype(int)
    df["destination_other"] = df["destination_other"].astype(int)
    df["destination_leisure"] = df["destination_leisure"].astype(int)
    df["origin_home"] = df["origin_home"].astype(int)
    df["sex"] = df["sex"].astype(int)
    df["mode"] = df["mode"].apply(modes.index)
    df["person_weight"] = len(df) * df["person_weight"] / df["person_weight"].sum()
    if not ignore_car_passenger:
        df["is_car_passenger"] = df["is_car_passenger"].astype(int)
    
    municipalities = ['rural', 'suburban', 'urban']
    df['home_municipality'] = df['home_municipality'].apply(municipalities.index).astype(int)
    df['origin_municipality'] = df['origin_municipality'].apply(municipalities.index).astype(int)
    df['destination_municipality'] = df['destination_municipality'].apply(municipalities.index).astype(int)
    df["urban_destination"] = (df["destination_municipality"] == 2).astype(int)

    df["region_1"] = (df["region"]==0).astype(int)
    df["region_2"] = (df["region"]==1).astype(int)
    df["region_3"] = (df["region"]==2).astype(int)

    df["short_distance"] = (df["euclidean_distance_km"]<SHORT_DISTANCE_LIMIT_KM).astype(int) # 80% of bike and walk trips are below 1 km
    df["long_distance"]  = (df["euclidean_distance_km"]>LONG_DISTANCE_LIMIT_KM).astype(int) # 80% of car and pt trips are below 12 km
    return df, modes

def define_variables(database, ignore_car_passenger):
    variables = {
        "mode": db.Variable("mode"),
        "weight": db.Variable("person_weight"),
        "euclidean_distance_km": db.Variable("euclidean_distance_km"),
        "age": db.Variable("age"),
        "income": db.Variable("income"),
        "sex": db.Variable("sex"),
        "driving_license": db.Variable("driving_license"),
        "parking_cost_CHF": db.Variable("parking_cost_CHF"),
        "destination_work": db.Variable("destination_work"),
        "destination_other": db.Variable("destination_other"),
        "destination_leisure": db.Variable("destination_leisure"),
        "destination_municipality": db.Variable("destination_municipality"),
        "urban_destination": db.Variable("urban_destination"),
        "region_2": db.Variable("region_2"),
        "region_3": db.Variable("region_3"),
        "origin_home": db.Variable("origin_home"),        
        "short_distance": db.Variable("short_distance"),
        "long_distance": db.Variable("long_distance"),
        # car
        "car_availability": db.Variable("car_availability"),
        "car_travel_time_min": db.Variable("car_travel_time_min"),
        "car_distance_km": db.Variable("car_distance_km"),
        "car_cost_CHF": db.Variable("car_cost_CHF"),
        "parking_searching_duration_min": db.Variable("parking_searching_duration_min"),
        "car_access_egress_time_min": db.Variable("car_access_egress_time_min"),
        # pt
        "pt_availability": db.Variable("pt_availability"),
        "pt_access_egress_time_min": db.Variable("pt_access_egress_time_min"),
        "pt_in_vehicle_time_min": db.Variable("pt_in_vehicle_time_min"),        
        "pt_transfers": db.Variable("pt_transfers"),
        "pt_waiting_time_min": db.Variable("pt_waiting_time_min"),
        "pt_distance_km": db.Variable("pt_distance_km"),
        "pt_cost_CHF": db.Variable("pt_cost_CHF"),
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

def define_betas(ignore_car_passenger, use_exponents):
    trainable = 0 if use_exponents else 1
    betas = {
        # lambdas
        "lambda_cost_distance": Beta("lambda_cost_distance", -0.1, None, -0.001, 0),
        "lambda_cost_income": Beta("lambda_cost_income", -0.01, None, -0.001, 0),

        "lambda_car_travel_time": Beta("lambda_car_travel_time", 1.0, 0.01, None, trainable),
        "lambda_car_access_egress_time": Beta("lambda_car_access_egress_time", 1.0, 0.01, None, trainable),
        "lambda_pt_in_vehicle_time": Beta("lambda_pt_in_vehicle_time", 1.0, 0.01, None, trainable),
        "lambda_pt_access_egress_time": Beta("lambda_pt_access_egress_time", 1.0, 0.01, None, trainable),
        "lambda_pt_transfers": Beta("lambda_pt_transfers", 1.0, 0.01, None, trainable),
        "lambda_pt_waiting_time": Beta("lambda_pt_waiting_time", 1.0, 0.01, None, 1), #doesn't converge
        "lambda_pt_distance": Beta("lambda_pt_distance", 1.0, 0.01, None, 0),
        "lambda_car_passenger_travel_time": Beta("lambda_car_passenger_travel_time", 1.0, 0.01, None, trainable),            
        "lambda_bike": Beta("lambda_bike", 1.0, 0.01, None, trainable),
        "lambda_walk": Beta("lambda_walk", 1.0, 0.01, None, trainable),

        # cost
        "beta_cost_CHF": Beta("beta_cost_CHF", -0.1, None, -1e-3, 0),        

        # car
        "beta_car_asc": Beta("beta_car_asc", 0, None, None, 0),
        "beta_car_travel_time_min": Beta("beta_car_travel_time_min", -0.01, None, -1e-3, 0),                        
        "beta_car_access_egress_time_min": Beta("beta_car_access_egress_time_min", -0.01, None, 0, 0),
        "beta_car_work_destination": Beta("beta_car_work_destination", 0, None, None, 0),        
        "beta_car_urban_destination": Beta("beta_car_urban_destination", 0, None, None, 0),
        "beta_car_sex": Beta("beta_car_sex", 0, None, None, 0),
        "beta_car_age": Beta("beta_car_age", 0, None, None, 0),
        "beta_car_region_2": Beta("beta_car_region_2", 0, None, None, 0),
        "beta_car_region_3": Beta("beta_car_region_3", 0, None, None, 0),        
        "beta_car_origin_home": Beta("beta_car_origin_home", 0, None, None, 0),        
        "beta_car_short_distance": Beta("beta_car_short_distance", 0, None, None, 0),
        "beta_car_long_distance": Beta("beta_car_long_distance", 0, None, None, 0),
        "beta_car_destination_other": Beta("beta_car_destination_other", 0, None, None, 0),
        "beta_car_destination_leisure": Beta("beta_car_destination_leisure", 0, None, None, 0),
        # pt
        "beta_pt_asc": Beta("beta_pt_asc", 0, None, None, 1),
        "beta_pt_access_egress_time_min": Beta("beta_pt_access_egress_time_min", 0, None, -1e-3, 0),
        "beta_pt_in_vehicle_time_min": Beta("beta_pt_in_vehicle_time_min", -0.05, None, -1e-3, 0),                
        "beta_pt_transfers": Beta("beta_pt_transfers", 0, None, -1e-3, 0),
        "beta_pt_waiting_time_min": Beta("beta_pt_waiting_time_min", 0, None, -1e-3, 0),
        "beta_pt_distance_km": Beta("beta_pt_distance_km", 0, None, None, 0),
        "beta_pt_sex": Beta("beta_pt_sex", 0, None, None, 0),
        "beta_pt_work_destination": Beta("beta_pt_work_destination", 0, None, None, 0),
        "beta_pt_urban_destination": Beta("beta_pt_urban_destination", 0, None, None, 0),
        "beta_pt_age": Beta("beta_pt_age", 0, None, None, 0),
        "beta_pt_region_2": Beta("beta_pt_region_2", 0, None, None, 1),
        "beta_pt_region_3": Beta("beta_pt_region_3", 0, None, None, 1),
        "beta_pt_origin_home": Beta("beta_pt_origin_home", 0, None, None, 0),            
        "beta_pt_short_distance": Beta("beta_pt_short_distance", 0, None, None, 0),
        "beta_pt_long_distance": Beta("beta_pt_long_distance", 0, None, None, 0),
        "beta_pt_destination_other": Beta("beta_pt_destination_other", 0, None, None, 0),
        "beta_pt_destination_leisure": Beta("beta_pt_destination_leisure", 0, None, None, 0),
        #bike
        "beta_bike_asc": Beta("beta_bike_asc", 0, None, None, 0),
        "beta_bike_travel_time_min": Beta("beta_bike_travel_time_min", 0, None, -1e-3, 0),
        "beta_bike_age": Beta("beta_bike_age", 0, None, None, 0),
        "beta_bike_sex": Beta("beta_bike_sex", 0, None, None, 0),        
        "beta_bike_urban_destination": Beta("beta_bike_urban_destination", 0, None, None, 0),
        "beta_bike_region_2": Beta("beta_bike_region_2", 0, None, None, 0),
        "beta_bike_region_3": Beta("beta_bike_region_3", 0, None, None, 0),
        "beta_bike_origin_home": Beta("beta_bike_origin_home", 0, None, None, 0),        
        "beta_bike_short_distance": Beta("beta_bike_short_distance", 0, None, None, 0),        
        "beta_bike_work_destination": Beta("beta_bike_work_destination", 0, None, None, 0),
        "beta_bike_long_distance": Beta("beta_bike_long_distance", 0, None, None, 0),
        "beta_bike_destination_other": Beta("beta_bike_destination_other", 0, None, None, 0),
        "beta_bike_destination_leisure": Beta("beta_bike_destination_leisure", 0, None, None, 0),
        # walk
        "beta_walk_asc": Beta("beta_walk_asc", 0, None, None, 0),
        "beta_walk_travel_time_min": Beta("beta_walk_travel_time_min", 0, None, -1e-3, 0),
        "beta_walk_age": Beta("beta_walk_age", 0, None, None, 0),
        "beta_walk_sex": Beta("beta_walk_sex", 0, None, None, 0),        
        "beta_walk_region_2": Beta("beta_walk_region_2", 0, None, None, 0),
        "beta_walk_region_3": Beta("beta_walk_region_3", 0, None, None, 0),        
        "beta_walk_short_distance": Beta("beta_walk_short_distance", 0, None, None, 0),        
        "beta_walk_origin_home": Beta("beta_walk_origin_home", 0, None, None, 0),
        "beta_walk_work_destination": Beta("beta_walk_work_destination", 0, None, None, 0),
        "beta_walk_urban_destination": Beta("beta_walk_urban_destination", 0, None, None, 0),
        "beta_walk_destination_other": Beta("beta_walk_destination_other", 0, None, None, 0),
        "beta_walk_destination_leisure": Beta("beta_walk_destination_leisure", 0, None, None, 0),
    }
    if not ignore_car_passenger:
        betas.update({
            "beta_car_passenger_asc": Beta("beta_car_passenger_asc", 0, None, None, 0),
            "beta_car_passenger_travel_time_min": Beta("beta_car_passenger_travel_time_min", 0, None, -1e-3, 0),
            "beta_car_passenger_driving_permit": Beta("beta_car_passenger_driving_permit", 0, None, None, 0),            
            "beta_car_passenger_work_destination": Beta("beta_car_passenger_work_destination", 0, None, None, 0),
            "beta_car_passenger_age": Beta("beta_car_passenger_age", 0, None, None, 0),
            "beta_car_passenger_sex": Beta("beta_car_passenger_sex", 0, None, None, 0),
            "beta_car_passenger_urban_destination": Beta("beta_car_passenger_urban_destination", 0, None, None, 0),
            "beta_car_passenger_region_2": Beta("beta_car_passenger_region_2", 0, None, None, 0),
            "beta_car_passenger_region_3": Beta("beta_car_passenger_region_3", 0, None, None, 0),
            "beta_car_passenger_short_distance": Beta("beta_car_passenger_short_distance", 0, None, None, 0),
            "beta_car_passenger_long_distance": Beta("beta_car_passenger_long_distance", 0, None, None, 0),
            "beta_car_passenger_origin_home": Beta("beta_car_passenger_origin_home", 0, None, None, 0),
            "beta_car_passenger_destination_other": Beta("beta_car_passenger_destination_other", 0, None, None, 0),
            "beta_car_passenger_destination_leisure": Beta("beta_car_passenger_destination_leisure", 0, None, None, 0),
        })
    return betas

def build_utilities(context, vars, betas, modes, ignore_car_passenger):
    if context.config("distance_cost_interaction"):
        euclidean_interaction_cost = (vars["euclidean_distance_km"] / MEAN_EUCLIDEAN_DISTANCE_KM) ** betas["lambda_cost_distance"]
    else:
        euclidean_interaction_cost = 1

    if context.config("income_cost_interaction"):
        income_interaction_cost = (vars["income"] / MEAN_INCOME_CHF) ** betas["lambda_cost_income"]
    else:
        income_interaction_cost = 1

    cost_interaction = euclidean_interaction_cost * income_interaction_cost    
    
    car_cost = (vars["car_cost_CHF"] + vars["parking_cost_CHF"])
    car_time = (vars["car_travel_time_min"] + vars["parking_searching_duration_min"])
    transformed_car_time = car_time ** betas["lambda_car_travel_time"]   
    transformed_car_access_egress_time = vars["car_access_egress_time_min"] ** betas["lambda_car_access_egress_time"] 
    car_utility = (
        betas["beta_car_asc"]
        + betas["beta_car_travel_time_min"] * transformed_car_time        
        + betas["beta_car_access_egress_time_min"] * transformed_car_access_egress_time
        + betas["beta_cost_CHF"] * car_cost * cost_interaction

        + betas["beta_car_work_destination"] * vars["destination_work"]
        + betas["beta_car_urban_destination"] * vars["urban_destination"]
        + betas["beta_car_sex"] * vars["sex"]
        + betas["beta_car_age"] * bioMax(0, vars["age"] - 18)
        + betas["beta_car_region_2"] * vars["region_2"]
        + betas["beta_car_region_3"] * vars["region_3"]
        + betas["beta_car_origin_home"] * vars["origin_home"]    
        + betas["beta_car_short_distance"] * vars["short_distance"]
        + betas["beta_car_long_distance"] * vars["long_distance"]
        + betas["beta_car_destination_other"] * vars["destination_other"]
        + betas["beta_car_destination_leisure"] * vars["destination_leisure"]
    )

    transformed_pt_in_vehicle_time = vars["pt_in_vehicle_time_min"]** betas["lambda_pt_in_vehicle_time"]
    transformed_pt_transfers      = vars["pt_transfers"] ** betas["lambda_pt_transfers"]
    transformed_access_egress_time = vars["pt_access_egress_time_min"] ** betas["lambda_pt_access_egress_time"]
    transformed_pt_waiting_time = vars["pt_waiting_time_min"] ** betas["lambda_pt_waiting_time"]
        
    cost_correction = betas["beta_pt_distance_km"] * bioMax(10.0-vars["euclidean_distance_km"], 0.0)**betas["lambda_pt_distance"]
    pt_cost = vars["pt_cost_CHF"] + cost_correction

    pt_utility = (
        betas["beta_pt_asc"]
        + betas["beta_pt_access_egress_time_min"] * transformed_access_egress_time        
        + betas["beta_pt_in_vehicle_time_min"] * transformed_pt_in_vehicle_time
        + betas["beta_pt_waiting_time_min"] * transformed_pt_waiting_time
        + betas["beta_pt_transfers"] * transformed_pt_transfers        
        + betas["beta_cost_CHF"] * pt_cost * cost_interaction

        + betas["beta_pt_sex"] * vars["sex"]
        + betas["beta_pt_age"] * bioMax(0, vars["age"] - 18)
        + betas["beta_pt_work_destination"] * vars["destination_work"]
        + betas["beta_pt_urban_destination"] * vars["urban_destination"]        
        + betas["beta_pt_region_2"] * vars["region_2"]
        + betas["beta_pt_region_3"] * vars["region_3"]
        + betas["beta_pt_origin_home"] * vars["origin_home"]
        + betas["beta_pt_short_distance"] * vars["short_distance"]
        + betas["beta_pt_long_distance"] * vars["long_distance"]
        + betas["beta_pt_destination_other"] * vars["destination_other"]
        + betas["beta_pt_destination_leisure"] * vars["destination_leisure"]
    )

    bike_utility = (
        betas["beta_bike_asc"]
        + betas["beta_bike_travel_time_min"] * (vars["bike_travel_time_min"]**betas["lambda_bike"])

        + betas["beta_bike_age"] * bioMax(0, vars["age"] - 18)
        + betas["beta_bike_sex"] * vars["sex"]
        + betas["beta_bike_urban_destination"] * vars["urban_destination"] 
        + betas["beta_bike_region_2"] * vars["region_2"]
        + betas["beta_bike_region_3"] * vars["region_3"]
        + betas["beta_bike_origin_home"] * vars["origin_home"]        
        + betas["beta_bike_short_distance"] * vars["short_distance"]
        + betas["beta_bike_work_destination"] * vars["destination_work"]
        + betas["beta_bike_long_distance"] * vars["long_distance"]
        + betas["beta_bike_destination_other"] * vars["destination_other"]
        + betas["beta_bike_destination_leisure"] * vars["destination_leisure"]
    )

    walk_utility = (
        betas["beta_walk_asc"]
        + betas["beta_walk_travel_time_min"] * (vars["walk_travel_time_min"]**betas["lambda_walk"])

        + betas["beta_walk_age"] * bioMax(0, vars["age"] - 18)
        + betas["beta_walk_sex"] * vars["sex"]
        + betas["beta_walk_region_2"] * vars["region_2"]
        + betas["beta_walk_region_3"] * vars["region_3"]        
        + betas["beta_walk_short_distance"] * vars["short_distance"]
        + betas["beta_walk_origin_home"] * vars["origin_home"]
        + betas["beta_walk_work_destination"] * vars["destination_work"]
        + betas["beta_walk_urban_destination"] * vars["urban_destination"]
        + betas["beta_walk_destination_other"] * vars["destination_other"]
        + betas["beta_walk_destination_leisure"] * vars["destination_leisure"]
    )

    if not ignore_car_passenger:
        cp_tt = vars["car_passenger_travel_time_min"]**betas["lambda_car_passenger_travel_time"]        
        car_passenger_utility = (
            betas["beta_car_passenger_asc"]            
            + betas["beta_car_passenger_travel_time_min"] * cp_tt            
            + betas["beta_car_passenger_driving_permit"] * vars["driving_license"]
            + betas["beta_car_passenger_work_destination"] * vars["destination_work"]
            + betas["beta_car_passenger_age"] * bioMax(0, vars["age"] - 18)
            + betas["beta_car_passenger_sex"] * vars["sex"]
            + betas["beta_car_passenger_urban_destination"] * vars["urban_destination"]
            + betas["beta_car_passenger_region_2"] * vars["region_2"]
            + betas["beta_car_passenger_region_3"] * vars["region_3"]
            + betas["beta_car_passenger_origin_home"] * vars["origin_home"]
            + betas["beta_car_passenger_short_distance"] * vars["short_distance"]            
            + betas["beta_car_passenger_long_distance"] * vars["long_distance"]
            + betas["beta_car_passenger_destination_other"] * vars["destination_other"]
            + betas["beta_car_passenger_destination_leisure"] * vars["destination_leisure"]
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
    df = context.stage("mode_choice.estimate_model.data.training_data")
    ignore_car_passenger = context.config("ignore_car_passenger")
    use_exponents = context.config("use_exponents")
    df, modes = preprocess_data(df, ignore_car_passenger)
    log_trip_stats(df, modes)

    if 'trip_id' in df.columns and df['trip_id'].dtype != int:
        # replace the trip_id with the trip_index
        df['trip_id'] = df['trip_id'].astype(str).str.split('_').str[1].astype(int)

    database = db.Database("data", df)
    vars = define_variables(database, ignore_car_passenger)
    betas = define_betas(ignore_car_passenger, use_exponents)
    utilities, availability = build_utilities(context, vars, betas, modes, ignore_car_passenger)

    # Training the model (do it in the cache because biogeme stores some files in the current working directory)
    cwd = os.getcwd()
    os.chdir(context.working_directory)

    logprob = models.loglogit(utilities, availability, vars["mode"])
    biogeme = bio.BIOGEME(database, {"loglike": logprob, "weight": vars["weight"]})
    biogeme.modelName = "DMC_model"
    # biogeme.generate_html = False
    # biogeme.generate_pickle = False
    # biogeme.loadSavedIterations = False
    # biogeme.saveIterations = False
    
    null_loglikelihood = biogeme.calculateNullLoglikelihood(availability)
    result = biogeme.estimate()
    os.chdir(cwd)
    
    # Print summary of the results
    logger.info(result.shortSummary())

    # write the optimal parameters to a yaml file in MATSim input format
    try:
        # write parameters to yml format
        path_to_params = os.path.join(context.path(),"model_parameters.yaml")
        writer(context, result, path_to_params).write()
        logger.info("The estimated model parameters have been written to: %s", path_to_params)
        # write parameters statistics to a csv file
        path_to_params_stats = os.path.join(context.path(),"model_parameters_stats.csv")
        result.getEstimatedParameters().to_csv(path_to_params_stats, index=False)        
        logger.info("The estimated model parameters statistics have been written to: %s", path_to_params_stats)

    except Exception as e:
        logger.warning("Could not write the model parameters to a yaml file: %s", e)
        logger.warning("You need to get the output of this stage and check why it failed.")
        path_to_params = None
    
    return (result, df, path_to_params)