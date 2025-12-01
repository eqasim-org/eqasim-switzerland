import numpy as np
import pandas as pd
from mode_choice.dmc_defaults import Defaults
from venv import logger
import time
import logging

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")    
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("mode_choice.trips.get_skim_matrices")

    context.config("pt_distance_factor")


def cost_no_zone_model(trips_no_zone, pt_distance_factor):
    euclidean_distance  = lambda x,y: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    distance_home       = lambda x: max(euclidean_distance((x["home_x"], x["home_y"]), (x["origin_x"], x["origin_y"])),
                                        euclidean_distance((x["home_x"], x["home_y"]), (x["destination_x"], x["destination_y"]))) * 1e-3
    
    home_distance = trips_no_zone.apply(distance_home, axis=1)

    in_vehicle_distance_km = (trips_no_zone["euclidean_distance_km"] * pt_distance_factor).values

    cost = np.maximum(2.8, 2 * (2.04 + 0.19 * in_vehicle_distance_km - 0.00011 * in_vehicle_distance_km**2))

    cost[trips_no_zone["hasHalbtaxSubscription"].fillna(False)] *= 0.5
    cost[trips_no_zone["hasGeneralSubscription"].fillna(False)] = 0.0
    cost[trips_no_zone["hasVerbundSubscription"].fillna(False) & (home_distance < Defaults.PT_COST_DISTANCE_THRESHOLD_KM)] = 0.0

    cost[trips_no_zone["age"] <= 6] = 0.0
    cost[trips_no_zone["age"] < 16] *= 0.5
    cost[(trips_no_zone["age"] < 16) & trips_no_zone["hasJuniorSubscription"]] = 0.0
    
    between7and5 = (trips_no_zone["departure_time"] >= 19*3600) | (trips_no_zone["departure_time"] < 5*3600)
    cost[(trips_no_zone["age"] < 25) & trips_no_zone["hasGleis7Subscription"] & between7and5] = 0.0

    return np.clip(np.round(cost, 1), 0, 50)



def execute(context):
    pt_distance_factor = context.config("pt_distance_factor")
    starting_time      = time.time()

    trips    = context.stage("mode_choice.trips.prepare_trips").copy()[
        ["person_id", "trip_id", "departure_time",
         "origin_x", "origin_y", 
         "destination_x", "destination_y", 
         "home_x", "home_y",
         "euclidean_distance_km",
         "origin_zone", "destination_zone"]
        ]
    

    persons  = context.stage("mode_choice.trips.prepare_persons").copy()[
        ["person_id", 
         "hasGeneralSubscription", "hasHalbtaxSubscription",
         "hasVerbundSubscription", "hasStreckenSubscription", 
         "hasJuniorSubscription",  "hasGleis7Subscription", 
         "age"]
        ]
    
    trips    = trips.merge(persons, on = "person_id", how = "left")
    
    matrices = context.stage("mode_choice.trips.get_skim_matrices").copy()[["origin_zone", "destination_zone", "price_halbtax", "price_no_halbtax"]]

    trips = trips.merge(matrices, on = ["origin_zone", "destination_zone"], how = "left")
    
    mask_in_study_area = trips["origin_zone"].notna() & trips["destination_zone"].notna()
    
    trips.loc[mask_in_study_area & trips["hasHalbtaxSubscription"], "cost_CHF"]  = trips[mask_in_study_area & trips["hasHalbtaxSubscription"]]["price_halbtax"].values
    trips.loc[mask_in_study_area & ~trips["hasHalbtaxSubscription"], "cost_CHF"] = trips[mask_in_study_area & ~trips["hasHalbtaxSubscription"]]["price_no_halbtax"].values

    logger.info(f"PT cost computation took {(time.time() - starting_time) / 60:.2f} minutes.")
    
    trips.loc[~mask_in_study_area, "cost_CHF"] = cost_no_zone_model(trips[~mask_in_study_area].copy(), pt_distance_factor)

    trips = trips[["person_id", "trip_id", "cost_CHF"]]

    logger.info(f"PT cost computation took {(time.time() - starting_time) / 60:.2f} minutes.")

    return trips



