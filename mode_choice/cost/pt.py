"""
This is a simple model to estimate pt cost. 
Aurore will impliment a better one later.
"""
from venv import logger
import numpy as np
import pandas as pd
import os
from mode_choice.dmc_defaults import Defaults
import matsim.runtime.eqasim as eqasim

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")    
    context.stage("mode_choice.trips.prepare_persons")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    
    context.config("data_path")
    context.config("dmc_simulation_data_path", default = os.path.join(context.config("data_path"), "simulation_data"))    
    context.config("dmc_matsim_config_file", default=os.path.join(context.config("dmc_simulation_data_path"), "matsim_config.xml"))

    context.config("pt_distance_factor", default=Defaults.DEFAULT_PT_DISTANCE_FACTOR)

################# OLD MODEL #######################
euc_distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

homdistance = lambda x: max(euc_distance((x['home_x'],x['home_y']),(x['origin_x'],x['origin_y'])),
                            euc_distance((x['home_x'],x['home_y']),(x['destination_x'],x['destination_y'])))*1e-3

DISTANCE_THRESHOLD_KM = Defaults.PT_COST_DISTANCE_THRESHOLD_KM

def pt_cost_simple(context, df, distance_threshold_km=DISTANCE_THRESHOLD_KM):
    # compute the cost
    homeDistance_km = df.apply(homdistance, axis=1)
    in_vehicle_distance_km = (df["euclidean_distance_km"] * context.config("pt_distance_factor")).values
    
    cost = np.maximum(2.8, 2*(0.21 * in_vehicle_distance_km - 0.00015 * in_vehicle_distance_km**2))
    
    #### cases with subscriptions, and age
    cost[df["hasHalbtaxSubscription"].fillna(False)] *= 0.5
    cost[df["hasGeneralSubscription"].fillna(False)] = 0.0
    cost[df["hasRegionalSubscription"].fillna(False) & (homeDistance_km < distance_threshold_km)] = 0.0

    cost[df["age"]<=6] = 0.0
    cost[df["age"]<16] *= 0.5
    cost[(df["age"]<16)&df["hasJuniorSubscription"]] = 0.0
    
    between7and5 = (df["departure_time"]>=19*3600) | (df["departure_time"]<5*3600)
    cost[(df["age"]<25)&df["hasGleis7Subscription"]&between7and5] = 0.0

    ### Limit pt cost per person per day to 50 CHF, split among their trips (daily pass)
    return np.clip(cost,0,50)

################# NEW MODEL #######################
def pt_cost(context, df):    
    df = df.copy()
    df = df.rename(columns = {
            "trip_id": "ID",
            "origin_x": "originX",
            "origin_y": "originY",
            "destination_x": "destinationX",
            "destination_y": "destinationY",
            "home_x": "homeX",
            "home_y": "homeY",
            "departure_time": "departureTime_s",
            "hasGeneralSubscription": "hasGA",
            "hasHalbtaxSubscription": "hasHalbtaxSubscription",
            "hasVerbundSubscription": "hasVerbundAbo",
            "hasStreckenSubscription": "hasStreckenAbo",
            "hasGleis7Subscription": "hasGleis7Abo",
            "hasJuniorSubscription": "hasJuniorAbo"
    })

    df = df[["ID", "originX", "originY", "destinationX", "destinationY",
             "homeX", "homeY", "departureTime_s", "age",
             "hasGA", "hasHalbtaxSubscription", "hasVerbundAbo",
             "hasStreckenAbo", "hasGleis7Abo", "hasJuniorAbo",]]
    
    requests_path = context.path() + "/PricesRequests.csv"
    df.to_csv(requests_path, index = False)

    output_path = context.path() + "/PricesRequests_done.csv"
    config_path = context.config("dmc_matsim_config_file")    

    eqasim.run(context, "org.eqasim.switzerland.ch.utils.pricing.RunComputeTransitPrices",
               ["--config-path", config_path,
               "--requests-path", requests_path,
               "--output-path", output_path]
               )
    
    assert os.path.exists(output_path)    

    result = pd.read_csv(output_path)[["id","price"]]    
    df = df.merge(result, left_on="ID", right_on="id", how="left")
    assert not df["price"].isna().any(), "Some trips have no price computed"
    return df["price"].values


################# COMPUTE COST FOR SYNTHETIC TRIPS #######################
def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id", "trip_id","euclidean_distance_km","departure_time",
         "origin_x","origin_y", "destination_x", "destination_y", "home_x","home_y",]]
    
    # get subscriptions and age
    df_persons = context.stage("mode_choice.trips.prepare_persons")[
        ["person_id", "hasGeneralSubscription","hasHalbtaxSubscription",
         "hasRegionalSubscription", "hasJuniorSubscription", "hasVerbundSubscription", 
         "hasStreckenSubscription", "hasGleis7Subscription", 'age']]

    df = trips.merge(df_persons, on="person_id", how="left")
    assert not df["age"].isna().any(), "Some persons have no age!"

    # compute the cost    
    df["cost_MU"] = pt_cost(context, df)    
    
    # fill nans with the simple model
    f_nan = df["cost_MU"].isna()
    num_nans = f_nan.sum()
    if num_nans>0:
        logger.info(f"{num_nans} trips have no price estimation from the detailed model, using the simple model instead.")
        df.loc[f_nan, "cost_MU"] = pt_cost_simple(context,df[f_nan])
    
    return df[["person_id","trip_id","cost_MU"]]