import os
import matsim.runtime.eqasim as eqasim
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("dmc.cost.pt")

def get_cost(df, context, pt_regional_radius_km):
    df = df.copy()
    # get the prices dataframe
    prices = estimate_cost_from_eqasim_java(df, context, pt_regional_radius_km)
    prices = prices[["id","price"]]
    prices[["person_id", "trip_id"]] = prices["id"].str.extract(r'([^_]+)_(.+)')
    prices["person_id"] = prices["person_id"].astype(df["person_id"].dtype)
    prices["trip_id"] = prices["trip_id"].astype(df["trip_id"].dtype)
    prices.rename(columns={"price":"pt_cost_CHF"}, inplace=True)
    
    # merge with the trips dataframe
    df = df.merge(prices[["person_id", "trip_id", "pt_cost_CHF"]], on = ["person_id", "trip_id"], how = "left")
    
    # fill nans with the simple model
    f_nan = df["pt_cost_CHF"].isna()
    num_nans = f_nan.sum()
    if num_nans>0:
        logger.info(f"{num_nans} out of {len(df)} trips have no price estimation from the detailed model, using the simple model instead.")
        df.loc[f_nan, "pt_cost_CHF"] = estimate_simple_cost(df[f_nan], context, pt_regional_radius_km)
    
    return df["pt_cost_CHF"]




def estimate_cost_from_eqasim_java(df, context, pt_regional_radius_km):
    df = df.copy()
    df_persons = context.stage("data.microcensus.persons")

    df = df[["person_id", "trip_id","origin_x", "origin_y", "destination_x", "destination_y", 
             "departure_time","home_x", "home_y","age"]]

    persons = df_persons[["person_id", "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
                       "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior"]]
    
    df = df.merge(persons, on="person_id", how="left")

    logger.warning("Gleis 7 subscription is not considered in the pt cost calculation, all persons are considered as not having it.")
    logger.warning("Thisis because this subscription is not considered by the subscriptions model, and thus are not considered in the simulation.")
    df["subscriptions_gleis7"] = False

    df["ID"] = df["person_id"].astype(str) + "_" + df["trip_id"].astype(str)
    df = df.rename(columns = {
        "origin_x": "originX",
        "origin_y": "originY",
        "destination_x": "destinationX",
        "destination_y": "destinationY",
        "home_x": "homeX",
        "home_y": "homeY",
        "departure_time": "departureTime_s",
        "subscriptions_ga": "hasGA",
        "subscriptions_halbtax": "hasHalbtaxSubscription",
        "subscriptions_verbund": "hasVerbundAbo",
        "subscriptions_strecke": "hasStreckenAbo",
        "subscriptions_gleis7": "hasGleis7Abo",
        "subscriptions_junior": "hasJuniorAbo"
    })

    df = df[["ID", "originX", "originY", "destinationX", "destinationY",
             "homeX", "homeY", "departureTime_s", "age",
             "hasGA", "hasHalbtaxSubscription", "hasVerbundAbo",
             "hasStreckenAbo", "hasGleis7Abo", "hasJuniorAbo",]]
    
    requests_path = context.path() + "/mzPricesRequests.csv"
    df.to_csv(requests_path, index = False)

    output_path = context.path() + "/mzPricesRequests_done.csv"
    config_path = context.stage("calibration.pt_pricing.generate_config")    

    # check if the file existst and if it contains all the requests
    if os.path.exists(output_path):
        df_out = pd.read_csv(output_path)
        ## check if all requests are there and if all prices are computed
        if ((len(df_out) == len(df)) and 
            (df_out["ID"].equals(df["ID"])) and 
            (df_out["departureTime_s"].equals(df["departureTime_s"]))):
            return df_out

    eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.utils.pricing.RunComputeTransitPrices",
               ["--config-path", config_path,
               "--requests-path", requests_path,
               "--output-path", output_path,
               "--ptRegionalRadius_km", str(pt_regional_radius_km)],
               )
    
    assert os.path.exists(output_path)    

    result = pd.read_csv(output_path)
    return result




############### SIMPLE MODEL ###############

euc_distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

homdistance = lambda x: max(euc_distance((x['home_x'],x['home_y']),(x['origin_x'],x['origin_y'])),
                            euc_distance((x['home_x'],x['home_y']),(x['destination_x'],x['destination_y'])))*1e-3

def estimate_simple_cost(df, context, pt_regional_radius_km):

    homeDistance_km = df.apply(homdistance, axis=1)
    in_vehicle_distance_km = df.pt_in_vehicle_distance_km    
    
    cost = np.maximum(2.8, 2*(0.21 * in_vehicle_distance_km - 0.00015 * in_vehicle_distance_km**2)) # cost = np.maximum(2.0, 0.6 * in_vehicle_distance_km)    
    
    #### cases with subscriptions, and age    
    cost[df["hasGeneralSubscription"].fillna(False)] = 0.0
    cost[df["age"]<=6] = 0.0 # (source: https://www.sbb.ch/en/travel-information/individual-needs/travelling-with-children/tickets-travelcards.html#:~:text=The%20Junior%20Travelcard%20enables%20children,a%20valid%20ticket%20or%20travelcard.)
    cost[df["hasRegionalSubscription"].fillna(False) & (homeDistance_km <= pt_regional_radius_km)] = 0.0    
    cost[(df["age"]<16)&df["hasJuniorSubscription"]] = 0.0
    
    between7and5 = (df["departure_time"]>=19*3600) | (df["departure_time"]<5*3600) # (source: https://www.sbb.ch/en/tickets-offers/travelcards/ga-travelcard/night-ga-travelcard.html)
    cost[(df["age"]<25)&df["hasGleis7Subscription"]&between7and5] = 0.0

    half_fare_tariff = (df["age"]<16) | (df["hasHalbtaxSubscription"].fillna(False))
    cost[half_fare_tariff] *= 0.5        

    ### Limit pt cost per person per day (they usually pay for the tour for long distances)
    maximum_cost = 60 - 25 * half_fare_tariff.astype(int) # it is 60 CHF for people without halbtax and 35 CHF for people with halbtax
    return np.clip(cost,0,maximum_cost)