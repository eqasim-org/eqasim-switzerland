"""
This is a simple model to estimate pt cost. 
Aurore will impliment a better one later.
"""
import numpy as np
import pandas as pd


def configure(context):
    context.stage("mode_choice.trips.prepare_trips")    
    context.stage("synthesis.population.enriched")
    context.config("pt_distance_factor", 1.4) #CHF per hour









################# OLD MODEL BELOW, TO BE REPLACED LATER #######################
euc_distance = lambda x,y: np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

homdistance = lambda x: max(euc_distance((x['home_x'],x['home_y']),(x['origin_x'],x['origin_y'])),
                            euc_distance((x['home_x'],x['home_y']),(x['destination_x'],x['destination_y'])))*1e-3

DISTANCE_THRESHOLD_KM = 10.0

def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id", "trip_index","trip_id","crowfly_distance","departure_time",
         "origin_x","origin_y", "destination_x", "destination_y", "home_x","home_y",]]
    
    # get subscriptions and age
    df_persons = context.stage("synthesis.population.enriched")
    df_persons["hasGeneralSubscription"] = df_persons.subscriptions_ga
    df_persons["hasHalbtaxSubscription"] = df_persons.subscriptions_halbtax
    df_persons["hasRegionalSubscription"] = df_persons.subscriptions_verbund | df_persons.subscriptions_strecke
    df_persons["hasJuniorSubscription"] = df_persons.subscriptions_junior
    df_persons["hasGleis7Subscription"] = df_persons.subscriptions_gleis7
    
    df_persons = df_persons[["person_id", "hasGeneralSubscription","hasHalbtaxSubscription",
                             "hasRegionalSubscription", "hasJuniorSubscription",
                             "hasGleis7Subscription", 'age']]

    df = df.merge(df_persons, on="person_id", how="left")
    assert not df["age"].isna().any(), "Some persons have no age!"

    # compute the cost
    homeDistance_km = df.apply(homdistance, axis=1)
    in_vehicle_distance_km = df["crowfly_distance"] / 1000 * context.config("pt_distance_factor")    
    
    cost = np.maximum(2.8, 2*(0.21 * in_vehicle_distance_km - 0.00015 * in_vehicle_distance_km**2))
    
    #### cases with subscriptions, and age
    cost[df["hasHalbtaxSubscription"].fillna(False)] *= 0.5
    cost[df["hasGeneralSubscription"].fillna(False)] = 0.0
    cost[df["hasRegionalSubscription"].fillna(False) & (homeDistance_km < DISTANCE_THRESHOLD_KM)] = 0.0

    cost[df["age"]<=6] = 0.0
    cost[df["age"]<16] *= 0.5
    cost[(df["age"]<16)&df["hasJuniorSubscription"]] = 0.0
    
    between7and5 = (df["departure_time"]>=19*3600) | (df["departure_time"]<5*3600)
    cost[(df["age"]<25)&df["hasGleis7Subscription"]&between7and5] = 0.0

    ### Limit pt cost per person per day to 50 CHF, split among their trips (daily pass)
    return np.clip(cost,0,50)