import pandas as pd
import os
import matsim.runtime.eqasim as eqasim

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.households")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("calibration.pt_pricing.generate_config")


def execute(context):

    trips = context.stage("data.microcensus.trips")[0].copy()
    persons = context.stage("data.microcensus.persons").copy()
    households = context.stage("data.microcensus.persons").copy()

    trips = trips[["person_id", "trip_id", 
                   "origin_x", "origin_y",
                   "destination_x", "destination_y",
                   "departure_time"]]
    
    persons = persons[["person_id", "age", 
                       "subscriptions_ga", "subscriptions_halbtax",
                       "subscriptions_verbund", "subscriptions_strecke",
                       "subscriptions_gleis7", "subscriptions_junior"]]
    
    households = households[["person_id", "home_x", "home_y"]]

    persons = persons.merge(households, on = "person_id", how = "left")

    trips = trips.merge(persons, on = "person_id", how = "left")

    trips["ID"] = trips["person_id"].astype(str) + "_" + trips["trip_id"].astype(str)

    trips = trips.rename(columns = {
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

    trips = trips[["ID", "originX", "originY", "destinationX", "destinationY",
                   "homeX", "homeY", "departureTime_s",
                   "hasGA", "hasHalbtaxSubscription", "hasVerbundAbo",
                   "hasStreckenAbo", "hasGleis7Abo", "hasJuniorAbo",
                   "age"]]
    
    requests_path = context.path() + "/mzRequests.csv"
    trips.to_csv(requests_path, index = False)

    output_path = context.path() + "/mzRequests_price.csv"
    config_path = context.stage("calibration.pt_pricing.generate_config")

    eqasim.run(context, "org.eqasim.switzerland.ch.utils.pricing.RunComputeTransitPrices",
               ["--config-path", config_path,
               "--requests-path", requests_path,
               "--output-path", output_path]
               )
    
    assert os.path.exists(output_path)    

    result = pd.read_csv(output_path)

    return result