import pandas as pd
import os
import eqasim.location_assignment as eqla
import os

import eqasim.location_assignment as eqla
import pandas as pd


def configure(context):
    context.stage("matsim.population")
    context.stage("matsim.facilities")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("matsim.java.eqasim")
    context.stage("utils.java")

def execute(context):
    primary_activities = ["home", "work", "education"]

    df_persons = context.stage("data.microcensus.persons")[["person_id", "person_weight"]]
    df_trips = context.stage("data.microcensus.trips")[["person_id", "trip_id", "mode", "crowfly_distance", "departure_time", "arrival_time", "purpose"]]
    df_trips = pd.merge(df_trips, df_persons[["person_id", "person_weight"]], on = "person_id")

    df_trips["travel_time"] = df_trips["arrival_time"] - df_trips["departure_time"]
    df_trips = df_trips[df_trips["travel_time"] > 0.0]
    df_trips = df_trips[df_trips["crowfly_distance"] > 0.0]

    df_trips["following_purpose"] = df_trips["purpose"]
    df_trips["preceeding_purpose"] = df_trips["purpose"].shift(1)
    df_trips.loc[df_trips["trip_id"] == 1, "preceeding_purpose"] = "home"

    df_trips = df_trips[~(
        df_trips["preceeding_purpose"].isin(primary_activities) &
        df_trips["following_purpose"].isin(primary_activities)
    )]

    df_trips = df_trips.rename(columns = {
        "person_weight": "weight",
        "crowfly_distance": "distance"
    })[["mode", "travel_time", "distance", "weight"]]

    eqla.create_input_distributions(
        df_trips, context.cache_path,
        modes = ["car", "pt", "bike", "walk", "car_passenger"],
        resampling_factors = {
            "car": 0.8, "bike": 0.0, "walk": 0.0, "pt": 1.0, "car_passenger": 1.0
        }
    )

    quantiles_path = "%s/quantiles.dat" % context.cache_path
    distributions_path = "%s/distributions.dat" % context.cache_path

    java = context.stage("utils.java")
    input_population_path = context.stage("matsim.population")
    input_facilities_path = context.stage("matsim.facilities")

    output_population_path = "%s/population_with_locations.xml.gz" % context.cache_path
    number_of_threads = context.config["threads"]

    java(
        context.stage("matsim.java.eqasim"),
        "org.eqasim.core.scenario.location_assignment.RunLocationAssignment", [
        "--population-path", input_population_path,
        "--facilities-path", input_facilities_path,
        "--quantiles-path", quantiles_path,
        "--distributions-path", distributions_path,
        "--output-path", output_population_path,
        "--threads", str(number_of_threads),
        "--random-seed", str(0)
    ],cwd = context.cache_path)

    assert(os.path.exists(output_population_path))

    return output_population_path
