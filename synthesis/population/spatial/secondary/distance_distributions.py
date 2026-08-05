import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")


def calculate_bounds(values, bin_size):
    values = np.sort(values)

    bounds = []
    current_count = 0
    previous_bound = None

    for value in values:
        if value == previous_bound:
            continue

        if current_count < bin_size:
            current_count += 1
        else:
            current_count = 0
            bounds.append(value)
            previous_bound = value

    if len(bounds) > 0:
        bounds[-1] = np.inf
    else:
        bounds.append(np.inf)

    return bounds


def execute(context):
    df_persons = context.stage("data.microcensus.persons")[["person_id", "person_weight", "weekend"]].rename(
        columns={"person_weight": "weight"})
    df_trips = context.stage("data.microcensus.trips")[0][["person_id", "trip_id", "mode", "crowfly_distance",
                                                             "departure_time", "arrival_time", "purpose", "origin_purpose"]]
    # Filter out weekend trips
    weekend_persons = df_persons[df_persons["weekend"]]["person_id"].unique()
    df_trips = df_trips[~df_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    df_persons = df_persons[~df_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)
    
    df_trips = pd.merge(df_trips, df_persons[["person_id", "weight"]], on="person_id")
    df_trips["travel_time"] = df_trips["arrival_time"] - df_trips["departure_time"]

    df_trips = df_trips[df_trips["travel_time"] > 0.0]
    #df_trips = df_trips[df_trips["crowfly_distance"] > 0.0]

    df_trips["following_purpose"] = df_trips["purpose"]
    df_trips["preceding_purpose"] = df_trips["origin_purpose"]

    primary_activities = ["home", "work", "education"]
    df_trips = df_trips[~(df_trips["preceding_purpose"].isin(primary_activities)
                          & df_trips["following_purpose"].isin(primary_activities))]
    
    distance_column = "crowfly_distance" if "crowfly_distance" in df_trips else "network_distance"
    df = df_trips[["mode", "travel_time", distance_column, "weight"]].rename(columns={distance_column: "distance"})

    modes = df["mode"].unique()
    logger.info(f"INFO: computing distance distributions for the following modes: {modes}")

    bin_size = 200
    distributions = {}

    for mode in modes:
        f_mode = df["mode"] == mode
        bounds = calculate_bounds(df[f_mode]["travel_time"].values, bin_size)

        distributions[mode] = dict(bounds=np.array(bounds), distributions=[])

        for lower_bound, upper_bound in zip([-np.inf] + bounds[:-1], bounds):
            f_bound = (df["travel_time"] > lower_bound) & (df["travel_time"] <= upper_bound)

            values  = df[f_mode & f_bound]["distance"].values
            weights = df[f_mode & f_bound]["weight"].values

            sorter  = np.argsort(values)
            values  = values[sorter]
            weights = weights[sorter]

            cdf = np.cumsum(weights)
            cdf /= cdf[-1] 

            distributions[mode]["distributions"].append(dict(cdf=cdf, values=values, weights=weights))
            
    return distributions
