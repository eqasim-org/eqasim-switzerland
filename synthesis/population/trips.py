import numpy as np
import pandas as pd
from numpy import random

import data.constants as c

"""
This stage attaches all trip relevant information to the synthetic population.
"""


def configure(context):
    context.stage("synthesis.population.SNN_population")
    context.stage("data.microcensus.trips")
    context.config("random_seed")
    
    context.config("output_path")
    context.config("scaling_year")
    if context.config("scaling_year") > 2020:
        context.config("car_scaling_factor")


def execute(context):
    
    df_persons = context.stage("synthesis.population.SNN_population")[[
        "person_id", "mz_person_id", "age"
    ]]
    
    # Children do not have any trips from the microcensus
    df_persons = df_persons[df_persons["age"] >= c.MZ_AGE_THRESHOLD]

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips")[0], copy=True)    
    
    df_trips = df_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "activity_duration"
    ]]
    df_trips.columns = ["mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "following_purpose", "activity_duration"]

    # Assume the preceeding purpose for all trips is home
    df_trips["preceding_purpose"] = df_trips["following_purpose"].shift(1)
    df_trips.loc[df_trips["trip_id"] == 1, "preceding_purpose"] = "home"

    df_trips = pd.merge(df_persons, df_trips, on="mz_person_id")
    
    df_trips["mz_person_id"] = df_trips["mz_person_id"].astype(int)
    f = np.isnan(df_trips["mz_person_id"])
    assert ((df_trips[f]["age"] > c.MZ_AGE_THRESHOLD).all())

    # We deliberately delete them here, since other persons also may not have any
    # trips. May be improved later. TODO
    df_trips = df_trips[~f]

    df_trips.loc[:, "travel_time"] = df_trips.loc[:, "arrival_time"] - df_trips.loc[:, "departure_time"]

    df_trips = df_trips[["person_id", "trip_id",
                         "departure_time", "arrival_time",
                         "travel_time", "mode",
                         "preceding_purpose", "following_purpose", "activity_duration"]].sort_values(by=["person_id", "trip_id"])

    # Diversify departure times
    counts = df_trips[["person_id", "trip_id"]].groupby("person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["person_id", "departure_time"]]
                .groupby("person_id")
                .min()
                .reset_index()["departure_time"]
                .values)

    # If first departure time is just 5min after midnight, we only add a deviation of 5min
    interval = np.minimum(1800.0, interval)

    # Set up RNG
    rng = np.random.RandomState(context.config("random_seed"))
    offset = rng.random_sample(size=(len(counts),)) * interval * 2.0 - interval
    offset = np.repeat(offset, counts)

    df_trips["departure_time"] += offset
    df_trips["arrival_time"] += offset
    df_trips["departure_time"] = np.round(df_trips["departure_time"])
    df_trips["arrival_time"] = np.round(df_trips["arrival_time"])
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]

    # Define trip index
    df_trips = df_trips.sort_values(by=["person_id", "trip_id"])
    df_count = df_trips.groupby("person_id").size().reset_index(name="count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])
    
    df_trips.to_csv("%s/trips_matched.csv" % context.config("output_path"), encoding = "latin1", index = False)
    
    print(len(np.unique(df_persons["person_id"])))
    print(len(np.unique(df_trips["person_id"])))
    
    print("Percentage of agents staying home: %d" %( (len(np.unique(df_persons["person_id"])) - len(np.unique(df_trips["person_id"])))/len(np.unique(df_persons["person_id"])) *100))
    
    ### if scaling_year is later than 2020, change some car trips to "pt"
    #if context.config("scaling_year") > 2020:
    #    print(len(df_trips[df_trips["mode"] == "car"]))
    #    car_users = list(np.unique(df_trips[df_trips["mode"] == "car"]["person_id"]))
     #   nb_switchers = int(context.config("car_scaling_factor") * len(car_users))
    #    car_switchers = random.choice(car_users, nb_switchers, replace = False)
    #    df_trips.loc[(df_trips["person_id"].isin(car_switchers)) & (df_trips["mode"] == "car"), "mode"] = "pt" 
     #   print(len(df_trips[df_trips["mode"] == "car"])) 
    

    return df_trips[[
        "person_id", "trip_index",
        "departure_time", "arrival_time",
        "preceding_purpose",
        "following_purpose",
        # "is_first_trip", "is_last_trip",
        "trip_duration",
        "activity_duration",
        "mode"
    ]]


