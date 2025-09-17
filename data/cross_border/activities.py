import numpy as np
import pandas as pd
import geopandas as gpd

def configure(context):
    context.config("random_seed")

    context.stage("data.cross_border.destinations")
    context.stage("data.microcensus.trips")

def execute(context):
    df       = context.stage("data.cross_border.destinations").copy()
    mz_trips = context.stage("data.microcensus.trips")[0].copy()

    population = df[["cross_border_person_id", "label", "mz_person_id",
                     "origin_x", "origin_y", 
                     "destination_x", "destination_y", "destination_id"]]
    
    mz_trips   = mz_trips[["person_id", "trip_id", 
                           "departure_time", "arrival_time", 
                           "mode", "purpose"]].rename(columns = {"purpose": "following_purpose"})
    
    mz_trips["preceding_purpose"] = mz_trips["following_purpose"].shift(1)
    mz_trips.loc[mz_trips["trip_id"] == 1, "preceding_purpose"] = "home"

    df_trips = pd.merge(population, mz_trips, left_on="mz_person_id", right_on = "person_id")
    del df_trips["person_id"]

    df_trips = df_trips[["cross_border_person_id", "label", "trip_id",
                         "origin_x", "origin_y",
                         "destination_x", "destination_y",
                         "destination_id",
                         "departure_time", "arrival_time",
                         "mode",
                         "preceding_purpose", "following_purpose"]].sort_values(by=["cross_border_person_id", "trip_id"])
    
    # Diversify departure times
    counts = df_trips[["cross_border_person_id", "trip_id"]].groupby("cross_border_person_id").size().reset_index(name="count")["count"].values

    interval = (df_trips[["cross_border_person_id", "departure_time"]]
                .groupby("cross_border_person_id")
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
    df_trips["arrival_time"]   += offset
    df_trips["departure_time"]  = np.round(df_trips["departure_time"])
    df_trips["arrival_time"]    = np.round(df_trips["arrival_time"])

    # Define trip index
    df_trips = df_trips.sort_values(by=["cross_border_person_id", "trip_id"])
    df_count = df_trips.groupby("cross_border_person_id").size().reset_index(name="count")
    df_trips["trip_index"] = np.hstack([np.arange(count) for count in df_count["count"].values])

    # Adjust origin and destination coordinates
    mask = df_trips["trip_index"] == 1
    df_trips.loc[mask, ["origin_x", "destination_x"]] = df_trips.loc[mask, ["destination_x", "origin_x"]].values
    df_trips.loc[mask, ["origin_y", "destination_y"]] = df_trips.loc[mask, ["destination_y", "origin_y"]].values

    df_trips = df_trips.drop(columns = ["trip_index"])
    df_trips = df_trips.rename(columns = {"cross_border_person_id": "person_id",
                                          "trip_id": "trip_index"})
    
    df_trips = df_trips.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    # For the people going through Switzerland (label = "Through"), only the first trip is relevant
    mask = (df_trips["label"] != "Through") | ((df_trips["label"] == "Through") & (df_trips["trip_index"] == 1))
    df_trips = df_trips[mask]

    activities = pd.DataFrame({
        "person_id": df_trips["person_id"],
        "label": df_trips["label"],
        "activity_index": df_trips["trip_index"],
        "end_time": df_trips["departure_time"],
        "purpose": df_trips["preceding_purpose"],
        "location_x": df_trips["origin_x"],
        "location_y": df_trips["origin_y"],
        "destination_id": None,
        "following_mode": df_trips["mode"]
    })

    activities["start_time"] = df_trips.groupby("person_id")["arrival_time"].shift()
    first_trip_idx = df_trips.groupby("person_id")["trip_index"].idxmin()
    activities.loc[first_trip_idx, "start_time"] = 0.0

    activities["destination_id"] = df_trips.groupby("person_id")["destination_id"].shift()

    last_activities = df_trips.groupby("person_id").tail(1).copy()
    final_activities = pd.DataFrame({
        "person_id": last_activities["person_id"],
        "label": last_activities["label"],
        "activity_index": last_activities["trip_index"] + 1,
        "start_time": last_activities["arrival_time"],
        "end_time": 30*3600,
        "purpose": last_activities["following_purpose"],
        "location_x": last_activities["destination_x"],
        "location_y": last_activities["destination_y"],
        "destination_id": None,
        "following_mode": None
    })

    df_activities = pd.concat([activities, final_activities], ignore_index=True)
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"]).reset_index(drop=True)
    df_activities["start_time"]     = df_activities["start_time"].fillna(0)
    df_activities["destination_id"] = df_activities["destination_id"].fillna(-1)

    geometry = gpd.GeoSeries.from_xy(df_activities["location_x"], df_activities["location_y"])

    df_activities["geometry"] = geometry
    df_activities["duration"] = df_activities["end_time"] - df_activities["start_time"]
    df_activities["is_last"]  = df_activities["following_mode"].isna()

    df_activities = df_activities[["person_id", "activity_index", "label",
                                   "start_time", "end_time", "duration",
                                   "purpose", "is_last",
                                   "geometry", "destination_id", "following_mode"
                                   ]]

    return df_activities