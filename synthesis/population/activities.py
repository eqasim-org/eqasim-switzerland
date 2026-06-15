import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")
"""
Transforms the synthetic trip table into a synthetic activity table.
"""


def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("synthesis.population.trips")


def execute(context):
    df_trips = pd.DataFrame(context.stage("synthesis.population.trips"), copy=True)

    df_trips = df_trips.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    activities = pd.DataFrame({
        "person_id": df_trips["person_id"],
        "activity_index": df_trips["trip_index"],
        "end_time": df_trips["departure_time"],
        "purpose": df_trips["preceding_purpose"],
        "following_mode": df_trips["mode"]
    })

    activities["start_time"] = df_trips.groupby("person_id")["arrival_time"].shift()
    first_trip_idx = df_trips.groupby("person_id")["trip_index"].idxmin()
    activities.loc[first_trip_idx, "start_time"] = 0.0

    last_activities = df_trips.groupby("person_id").tail(1).copy()
    final_activities = pd.DataFrame({
        "person_id": last_activities["person_id"],
        "activity_index": last_activities["trip_index"] + 1,
        "start_time": last_activities["arrival_time"],
        "end_time": 30*3600,
        "purpose": last_activities["following_purpose"],
        "following_mode": None
    })

    df_activities = pd.concat([activities, final_activities], ignore_index=True)
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"]).reset_index(drop=True)
    df_activities["start_time"]     = df_activities["start_time"].fillna(0)

    df_activities["duration"] = df_activities["end_time"] - df_activities["start_time"]
    df_activities["is_last"]  = df_activities["following_mode"].isna()

    # We're still missing activities for people who don't have a any trips
    df_persons = context.stage("synthesis.population.enriched")#[["person_id"]]

    missing_ids = set(np.unique(df_persons["person_id"])) - set(np.unique(df_activities["person_id"]))
    print(f"Found {len(missing_ids)} persons without activities")

    df_missing = pd.DataFrame.from_records([
        (person_id, 0, "home", True) for person_id in missing_ids
    ], columns=["person_id", "activity_index", "purpose", "is_last"])

    df_activities = pd.concat([df_activities, df_missing], sort=True)
    assert (len(np.unique(df_persons["person_id"])) == len(np.unique(df_activities["person_id"])))

    # Truck drivers
    truck_drivers = list(set(np.unique(df_persons[df_persons["is_truck_driver"]][["person_id"]].astype(int))))

    initial_length   = len(df_activities)
    df_activities    = df_activities[~df_activities["person_id"].isin(truck_drivers)]
    df_truck_drivers = pd.DataFrame.from_records([(person_id, 0, "truck_driver", True) for person_id in truck_drivers], columns = ["person_id", "activity_index", "purpose", "is_last"])
    df_activities    = pd.concat([df_activities, df_truck_drivers], sort=True)
    final_length     = len(df_activities)
    share            = round((final_length - initial_length) / initial_length * 100, 2)

    print(f"Removed {initial_length - final_length} ({share}%) activities (truck drivers)")

    assert (len(np.unique(df_persons["person_id"])) == len(np.unique(df_activities["person_id"])))

    # International
    mask = df_persons["is_outside_of_switzerland"].astype("boolean").fillna(False).astype(bool)
    international = df_persons.loc[mask, "person_id"].astype(int).unique().tolist()

    initial_length   = len(df_activities)
    df_activities    = df_activities[~df_activities["person_id"].isin(international)]
    df_international = pd.DataFrame.from_records([(person_id, 0, "international", True) for person_id in international], columns = ["person_id", "activity_index", "purpose", "is_last"])
    df_activities    = pd.concat([df_activities, df_international], sort=True)
    final_length     = len(df_activities)
    share            = round((final_length - initial_length) / initial_length * 100, 2)

    print(f"Removed {initial_length - final_length} ({share}%) activities (people outside of Switzerland)")
    
    assert (len(np.unique(df_persons["person_id"])) == len(np.unique(df_activities["person_id"])))

    # Some cleanup
    df_activities = df_activities.sort_values(by=["person_id", "activity_index"])
    df_activities["start_time"] = df_activities["start_time"].fillna(0)
    df_activities.loc[:, "duration"] = df_activities.loc[:, "end_time"] - df_activities.loc[:, "start_time"]

    df_activities = df_activities[[
        "person_id", "activity_index", "start_time", "end_time", "duration", "purpose", "is_last"
    ]]
    
    return df_activities
