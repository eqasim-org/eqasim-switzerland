import numpy as np
from mode_choice.dmc_defaults import Defaults

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.travel_times.car")

    context.config("parking_cost_per_hour_CHF_urban", Defaults.PARKING_COST_PER_HOUR_URBAN) #CHF per hour
    context.config("parking_cost_per_hour_CHF_suburban", Defaults.PARKING_COST_PER_HOUR_SUBURBAN) #CHF per hour


def parking_cost(df,context):
    parking_cost_per_hour_CHF_urban = context.config("parking_cost_per_hour_CHF_urban")
    parking_cost_per_hour_CHF_suburban = context.config("parking_cost_per_hour_CHF_suburban")
    # situations
    destination_urban = df.destination_municipality=="urban"
    destination_suburban = df.destination_municipality=="suburban"
    if "following_purpose" in df.columns:
        destination_home = df.following_purpose=="home"
    else:
        destination_home = df.purpose=="home"    

    # compute parking cost
    parking_cost = np.zeros(len(df))
    pay_parking_urban    = destination_urban & (~destination_home) & (df["parking_duration_min"]>60)
    pay_parking_suburban = destination_suburban & (~destination_home) & (df["parking_duration_min"]>60)

    parking_cost[pay_parking_urban]    = (df["parking_duration_min"][pay_parking_urban]/60.0) * parking_cost_per_hour_CHF_urban
    parking_cost[pay_parking_suburban] = (df["parking_duration_min"][pay_parking_suburban]/60.0) * parking_cost_per_hour_CHF_suburban    
    
    return np.clip(parking_cost, 0, 40)


def execute(context):
    # read the travel times and trips
    travel_times = context.stage("mode_choice.travel_times.car").copy()
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id", "trip_index", "trip_id", "destination_municipality", "following_purpose","departure_time"]]
    df = travel_times.merge(trips, on=["person_id", "trip_id"], how="left")

    # determine arrival time
    total_travel_time_sec = (df["travel_time_min"] + df["access_egress_time_min"]) * 60
    df["arrival_time"] = df["departure_time"] + total_travel_time_sec

    # idensity last activity    
    df["is_last"] = df["person_id"].shift(-1) != df["person_id"]

    # compute activity duration
    df = df.sort_values(by=["person_id", "trip_index"]).reset_index(drop=True)

    df["parking_duration_min"] = (np.clip(df["departure_time"].shift(-1), 8*3600, 19*3600) - 
                                  np.clip(df["arrival_time"], 8*3600, 19*3600)) / 60.0

    df.loc[df["parking_duration_min"]<=0, "parking_duration_min"] = 0.0  # do not pay parking (duration out of bounds)
    df.loc[df["is_last"].values, "parking_duration_min"] = 0.0 # do not pay parking (home parking at night)

    df["parking_duration_min"] = df["parking_duration_min"].clip(0.0, 11 * 60.0)  # ensure max 11 hours (from 8am to 7pm)

    # compute parking cost
    df["parking_cost_CHF"] = parking_cost(df, context)

    return df[["person_id", "trip_id", "parking_cost_CHF"]]