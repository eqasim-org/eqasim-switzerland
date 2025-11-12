import numpy as np

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.travel_times.car")
    context.stage("mode_choice.trips.prepare_trips")

    context.config("parking_cost_per_hour_CHF_urban", 1.0) #CHF per hour
    context.config("parking_cost_per_hour_CHF_suburban", 0.5) #CHF per hour


def execute(context):
    # read the travel times
    travel_times = context.stage("mode_choice.travel_times.car").copy()
    
    # determine arrival time
    travel_times["arrival_time"] = travel_times["departure_time"] + travel_times["total_travel_time"]

    # idensity last activity    
    travel_times["is_last"] = travel_times["person_id"].shift(-1) != travel_times["person_id"]

    # compute activity duration
    travel_times = travel_times.sort_values(
        by=["person_id", "trip_index"]
    ).reset_index(drop=True)
    
    travel_times["parking_duration_min"] = (np.clip(travel_times["departure_time"].shift(-1), 8*3600, 19*3600) - 
                                            np.clip(travel_times["arrival_time"], 8*3600, 19*3600)) / 60.0

    travel_times.loc[travel_times["parking_duration_min"]<=0, "parking_duration_min"] = 0.0  # do not pay parking (duration out of bounds)
    travel_times.loc[travel_times["is_last"].values, "parking_duration_min"] = 0.0 # do not pay parking (home parking at night)

    travel_times["parking_duration_min"] = travel_times["parking_duration_min"].clip(0.0, 11 * 60.0)  # ensure max 11 hours (from 8am to 7pm)

    # determine destination type
    trips = context.stage("mode_choice.trips.prepare_trips")[["person_id", "trip_index", "destination_municipality", "following_purpose"]]
    df = travel_times.merge(trips, on=["person_id", "trip_index"], how="left")

    # situations
    destination_urban = df.destination_municipality=="urban"
    destination_suburban = df.destination_municipality=="suburban"
    destination_home = df.following_purpose=="home"    

    # compute parking cost
    parking_cost = np.zeros(len(df))
    pay_parking_urban    = destination_urban & (~destination_home) & (df["parking_duration_min"]>60)
    pay_parking_suburban = destination_suburban & (~destination_home) & (df["parking_duration_min"]>60)

    parking_cost[pay_parking_urban]    = (df["parking_duration_min"][pay_parking_urban]/60.0) * context.config("parking_cost_per_hour_CHF_urban")
    parking_cost[pay_parking_suburban] = (df["parking_duration_min"][pay_parking_suburban]/60.0) * context.config("parking_cost_per_hour_CHF_suburban")    
    
    df["parking_cost"] = np.clip(parking_cost, 0, 40)
    
    return df[["person_id", "trip_index", "parking_cost"]]