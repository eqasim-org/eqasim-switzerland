import numpy as np
from mode_choice.dmc_defaults import Defaults

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.stage("mode_choice.dmc_defaults")
    context.config("urban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_URBAN)   
    context.config("urbancore_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_URBANCORE)
    context.config("suburban_parking_search_min", default = Defaults.PARKING_SEARCH_MIN_SUBURBAN) 

def parking_search_time(df, context):
    urban_search_min = context.config("urban_parking_search_min")
    urbancore_search_min = context.config("urbancore_parking_search_min")
    suburban_search_min = context.config("suburban_parking_search_min")

    parking_searching_duration_min = np.zeros_like(df["destination_municipality"], dtype=float)

    parking_searching_duration_min[df["destination_municipality"] == "urban"] += urban_search_min
    parking_searching_duration_min[df["destination_municipality"] == "urbancore"] += urbancore_search_min
    parking_searching_duration_min[df["destination_municipality"] == "suburban"] += suburban_search_min

    ### don't search for parking when going home or work
    if "following_purpose" in df.columns:
        parking_searching_duration_min[df["following_purpose"].isin(["home", "work"])] = 0.0
    else:
        parking_searching_duration_min[df["purpose"].isin(["home", "work"])] = 0.0
        
    return parking_searching_duration_min

def execute(context):
    df = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_id","destination_municipality","following_purpose"]
    ].copy()
    df["parking_searching_duration_min"] = parking_search_time(df, context)
    return df[["person_id","trip_id","parking_searching_duration_min"]]