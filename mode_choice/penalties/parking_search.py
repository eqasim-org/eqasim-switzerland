import numpy as np

def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.config("urban_parking_search_min", 2.0)   
    context.config("suburban_parking_search_min", 1.0) 

def execute(context):
    df = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id","trip_id","destination_municipality","following_purpose"]
    ]

    urban_search_min = context.config("urban_parking_search_min")
    suburban_search_min = context.config("suburban_parking_search_min")

    parking_searching_duration_min = np.zeros_like(df["destination_municipality"], dtype=float)

    parking_searching_duration_min[df["destination_municipality"] == "urban"] += urban_search_min
    parking_searching_duration_min[df["destination_municipality"] == "suburban"] += suburban_search_min

    ### don't search for parking when going home or work
    parking_searching_duration_min[df["following_purpose"].isin(["home", "work"])] = 0.0

    df["parking_searching_duration_min"] = parking_searching_duration_min
    return df[["person_id","trip_id","parking_searching_duration_min"]]