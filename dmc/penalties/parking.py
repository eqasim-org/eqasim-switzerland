import numpy as np


def get_parking_search_min(df, context):
    urbancore_search_min = context.config("urbancore_parking_search_min")
    urban_search_min = context.config("urban_parking_search_min")
    suburban_search_min = context.config("suburban_parking_search_min")
    
    parking_searching_duration_min = np.zeros_like(df["destination_municipality"], dtype=float)

    parking_searching_duration_min[df["destination_municipality"] == "urbancore"] += urbancore_search_min
    parking_searching_duration_min[df["destination_municipality"] == "urban"] += urban_search_min
    parking_searching_duration_min[df["destination_municipality"] == "suburban"] += suburban_search_min

    ### don't search for parking when going home or work
    parking_searching_duration_min[df["destination_home"]|df["destination_work"]] = 0.0
    
    return parking_searching_duration_min