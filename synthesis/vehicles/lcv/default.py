import re
import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type for the dummy passenger mode
"""

def configure(context):
    context.stage("synthesis.lcv.trips")

def execute(context):
    df_lcv = context.stage("synthesis.lcv.trips")
    df_vehicle_types = pd.DataFrame.from_records([{
        "type_id": "default_lcv", "nb_seats": 1, "length": 9.0, "width": 1.0, "pce": 1.2, "mode": "truck",
        "hbefa_cat": "HEAVY_GOODS_VEHICLE", "hbefa_tech": "average", "hbefa_size": "average", "hbefa_emission": "average", ##TODO: check hbefa_category
        "maxVelocity": round(130/3.6, 2), "flowEfficiencyFactor": 1.0
    }])

    df_vehicles = df_lcv[["trip_id"]].copy()
    df_vehicles = df_vehicles.rename(columns = { "trip_id": "owner_id" })
    
    df_vehicles["mode"] = "truck"

    df_vehicles["vehicle_id"] = df_vehicles["owner_id"].astype(str) + ":lcv"
    df_vehicles["type_id"] = "default_lcv"
    df_vehicles["age"] = 0
    df_vehicles["euro"] = 6

    return df_vehicle_types, df_vehicles