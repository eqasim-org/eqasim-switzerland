import re
import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type for the dummy passenger mode
"""

def configure(context):
    context.stage("synthesis.freight.trips")

def execute(context):
    df_persons = context.stage("synthesis.freight.trips")

    df_vehicle_types = pd.DataFrame.from_records([{
        "type_id": "default_truck", "nb_seats": 1, "length": 12.0, "width": 1.0, "pce": 3.0, "mode": "car_passenger",
        "hbefa_cat": "HEAVY_GOODS_VEHICLE", "hbefa_tech": "average", "hbefa_size": "average", "hbefa_emission": "average",
    }])

    df_vehicles = df_persons[["agent_id"]].copy()
    df_vehicles = df_vehicles.rename(columns = { "agent_id": "owner_id" })
    
    df_vehicles["mode"] = "truck"

    df_vehicles["vehicle_id"] = df_vehicles["owner_id"].astype(str) + ":truck"
    df_vehicles["type_id"] = "default_truck"
    df_vehicles["age"] = 0
    df_vehicles["euro"] = 6

    return df_vehicle_types, df_vehicles