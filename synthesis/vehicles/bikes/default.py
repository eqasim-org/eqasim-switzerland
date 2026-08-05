import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type for the dummy passenger mode
"""

def configure(context):
    context.stage("synthesis.population.enriched")
    context.config("maximum_bike_speed_m_s", default = 8.0)

def execute(context):
    df_persons = context.stage("synthesis.population.enriched")

    df_bikes_types = pd.DataFrame.from_records([{
        "type_id": "default_bike", "nb_seats": 1, "length": 2.0, "width": 0.5, "pce": 0.2, "mode": "bike", 
        "maxVelocity": context.config("maximum_bike_speed_m_s"), "flowEfficiencyFactor": 1.0
    }])

    df_bikes = df_persons[["person_id"]].copy()
    df_bikes = df_bikes.rename(columns = { "person_id": "owner_id" })
    
    df_bikes["mode"] = "bike"

    df_bikes["vehicle_id"] = df_bikes["owner_id"].astype(str) + ":bike"
    df_bikes["type_id"] = "default_bike"   
    df_bikes["age"] = 0
    df_bikes["euro"] = 6

    return df_bikes_types, df_bikes