import pandas as pd

"""
Creates a vehicle fleet based on a default vehicle type
"""

def configure(context):
    context.stage("data.cross_border.population")

def execute(context):
    df_persons = context.stage("data.cross_border.population")

    df_vehicles_car = df_persons[["person_id"]].copy()
    df_vehicles_car = df_vehicles_car.rename(columns = { "person_id": "owner_id" })
    
    df_vehicles_car["mode"]       = "car"
    df_vehicles_car["vehicle_id"] = df_vehicles_car["owner_id"].astype(str) + ":car"
    df_vehicles_car["type_id"]    = "default_car"
    df_vehicles_car["age"]        = 0
    df_vehicles_car["euro"]       = 6

    df_vehicles_passenger = df_persons[["person_id"]].copy()
    df_vehicles_passenger = df_vehicles_passenger.rename(columns = { "person_id": "owner_id" })
    
    df_vehicles_passenger["mode"]       = "car_passenger"
    df_vehicles_passenger["vehicle_id"] = df_vehicles_passenger["owner_id"].astype(str) + ":car_passenger"
    df_vehicles_passenger["type_id"]    = "default_car_passenger"
    df_vehicles_passenger["age"]        = 0
    df_vehicles_passenger["euro"]       = 6

    df_cars       = df_vehicles_car
    df_pass       = df_vehicles_passenger

    df_vehicles_person = pd.concat([df_cars, df_pass])

    return df_vehicles_person