import pandas as pd

def configure(context):
    method = context.config("vehicles_method", "default")

    if method == "default":
        context.stage("synthesis.vehicles.cars.default", alias = "cars")
    else:
        raise RuntimeError("Unknown vehicles generation method : %s" % method)
    
    context.stage("synthesis.vehicles.passengers.default")
    context.stage("synthesis.vehicles.trucks.default")

def execute(context):
    df_car_types, df_cars = context.stage("cars")
    df_passenger_types, df_passengers = context.stage("synthesis.vehicles.passengers.default")
    df_truck_types, df_trucks = context.stage("synthesis.vehicles.trucks.default")

    df_vehicles_person = pd.concat([df_cars, df_passengers])
    df_types = pd.concat([df_car_types, df_passenger_types, df_truck_types])
    
    return df_types, df_vehicles_person, df_trucks