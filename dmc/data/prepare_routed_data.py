import pandas as pd
import numpy as np

def configure(context):
    # Path to the csv file containing the dataset of routed trips  
    context.stage("dmc.data.clean_routed_data")

def execute(context):
    routes = context.stage("dmc.data.clean_routed_data")

    routes["car_travel_time_min"] = routes["travelTime_car"]/60.0
    routes["car_distance_km"] = routes["distance_car"]/1000.0

    routes["car_passenger_travel_time_min"] = routes["travelTime_car"]/60.0
    routes["car_passenger_distance_km"] = routes["distance_car"]/1000.0

    routes["pt_travel_time_min"] = routes["travelTime_pt"]/60.0
    routes["pt_in_vehicle_time_min"] = routes["inVehicleTime_pt"]/60.0
    routes["pt_egress_time_min"] = routes["egressTime_pt"]/60.0
    routes["pt_access_time_min"] = routes["accessTime_pt"]/60.0
    routes["pt_transfer_time_min"] = routes["transferTime_pt"]/60.0
    routes["pt_transfers"] = routes["transfers_pt"]
    routes["pt_in_vehicle_distance_km"] = routes["inVehicleDistance_pt"]/1000
    routes["pt_access_egress_time_min"] = (routes["egressTime_pt"]+ routes["accessTime_pt"])/60.0

    routes["bike_travel_time_min"] = routes["travelTime_bike"]/60.0
    routes["bike_distance_km"] = routes["distance_bike"]/1000.0

    routes["walk_travel_time_min"] = routes["travelTime_walk"]/60.0
    routes["walk_distance_km"] = routes["distance_walk"]/1000.0

    cols = ['person_id','trip_id','mode',
            # car
            'car_travel_time_min', 'car_distance_km',
            # car passenger
            'car_passenger_travel_time_min', 'car_passenger_distance_km',
            # pt
            'pt_travel_time_min', 'pt_in_vehicle_time_min', 'pt_egress_time_min',
            'pt_access_time_min', 'pt_transfer_time_min', 'pt_access_egress_time_min',
            'pt_transfers', 'pt_in_vehicle_distance_km', 'types_pt',
            # walk
            'walk_travel_time_min', 'walk_distance_km',
            # bike
            'bike_travel_time_min', 'bike_distance_km',
            # expected mode usage
            'expectedModeUsed_car','expectedModeUsed_pt','expectedModeUsed_bike','expectedModeUsed_walk',
            ]
    routes = routes[cols]

    return routes