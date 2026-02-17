import numpy as np
import pandas as pd
import geopandas as gpd

def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")


def execute(context):
    data_path = context.config("data_path")
    file_path = f"{data_path}/vehicle_counts/ETHZ_MBALAC_AK.xlsx"

    vehicle_counts = pd.read_excel(file_path)

    del vehicle_counts["FUEL_TYPE"]
    del vehicle_counts["_TYPE_"]

    vehicle_counts.columns          = ["vehicle_group", "owner_type", "canton_id", "canton_name", "municipality_id", "count"]
    
    vehicle_counts["vehicle_group"] = vehicle_counts["vehicle_group"].map({1: "car", 2: "person transport", 6: "motorcycle"}) # sum up person transport and cars?
    vehicle_counts["owner_type"]    = vehicle_counts["owner_type"].map({1: "person", 3: "company", 4: "unknown"})

    vehicle_counts["vehicle_category"] = vehicle_counts["vehicle_group"] + "_" + vehicle_counts["owner_type"]
    vehicle_counts = vehicle_counts.groupby(["canton_id", "canton_name", "municipality_id", "vehicle_category"])["count"].sum().unstack(fill_value=0).reset_index()

    vehicle_counts["vehicles_person"]  = vehicle_counts["car_person"] + vehicle_counts["motorcycle_person"] + vehicle_counts["person transport_person"]
    vehicle_counts["vehicles_company"] = vehicle_counts["car_company"] + vehicle_counts["motorcycle_company"] + vehicle_counts["person transport_company"]
    vehicle_counts["vehicles_unknown"] = vehicle_counts["car_unknown"] + vehicle_counts["motorcycle_unknown"] + vehicle_counts["person transport_unknown"]
    vehicle_counts["vehicles_total"]   = vehicle_counts["vehicles_person"] + vehicle_counts["vehicles_company"] + vehicle_counts["vehicles_unknown"]

    vehicle_counts["cars_total"]             = vehicle_counts["car_person"] + vehicle_counts["car_company"] + vehicle_counts["car_unknown"]
    vehicle_counts["person transport_total"] = vehicle_counts["person transport_person"] + vehicle_counts["person transport_company"] + vehicle_counts["person transport_unknown"]
    vehicle_counts["cars_like_total"]        = vehicle_counts["cars_total"] #+ vehicle_counts["person transport_total"]
    vehicle_counts["cars_like_person"]       = vehicle_counts["car_person"] #+ vehicle_counts["person transport_person"]

    #municipalities = context.stage("data.spatial.municipalities")[0].copy()

    #map_vehicle_counts =  municipalities.merge(vehicle_counts, left_on = "municipality_id", right_on = "municipality", how = "left")

    #map_vehicle_counts.to_file(f"{context.path()}/number_of_vehicles.gpkg", driver ="GPKG")

    return vehicle_counts



