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

    vehicle_counts.columns          = ["vehicle_group", "owner_type", "canton_id", "canton_name", "municipality", "count"]
    
    vehicle_counts["vehicle_group"] = vehicle_counts["vehicle_group"].map({1: "cars", 2: "person transport", 6: "motorcycles"}) # sum up person transport and cars?
    vehicle_counts["owner_type"]    = vehicle_counts["owner_type"].map({1: "person", 3: "company", 4: "unknown"})

    vehicle_counts["vehicle_category"] = vehicle_counts["vehicle_group"] + "_" + vehicle_counts["owner_type"]
    vehicle_counts = vehicle_counts.groupby(["canton_id", "canton_name", "municipality", "vehicle_category"])["count"].sum().unstack(fill_value=0).reset_index()

    municipalities = context.stage("data.spatial.municipalities")[0].copy()

    map_vehicle_counts =  municipalities.merge(vehicle_counts, left_on = "municipality_id", right_on = "municipality", how = "left")

    map_vehicle_counts.to_file(f"{context.path()}/number_of_vehicles.gpkg", driver ="GPKG")

    return vehicle_counts



