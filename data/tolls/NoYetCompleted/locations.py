import os
import geopandas as gpd


def configure(context):
    context.config("data_path")
    context.config("tolls_locations_file", "gares-peage-2025-shp.zip")

def execute(context):
    file_path = os.path.join(context.config("data_path"), "tolls", context.config("tolls_locations_file"))
    df = gpd.read_file(file_path).to_crs("EPSG:2056")
    
    # rename some columns
    df = df.rename(columns={
        "nomGare": "station_name",
        "nbVoies": "number_of_lanes",
    })

    # correct names (often é is replaceed by ? in the data, I need to correct that)
    df["station_name"] = df["station_name"].str.replace("?", "e", regex=False)
    df["typeGare_l"] = df["typeGare_l"].str.replace("?", "e", regex=False)

    return df




