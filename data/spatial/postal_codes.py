import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df = gpd.read_file(
        "%s/postal_codes/PLZO_SHP_LV95/PLZO_PLZ.shp" % raw_data_path,
        encoding = "latin1"
    ).to_crs({'init': 'EPSG:2056'})

    df["postal_code"] = df["PLZ"]
    df = df.sort_values(by="postal_code").reset_index()
    df = df[["postal_code", "geometry"]]

    return df