import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df_nuts = gpd.read_file(
        "%s/nuts_borders/ref-nuts-2013-01m.shp/NUTS_RG_01M_2013_4326.shp/NUTS_RG_01M_2013_4326.shp" % raw_data_path,
        encoding = "utf-8"
    )
    df_nuts.crs = {'init' :'EPSG:4326'}
    df_nuts = df_nuts.to_crs({'init': 'EPSG:2056'})

    df_nuts["nuts_id"] = df_nuts["NUTS_ID"]
    df_nuts["nuts_name"] = df_nuts["NUTS_NAME"]
    df_nuts["nuts_level"] = df_nuts["LEVL_CODE"]
    df_nuts = df_nuts.sort_values(by=["nuts_id", "nuts_level"]).reset_index()
    df_nuts = df_nuts[["nuts_id", "nuts_name", "nuts_level", "geometry"]]

    return df_nuts
