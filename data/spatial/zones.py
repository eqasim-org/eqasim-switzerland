import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree

def configure(context, require):
    require.stage("data.spatial.countries")
    require.stage("data.spatial.municipalities")
    require.stage("data.spatial.quarters")

def execute(context):
    df_countries = context.stage("data.spatial.countries")
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_quarters = context.stage("data.spatial.quarters")

    df_countries["zone_level_id"] = df_countries["country_id"]
    df_municipalities["zone_level_id"] = df_municipalities["municipality_id"]
    df_quarters["zone_level_id"] = df_quarters["quarter_id"]

    df_countries["zone_name"] = df_countries["country_name"]
    df_municipalities["zone_name"] = df_municipalities["municipality_name"]
    df_quarters["zone_name"] = df_quarters["quarter_name"]

    df_countries["zone_level"] = "country"
    df_municipalities["zone_level"] = "municipality"
    df_quarters["zone_level"] = "quarter"

    df_zones = pd.concat([
        df_countries[["zone_level_id", "zone_name", "zone_level"]],
        df_municipalities[["zone_level_id", "zone_name", "zone_level"]],
        df_quarters[["zone_level_id", "zone_name", "zone_level"]]
    ])

    df_zones.loc[:, "zone_id"] = np.arange(len(df_zones))
    df_zones["zone_level"] = df_zones["zone_level"].astype("category")

    return df_zones[["zone_id", "zone_name", "zone_level", "zone_level_id"]]
