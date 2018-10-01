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

def impute(df, df_zones):
    print("Imputing %d zones" % len(df))
    remaining_mask = np.ones((len(df),), dtype = np.bool)

    if "quarter_id" in df:
        f = ~np.isnan(df["quarter_id"]) & remaining_mask

        df_join = pd.merge(
            df[f][["quarter_id"]],
            df_zones[df_zones["zone_level"] == "quarter"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = "quarter_id", right_on = "zone_level_id")

        df.loc[f, "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= np.isnan(df["zone_id"])

        print("  Found %d quarters" % np.count_nonzero(df["zone_level"] == "quarter"))

    if "municipality_id" in df:
        f = ~np.isnan(df["municipality_id"]) & remaining_mask

        df_join = pd.merge(
            df[f][["municipality_id"]],
            df_zones[df_zones["zone_level"] == "municipality"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = "municipality_id", right_on = "zone_level_id")

        df.loc[f, "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= np.isnan(df["zone_id"])

        print("  Found %d municipalities" % np.count_nonzero(df["zone_level"] == "municipality"))

    if "country_id" in df:
        f = ~np.isnan(df["country_id"]) & remaining_mask

        df_join = pd.merge(
            df[f][["country_id"]],
            df_zones[df_zones["zone_level"] == "country"][["zone_level_id", "zone_id", "zone_level"]],
            how = "left", left_on = "country_id", right_on = "zone_level_id")

        df.loc[f, "zone_id"] = df_join.loc[:, "zone_id"].values
        df.loc[f, "zone_level"] = df_join.loc[:, "zone_level"].values
        remaining_mask &= np.isnan(df["zone_id"])

        print("  Found %d countries" % np.count_nonzero(df["zone_level"] == "country"))

    unknown_count = np.count_nonzero(np.isnan(df["zone_id"]))

    if unknown_count > 0:
        print("  No information for %d observations" % unknown_count)

    return df
