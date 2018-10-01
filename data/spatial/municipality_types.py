import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree

def configure(context, require):
    require.config("raw_data_path")
    require.stage("data.spatial.municipalities")

def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    print("  Reading structure information ...")
    df_structure = pd.read_excel(
        "%s/spatial_structure_2018.xlsx" % raw_data_path,
        skiprows = (0, 1, 2, 4, 5, 6)
    )[["BFS Nr.", "Raum mit städtischem Charakter 2012"]]
    df_structure.columns = ["municipality_id", "spatial_type_raw"]

    # Rewrite classification
    df_structure.loc[df_structure["spatial_type_raw"] == 1, "spatial_type"] = "urban"
    df_structure.loc[df_structure["spatial_type_raw"] == 2, "spatial_type"] = "urban"
    df_structure.loc[df_structure["spatial_type_raw"] == 3, "spatial_type"] = "suburban"
    df_structure.loc[df_structure["spatial_type_raw"] == 4, "spatial_type"] = "suburban"
    df_structure.loc[df_structure["spatial_type_raw"] == 5, "spatial_type"] = "rural"
    df_structure.loc[df_structure["spatial_type_raw"] == 6, "spatial_type"] = "rural"
    df_structure.loc[df_structure["spatial_type_raw"] == 0, "spatial_type"] = "rural"

    df_existing = pd.merge(df_municipalities, df_structure, on ="municipality_id")
    df_existing["imputed_spatial_type"] = False
    df_existing = df_existing[["municipality_id", "spatial_type", "imputed_spatial_type", "geometry"]]

    # Some ids are missing (because they are special zones)
    df_missing = gpd.GeoDataFrame(df_municipalities[
        ~df_municipalities["municipality_id"].isin(df_existing["municipality_id"])
    ])
    df_missing.crs = df_municipalities.crs
    df_missing = df_missing[["municipality_id", "geometry"]]

    print("Imputing %d spatial types by distance..." % len(df_missing))
    coordinates = np.vstack([df_existing["geometry"].centroid.x, df_existing["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance = False).flatten()

    df_missing.loc[:, "spatial_type"] = df_existing.iloc[indices]["spatial_type"].values
    df_missing.loc[:, "imputed_spatial_type"] = True
    df_missing = df_missing[["municipality_id", "spatial_type", "imputed_spatial_type", "geometry"]]

    df_mapping = pd.concat((df_existing, df_missing))

    assert(len(df_mapping) == len(df_municipalities))
    assert(set(np.unique(df_mapping["municipality_id"])) == set(np.unique(df_municipalities["municipality_id"])))

    df_mapping = pd.DataFrame(df_mapping[["municipality_id", "spatial_type", "imputed_spatial_type"]])
    df_mapping["spatial_type"] = df_mapping["spatial_type"].astype("category")

    return df_mapping

def impute(df, df_municipality_types, remove_unknown = False):
    assert("municipality_id" in df.columns)
    df = pd.merge(df, df_municipality_types, on = "municipality_id")

    if remove_unknown:
        return df[~np.isnan(df["spatial_type"])]
    else:
        return df
