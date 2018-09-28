import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.neighbors import KDTree

def configure(context, require):
    require.config("raw_data_path")
    require.stage("data.od.municipality_shapes")

def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df_shapes, _ = context.stage("data.od.municipality_shapes")

    print("  Reading structure information ...")
    df_structure = pd.read_excel(
        "%s/spatial_structure_2018.xlsx" % raw_data_path,
        skiprows = (0, 1, 2, 4, 5, 6)
    )[["BFS Nr.", "Raum mit städtischem Charakter 2012"]]
    df_structure.columns = ["zone", "spatial_type_raw"]

    unique_shape_ids = set(np.unique(df_shapes["zone"]))
    print("  Found %d municipalities in the shape file" % len(unique_shape_ids))

    unique_structure_ids = set(np.unique(df_structure["zone"]))
    print("  Found %d municipalities in the structural information" % len(unique_structure_ids))

    unassigned_ids = unique_shape_ids - unique_structure_ids
    print("  Found %d municipalities for which the structural type is unknown" % len(unassigned_ids))

    # First, add in the existing information
    df = pd.merge(df_shapes, df_structure, on = "zone", how = "left")

    # Then, take the structure type from the closest known zone by centroid
    f_missing = df["zone"].isin(unassigned_ids)
    df_existing = df[~f_missing]
    df_missing = df[f_missing]

    coordinates = np.vstack([df_existing["geometry"].centroid.x, df_existing["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance = False).flatten()

    df.loc[f_missing, "spatial_type_raw"] = df_existing.iloc[indices]["spatial_type_raw"].values
    df.loc[:, "imputed_spatial_type"] = f_missing

    assert(len(df) == len(df.dropna()))
    assert(set(np.unique(df["zone"])) == set(np.unique(df_shapes["zone"])))

    # Redefine the types

    df.loc[df["spatial_type_raw"] == 1, "spatial_type"] = "urban"
    df.loc[df["spatial_type_raw"] == 2, "spatial_type"] = "urban"
    df.loc[df["spatial_type_raw"] == 3, "spatial_type"] = "suburban"
    df.loc[df["spatial_type_raw"] == 4, "spatial_type"] = "suburban"
    df.loc[df["spatial_type_raw"] == 5, "spatial_type"] = "rural"
    df.loc[df["spatial_type_raw"] == 6, "spatial_type"] = "rural"
    df.loc[df["spatial_type_raw"] == 0, "spatial_type"] = "rural"

    df["spatial_type"] = df["spatial_type"].astype("category")
    df = df[["zone", "spatial_type", "geometry"]]

    print("  Dissolving shapes...")
    df = df.dissolve(by = "spatial_type").reset_index()

    return df
