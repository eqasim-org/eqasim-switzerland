import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
from tqdm import tqdm
import multiprocessing as mp
from sklearn.neighbors import KDTree

def configure(context, require):
    require.stage("data.misc.spatial_structure")
    require.stage("data.microcensus.microcensus")

def execute(context):
    df_structure = context.stage("data.misc.spatial_structure")

    df = context.stage("data.microcensus.microcensus")[0]

    df["geometry"] = [
        geo.Point(*coord) for coord in tqdm(
            zip(df["home_x"], df["home_y"]), total = len(df),
            desc = "Converting coordinates"
        )]

    df = gpd.GeoDataFrame(df[["person_id", "geometry"]])
    df.crs = {"init":"EPSG:2056"}

    size_before_join = len(df)

    # Now do the spatial join
    result = []
    for chunk in tqdm(np.array_split(df, 50), total = 50, desc = "Performing spatial join"):
        result.append(gpd.sjoin(chunk, df_structure, op = "within")[[
            "person_id", "geometry", "spatial_type", "zone"
        ]])
    df_matched = pd.concat(result)

    # The missing ones are found by nearest neighbor
    missing_ids = set(np.unique(df["person_id"])) - set(np.unique(df_matched["person_id"]))
    df_missing = gpd.GeoDataFrame(df[df["person_id"].isin(missing_ids)])

    print("Matching %d missing persons by distance" % len(df_missing))

    coordinates = np.vstack([df_structure["geometry"].centroid.x, df_structure["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance = False).flatten()

    df_missing.loc[:, "spatial_type"] = df_structure.iloc[indices]["spatial_type"].values
    df_missing.loc[:, "zone"] = df_structure.iloc[indices]["zone"].values

    df = pd.concat([df_matched, df_missing])
    assert(size_before_join == len(df))

    df.crs = {"init" : "EPSG:2056"}
    return df
