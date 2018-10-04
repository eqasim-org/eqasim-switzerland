import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree
import shapely.geometry as geo

def configure(context, require):
    require.stage("data.spatial.zones")
    require.stage("data.spatial.municipalities")
    require.stage("data.spatial.quarters")

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_quarters = context.stage("data.spatial.quarters")

    df_municipalities = pd.merge(
        df_municipalities, df_zones[df_zones["zone_level"] == "municipality"],
        right_on = "zone_level_id", left_on = "municipality_id"
    )[["zone_id", "zone_level_id", "zone_level", "geometry"]]

    df_quarters = pd.merge(
        df_quarters, df_zones[df_zones["zone_level"] == "quarter"],
        right_on = "zone_level_id", left_on = "quarter_id"
    )[["zone_id", "zone_level_id", "zone_level", "geometry"]]

    df = pd.concat([df_municipalities, df_quarters])

    #df["zone_level"] = df["zone_level"].astype("str")
    #df.to_file("/home/sebastian/zones.shp")

    return df

def sample_coordinates(row, count):
    samples = []
    bounds = row["geometry"].bounds

    while len(samples) < count:
        x = bounds[0] + np.random.random(size = (1000,)) * (bounds[2] - bounds[0])
        y = bounds[1] + np.random.random(size = (1000,)) * (bounds[3] - bounds[1])
        points = map(geo.Point, zip(x, y))
        points = [point for point in points if row["geometry"].contains(point)]
        samples += points

    return np.array(list(map(lambda p: (p.x, p.y), samples[:count])))
