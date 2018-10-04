import shapely.geometry as geo
import numpy as np
from tqdm import tqdm
import geopandas as gpd

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

def to_gpd(df, x = "x", y = "y", crs = {"init" : "EPSG:2056"}):
    df["geometry"] = [
        geo.Point(*coord) for coord in tqdm(
            zip(df[x], df[y]), total = len(df),
            desc = "Converting coordinates"
        )]
    df = gpd.GeoDataFrame(df)
    df.crs = crs

    if not crs == {"init" : "EPSG:2056"}:
        df = df.to_crs({"init" : "EPSG:2056"})

    return df
