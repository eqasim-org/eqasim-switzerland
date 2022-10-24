import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.geometry as geo
from sklearn.neighbors import KDTree


def sample_coordinates(row, count):
    samples = []
    bounds = row["geometry"].bounds

    while len(samples) < count:
        x = bounds[0] + np.random.random(size=(1000,)) * (bounds[2] - bounds[0])
        y = bounds[1] + np.random.random(size=(1000,)) * (bounds[3] - bounds[1])
        points = map(geo.Point, zip(x, y))
        points = [point for point in points if row["geometry"].contains(point)]
        samples += points

    return np.array(list(map(lambda p: (p.x, p.y), samples[:count])))


def to_gpd(context, df, x="x", y="y", crs="epsg:2056", coord_type=""):
    df["geometry"] = [
        geo.Point(*coord) for coord in context.progress(
            zip(df[x], df[y]), total=len(df),
            label="Converting %s coordinates" % coord_type
        )]
    df = gpd.GeoDataFrame(df)
    df.crs = crs

    if not crs == "epsg:2056":
        df = df.to_crs("epsg:2056")
        df.crs = "epsg:2056"

    return df


def impute(context, df_points, df_zones, point_id_field, zone_id_field, fix_by_distance=True, chunk_size=10000,
           zone_type="", point_type=""):
    assert (type(df_points) == gpd.GeoDataFrame)
    assert (type(df_zones) == gpd.GeoDataFrame)

    assert (point_id_field in df_points.columns)
    assert (zone_id_field in df_zones.columns)
    assert (not zone_id_field in df_points.columns)

    df_original = df_points
    df_points = df_points[[point_id_field, "geometry"]]
    df_zones = df_zones[[zone_id_field, "geometry"]]

    print("Imputing %d %s zones onto %d %s points by spatial join..."
          % (len(df_zones), zone_type, len(df_points), point_type))

    result = []
    chunk_count = max(1, int(len(df_points) / chunk_size))
    for chunk in context.progress(np.array_split(df_points, chunk_count), total=chunk_count):
        result.append(gpd.sjoin(df_zones, chunk, op="contains", how="right"))
    df_points = pd.concat(result).reset_index()

    if "left_index" in df_points: del df_points["left_index"]
    if "right_index" in df_points: del df_points["right_index"]

    invalid_mask = pd.isnull(df_points[zone_id_field])

    if fix_by_distance and np.any(invalid_mask):
        print("  Fixing %d points by centroid distance join..." % np.count_nonzero(invalid_mask))
        coordinates = np.vstack([df_zones["geometry"].centroid.x, df_zones["geometry"].centroid.y]).T
        kd_tree = KDTree(coordinates)

        df_missing = df_points[invalid_mask]
        coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
        indices = kd_tree.query(coordinates, return_distance=False).flatten()

        df_points.loc[invalid_mask, zone_id_field] = df_zones.iloc[indices][zone_id_field].values

    return pd.merge(df_original, df_points[[point_id_field, zone_id_field]], on=point_id_field, how="left")
