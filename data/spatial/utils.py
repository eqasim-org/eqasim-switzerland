import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as geo
from sklearn.neighbors import KDTree
import logging

logger = logging.getLogger("synpp")

def sample_coordinates(row, count, random_seed=0):
    samples = []
    bounds = row["geometry"].bounds
    
    # Set up RNG
    rng = np.random.RandomState(random_seed)

    while len(samples) < count:
        x = bounds[0] + rng.random_sample(size=(1000,)) * (bounds[2] - bounds[0])
        y = bounds[1] + rng.random_sample(size=(1000,)) * (bounds[3] - bounds[1])
        points = map(geo.Point, zip(x, y))
        points = [point for point in points if row["geometry"].contains(point)]
        samples += points

    return np.array(list(map(lambda p: (p.x, p.y), samples[:count])))

def to_gpd(context, df, x="x", y="y", crs="epsg:2056", coord_type="", chunk_size=10000):
    result = []
    chunk_count = max(1, int(len(df) / chunk_size))

    # pandas-native chunking instead of np.array_split
    indices = np.array_split(np.arange(len(df)), chunk_count)

    for idx in context.progress(indices,
                                 total=chunk_count,
                                 label="Converting %s coordinates" % coord_type):
        chunk = df.iloc[idx]
        result.append(
            gpd.GeoDataFrame(
                chunk,
                geometry=gpd.points_from_xy(chunk[x], chunk[y], crs=crs)
            )
        )

    df = gpd.GeoDataFrame(
        pd.concat(result).reset_index(),
        crs=result[0].crs
    )
    del result

    if not crs == "epsg:2056":
        df = df.to_crs("epsg:2056")
        df.crs = "epsg:2056"

    return df


def keep_one_zone_per_point(df_points, point_id_field, zone_id_field, zone_type="", point_type=""):
    """
    Reduces the result of the spatial join to one zone per point.

    The join returns one row per (zone, point) match, so overlapping zones
    silently multiply the rows and, through the merge at the end of impute(),
    the caller's own data frame. This happens as soon as an external-population
    region is configured: data.spatial.cantons and data.spatial.municipalities
    lay that region over the real zones as an extra one, so every point inside
    it matches twice.

    The real zone wins. The external region is the only zone carrying a
    negative id (see data.external_population.constants), which is what
    identifies it here.
    """

    duplicates = df_points[point_id_field].duplicated(keep=False)

    if not duplicates.any():
        return df_points

    logger.info("  %d %s points fall into several %s zones; keeping the real zone for each."
          % (df_points.loc[duplicates, point_id_field].nunique(), point_type, zone_type))

    # Sorting is stable, so points matching several real zones (overlapping
    # boundaries) keep the zone the join reported first, as before.
    zone_ids     = pd.to_numeric(df_points[zone_id_field], errors="coerce")
    is_last_resort = zone_ids.isna() | (zone_ids < 0)

    df_points = df_points.assign(_is_last_resort=is_last_resort).sort_values("_is_last_resort", kind="stable")
    df_points = df_points.drop_duplicates(point_id_field, keep="first").drop(columns=["_is_last_resort"])

    return df_points


def impute(context, df_points, df_zones, point_id_field, zone_id_field, fix_by_distance=True, chunk_size=10000,
           zone_type="", point_type=""):
    assert (type(df_points) == gpd.GeoDataFrame)
    assert (type(df_zones) == gpd.GeoDataFrame)

    assert (point_id_field in df_points.columns)
    assert (zone_id_field in df_zones.columns)
    assert (not zone_id_field in df_points.columns)

    df_original = df_points
    # GeoPandas 1.x preserves a named right index in the spatial-join output.
    # These indices are not part of the result contract, so normalize them
    # before joining instead of relying on the historical ``index_right`` name.
    df_points = df_points[[point_id_field, "geometry"]].reset_index(drop=True)
    df_zones = df_zones[[zone_id_field, "geometry"]].reset_index(drop=True)

    logger.info("Imputing %d %s zones onto %d %s points by spatial join..." 
          % (len(df_zones), zone_type, len(df_points), point_type))
    
    result = []
    chunk_count = max(1, int(len(df_points) / chunk_size))
    indices = np.array_split(np.arange(len(df_points)), chunk_count)

    for idx in context.progress(indices,
                                 total=chunk_count,
                                 label="Imputing %s zones onto %s points..." % (zone_type, point_type)):
        chunk = df_points.iloc[idx]
        result.append(gpd.sjoin(df_zones, chunk, predicate="contains", how="right"))
        
    df_points = pd.concat(result).reset_index()

    if "left_index" in df_points: del df_points["left_index"]
    if "right_index" in df_points: del df_points["right_index"]

    df_points = keep_one_zone_per_point(df_points, point_id_field, zone_id_field, zone_type, point_type)

    invalid_mask = pd.isnull(df_points[zone_id_field])

    if fix_by_distance and np.any(invalid_mask):
        logger.info("  Fixing %d points by centroid distance join..." % np.count_nonzero(invalid_mask))
        coordinates = np.vstack([df_zones["geometry"].centroid.x, df_zones["geometry"].centroid.y]).T
                
        kd_tree = KDTree(coordinates)

        df_missing = df_points[invalid_mask]
        coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
        indices = kd_tree.query(coordinates, return_distance=False).flatten()

        df_points.loc[invalid_mask, zone_id_field] = df_zones.iloc[indices][zone_id_field].values

    return pd.merge(df_original, df_points[[point_id_field, zone_id_field]], on=point_id_field, how="left")



def convert_crs(x, y, original_crs="EPSG:2056", target_crs="EPSG:4326"):
    """Convert coordinates from EPSG:2056 to EPSG:4326"""
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, crs=original_crs))
    gdf = gdf.to_crs(target_crs)
    x4326 = gdf.geometry.x.values
    y4326 = gdf.geometry.y.values
    return x4326, y4326
