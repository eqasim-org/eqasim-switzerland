"""Vectorized Swiss endpoint classification and upsampling boundary safeguards."""

import numpy as np
import polars as pl
from pyproj import Transformer
from scipy.spatial import cKDTree
import shapely

from sumo.cutter.tools import read_region
from sumo.tables import read_table


def endpoint_xy(frame, name, crs="EPSG:2056"):
    parts = frame[name].str.split_exact(",", 1)
    lon = parts.struct.field("field_0").cast(pl.Float64).to_numpy()
    lat = parts.struct.field("field_1").cast(pl.Float64).to_numpy()
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat, errcheck=True)
    if not np.isfinite([x, y]).all():
        raise ValueError("SUMO endpoint classification requires finite geographic coordinates")
    return np.column_stack([x, y])


def classify(frame, border):
    """Include polygon boundaries; deliberately discard outside-to-outside trips."""
    region = read_region(border)
    shapely.prepare(region)
    flags = []
    for name in ("origin", "destination"):
        xy = endpoint_xy(frame, name, border["crs"])
        flags.append(shapely.intersects_xy(region, xy[:, 0], xy[:, 1]))
    origin, destination = flags
    return origin & destination, origin ^ destination, ~(origin | destination)


def domestic_pools(pools, border):
    region = read_region(border)
    shapely.prepare(region)
    return {key: points[shapely.intersects_xy(region, points[:, 0], points[:, 1])]
            for key, points in pools.items()}


def fixed_endpoints(frame, originals, border):
    """Freeze newly introduced border anchors, leaving domestic activity ends movable."""
    original = originals.select(pl.col("id").alias("_base"),
        pl.col("origin").alias("_old_origin"), pl.col("destination").alias("_old_destination"))
    paired = frame.with_columns(pl.col("id").str.replace(r"__swiss_\d+$", "").alias("_base"))
    paired = paired.join(original, on="_base", how="left", validate="m:1", maintain_order="left")
    paired = paired.with_columns(pl.col("id").str.extract(r"__swiss_(\d+)$", 1)
        .cast(pl.Int64).fill_null(1).alias("_visit"))
    last_visit = paired.group_by("_base").agg(pl.col("_visit").max().alias("_last_visit"))
    paired = paired.join(last_visit, on="_base", how="left", validate="m:1", maintain_order="left")
    region = read_region(border)
    shapely.prepare(region)
    points = []
    for name in ("origin", "destination"):
        xy, previous = endpoint_xy(paired, name), endpoint_xy(paired, "_old_" + name)
        outside = ~shapely.intersects_xy(region, previous[:, 0], previous[:, 1])
        intermediate = ((paired["_visit"] > 1) if name == "origin" else
                        (paired["_visit"] < paired["_last_visit"])).to_numpy()
        points.append(xy[outside | intermediate])
    values = np.concatenate(points)
    return pl.DataFrame({"x": values[:, 0], "y": values[:, 1]}).unique()


def lock_border_anchors(anchors, path):
    fixed = read_table(path).select("x", "y").to_numpy() if path else np.empty((0, 2))
    locked = np.zeros(anchors.height, dtype=bool)
    if len(fixed):
        distances, _ = cKDTree(fixed).query(anchors.select("x", "y").to_numpy(), distance_upper_bound=1.)
        locked = np.isfinite(distances)
    return anchors.with_columns(pl.Series("_fixed", locked))
