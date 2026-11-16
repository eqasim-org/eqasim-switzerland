"""Shared residence-region lookup and optional matching dependency."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd


REGION = "opening_hours_region"
STAGE = "data.spatial.opening_hours_regions"
logger = logging.getLogger("synpp")


def configure(context):
    context.config("consider_opening_hours", default=True)
    if context.config("consider_opening_hours"):
        context.stage(STAGE)


def add_regions(context, source, population):
    if not context.config("consider_opening_hours"):
        return source, population
    regions = context.stage(STAGE)
    return assign_regions(source, regions), assign_regions(population, regions)


def assign_regions(persons, regions):
    """Join residence points once, preserving original rows, index and IDs.

    Locations outside the region polygons use their nearest region; ties on
    shared boundaries or equal nearest distances prefer the lowest region ID.
    Missing/non-finite coordinates fail rather than creating a permissive mask.
    """
    coordinates = persons[["home_x", "home_y"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Opening-hours matching requires finite residence coordinates")
    if regions.empty or regions[REGION].isna().any() or regions[REGION].duplicated().any():
        raise ValueError("Opening-hours regions must have unique, non-null IDs")
    polygons = regions[[REGION, "geometry"]].to_crs(2056)
    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]), crs=2056)
    # Query the point tree with the few detailed polygons. GEOS prepares these
    # polygons once instead of repeatedly scanning their vertices per home.
    # Preserve intersects (including boundaries) and the lowest-ID tie rule.
    matches = gpd.sjoin(polygons, points, how="inner", predicate="intersects")
    joined = pd.DataFrame({REGION: matches[REGION].to_numpy()}, index=matches["index_right"].to_numpy())
    joined = joined.sort_values(REGION).loc[lambda df: ~df.index.duplicated()]
    joined = joined.reindex(points.index)
    missing = joined[REGION].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(points.loc[joined.index[missing]], polygons, how="left")
        nearest = nearest.sort_values(REGION).loc[lambda df: ~df.index.duplicated()]
        joined.loc[nearest.index, REGION] = nearest[REGION]
        logger.info("Assigned %d residences outside region polygons to their nearest opening-hours region", missing.sum())
    result = persons.copy()
    result[REGION] = joined[REGION].reindex(points.index).to_numpy(dtype=np.int64)
    return result
