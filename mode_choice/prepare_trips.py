"""
Prepares trip data for discrete mode choice in eqasim.

This module processes synthetic population trips by:
- Linking trips with their preceding and following activity locations
- Creating LineString geometries representing trip origins and destinations
- Computing crowfly distances between activity locations
"""

import pandas as pd
import geopandas as gpd
from shapely import linestrings
import numpy as np

def configure(context):
    context.stage("synthesis.population.trips")
    context.stage("synthesis.population.spatial.locations")

def execute(context):
    # read trips
    df_trips = context.stage("synthesis.population.trips")

    df_trips["preceding_activity_index"] = df_trips["trip_index"]
    df_trips["following_activity_index"] = df_trips["trip_index"] + 1

    # read spatial data set
    df_locations = context.stage("synthesis.population.spatial.locations")[[
        "person_id", "activity_index", "geometry"
    ]]
    
    # merge to get locations
    df_spatial = pd.merge(df_trips, df_locations[[
        "person_id", "activity_index", "geometry"
    ]].rename(columns = {
        "activity_index": "preceding_activity_index",
        "geometry": "preceding_geometry"
    }), how = "left", on = ["person_id", "preceding_activity_index"])

    df_spatial = pd.merge(df_spatial, df_locations[[
        "person_id", "activity_index", "geometry"
    ]].rename(columns = {
        "activity_index": "following_activity_index",
        "geometry": "following_geometry"
    }), how = "left", on = ["person_id", "following_activity_index"])

    # finalize geodataframe
    df_spatial = gpd.GeoDataFrame(df_spatial, crs = "EPSG:2056", geometry = "following_geometry")
    df_spatial["following_purpose"] = df_spatial["following_purpose"].astype(str)
    df_spatial["preceding_purpose"] = df_spatial["preceding_purpose"].astype(str)
    df_spatial["mode"] = df_spatial["mode"].astype(str)
    df_spatial["crowfly_distance"] = df_spatial.following_geometry.distance(df_spatial.preceding_geometry)
    
    df_spatial['origin_x'] = df_spatial.preceding_geometry.x
    df_spatial['origin_y'] = df_spatial.preceding_geometry.y
    df_spatial['destination_x'] = df_spatial.following_geometry.x
    df_spatial['destination_y'] = df_spatial.following_geometry.y
    return df_spatial