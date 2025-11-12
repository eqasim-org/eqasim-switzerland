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
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")
    
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
    
    # transform it into geopandas dataframe
    df_spatial = gpd.GeoDataFrame(df_spatial, crs = "EPSG:2056", geometry = "preceding_geometry")

    # get coordinates
    df_spatial['origin_x'] = df_spatial.preceding_geometry.x
    df_spatial['origin_y'] = df_spatial.preceding_geometry.y
    df_spatial['destination_x'] = df_spatial.following_geometry.x
    df_spatial['destination_y'] = df_spatial.following_geometry.y  

    # get home locations
    homes = df_spatial.loc[df_spatial["preceding_purpose"]=="home", ["person_id","origin_x","origin_y"]].rename(
            columns={"origin_x":"home_x", "origin_y":"home_y"}).drop_duplicates("person_id").reset_index(drop=True)    
    df_spatial = df_spatial.merge(homes, on="person_id", how="left")
    assert not df_spatial[["home_x","home_y"]].isna().any().any(), "Some trips have no home location!"

    # determine destination type    
    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities, _ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type, on="municipality_id")
    df_municipalities = gpd.GeoDataFrame(df_municipalities[["municipality_type", "geometry"]], crs="EPSG:2056")

    # Spatial join for origin municipality
    df_origin = gpd.GeoDataFrame(df_spatial, geometry="preceding_geometry", crs="EPSG:2056")
    df_origin = df_origin.sjoin_nearest(df_municipalities, how="left")
    df_spatial["origin_municipality"] = df_origin["municipality_type"]

    # Spatial join for destination municipality
    df_dest = gpd.GeoDataFrame(df_spatial, geometry="following_geometry", crs="EPSG:2056")
    df_dest = df_dest.sjoin_nearest(df_municipalities, how="left")
    df_spatial["destination_municipality"] = df_dest["municipality_type"]

    # home municipality
    homes = df_spatial.loc[df_spatial["preceding_purpose"]=="home", ["person_id","origin_municipality"]].rename(
            columns={"origin_municipality":"home_municipality"}).drop_duplicates("person_id").reset_index(drop=True)    
    df_spatial = df_spatial.merge(homes, on="person_id", how="left")
    assert not df_spatial[["home_municipality","origin_municipality","destination_municipality"]].isna().any().any(), "Some trips have no municipality type!"
    
    # compute crowfly distance
    df_spatial["crowfly_distance"] = df_spatial.following_geometry.distance(df_spatial.preceding_geometry)  

    # finalize dataframe
    df_spatial["following_purpose"] = df_spatial["following_purpose"].astype(str)
    df_spatial["preceding_purpose"] = df_spatial["preceding_purpose"].astype(str)
    df_spatial["mode"] = df_spatial["mode"].astype(str)
    df_spatial["trip_id"] = df_spatial["person_id"].astype(str) + "_" + df_spatial["trip_index"].astype(str)
    
    return df_spatial[[
        "person_id", "trip_index", "trip_id", "preceding_purpose", "following_purpose",
        "departure_time", "mode", "crowfly_distance",
        "origin_x", "origin_y", "destination_x", "destination_y", "home_x", "home_y",
        "origin_municipality", "destination_municipality", "home_municipality"
    ]]