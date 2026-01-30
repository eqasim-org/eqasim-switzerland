import pandas as pd
import numpy as np
import geopandas as gpd

"""
Stage description
This stage reads the NPVM (after post-processing at SBB?) public transport subscription data 
at the NPVM zone level.

Data source
Joschka sent the data by mail on 25 November 2025
"""

def configure(context):
    context.config("data_path")
    context.stage("data.spatial.cantons")


def execute(context):
    data_path = context.config("data_path")

    npvm = pd.read_csv(f"{data_path}/pt_reference_data/106_release.mobi-zones.csv", 
                       sep = ";", nrows = 7979) # Excluding the rows below this one as the later zones are outside Switzerland

    npvm = npvm[["zone_id", "zone_index", "pop_ga", "pop_ht", "pop_va", "pop_va_ht"]]
    npvm = npvm.rename(columns = {"pop_ga": "N_ga_npvm", "pop_ht": "N_ht_npvm",
                                  "pop_va": "N_va_npvm", 
                                  "pop_va_ht": "N_va_and_ht_npvm"})

    zones_path = f"{data_path}/npvm/1_Verkehrszonen_Schweiz_NPVM_2023.gpkg" 
    zones_gdf  = gpd.read_file(zones_path, layer= None)
    zones_gdf = zones_gdf[["No", "geometry"]].copy().rename(columns = {"No": "zone_id"})
    zones_gdf["zone_id"] = zones_gdf["zone_id"].astype(int)

    cantons = context.stage("data.spatial.cantons")
    overlaps = gpd.overlay(
        zones_gdf,
        cantons[["canton_id", "canton_name", "geometry"]],
        how="intersection"
    )
    
    overlaps["overlap_area"] = overlaps.geometry.area
    dominant_zone_to_canton = (
        overlaps
        .sort_values(["zone_id", "overlap_area"], ascending=[True, False])
        .drop_duplicates(subset="zone_id")
        .drop(columns="overlap_area")
    )
    
    zones_with_canton = (
        zones_gdf
        .merge(
            dominant_zone_to_canton[["zone_id", "canton_id", "canton_name"]],
            on="zone_id",
            how="left"
        )
    )
    zones_with_canton = zones_with_canton[["zone_id", "canton_id", "canton_name"]]

    npvm = zones_with_canton.merge(npvm, on = "zone_id", how = "right")
    #npvm["canton_id"] = npvm["canton_id"].astype(int)

    return npvm