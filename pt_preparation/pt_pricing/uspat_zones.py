import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def configure(context):
    context.config("uspat_path")
    context.config("uspat_cantons")

def execute(context):
    upsat_path = context.config("uspat_path")
    cantons    = context.config("uspat_cantons")

    # UPSAT zones
    upsat_zones = gpd.read_file(upsat_path)
    upsat_zones = upsat_zones[upsat_zones["KT_ID"].isin(cantons)]
    upsat_zones = upsat_zones[["U1_ID", "geometry"]].rename(columns={"U1_ID": "zone_id"})

    return upsat_zones