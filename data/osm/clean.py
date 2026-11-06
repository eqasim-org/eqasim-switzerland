import data.osm.conversion_functions as cf
import data.osm.merging_functions as mf
from data.osm.update_osm_latest import update_osm_latest
import logging
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.ops import unary_union

logger = logging.getLogger("synpp")

def configure(context):    
    context.stage("data.spatial.simulation_region")

    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 50000)

    context.config("correct_osm_speeds", default = True)
    if context.config("correct_osm_speeds"):
        context.stage("data.osm.speed_corrections")

    context.config("use_latest_osm_for_correction", default = True)
    context.config("latest_osm_for_correction", default=None)
    if context.config("use_latest_osm_for_correction") and context.config("latest_osm_for_correction") == "":
        raise ValueError("If use_latest_osm_for_correction is True, latest_osm_for_correction must be provided.")


def execute(context):
    # if the path is not a list, treat it as a single file, else treat it as a list of files, merge them, and keep only
    # the network within 'border_offset' distance to the border
    output_file = '%s/osm_network.osm.gz' % context.path()
    osm_file = context.config("osm_file")

    speed_corrections = None
    if context.config("correct_osm_speeds"):
        speed_corrections = context.stage("data.osm.speed_corrections")
        logger.info(f"Speed corrections will be applied to {len(speed_corrections)} ways in the OSM network.")

    if not isinstance(osm_file,list):
        osm_file = '%s/osm/%s' % (context.config("data_path"), osm_file)
        cleaned_file = cf.from_pbf_to_osm_gz(context, osm_file, output_file, speed_corrections)
    
    else:
        osm_files = ['%s/osm/%s' % (context.config("data_path"), f) for f in osm_file]

        # Bounding Area
        border = get_region(context)
        border = border.to_crs("EPSG:4326") # because osm in in wgs84       
        # Merge and cut to the area
        cleaned_file = mf.merge_files(context, osm_files, border, output_file, speed_corrections)

    if context.config("use_latest_osm_for_correction"):
        latest_name = context.config("latest_osm_for_correction")
        # Retain the correction stage's legacy fallback when the option is omitted.
        latest_file = Path(context.config("data_path")) / "osm" / (latest_name if latest_name is not None else "switzerland-261004.osm.pbf")
        cleaned_file = update_osm_latest(
            context, cleaned_file, latest_file,
            Path(context.path()) / "osm_network_updated.osm", speed_corrections,
        )
    return cleaned_file


################### helper functions ####################

def get_region(context):
    # read simulation region
    combined = context.stage("data.spatial.simulation_region")

    # Apply buffer
    buffer = context.config("border_offset")    
    if buffer <= 0:
        buffer = 100 # just 100 meter to unify geometries if they are not for numerical reasons

    combined = combined.buffer(buffer)
    combined = combined.simplify(min(buffer, 3000), preserve_topology=True)
    
    # Return as GeoDataFrame with single row
    result = gpd.GeoDataFrame(geometry=[combined], crs="EPSG:2056")
    return result
