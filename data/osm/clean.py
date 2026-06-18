import data.osm.conversion_functions as cf
import data.osm.merging_functions as mf
import logging
import geopandas as gpd
import pandas as pd
from pathlib import Path
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("data.spatial.swiss_border")
    
    context.config("data_path")
    context.config("osm_file", "switzerland-latest.osm.gz")
    context.config("border_offset", 20000)
    # we include the network of this region, i don't know if this is the right config param to use, to check later!
    context.config("cross_border_exclude_shapefiles", default=None)


def execute(context):
    # if the path is not a list, treat it as a single file, else treat it as a list of files, merge them, and keep only
    # the network within 'border_offset' distance to the border
    output_file = '%s/osm_network.osm.gz' % context.path()
    osm_file = context.config("osm_file")

    if not isinstance(osm_file,list):
        osm_file = '%s/osm/%s' % (context.config("data_path"), osm_file)
        return cf.from_pbf_to_osm_gz(context, osm_file, output_file)
    
    else:
        osm_files = ['%s/osm/%s' % (context.config("data_path"), f) for f in osm_file]

        # Bounding Area
        border = get_region(context)
        border = border.to_crs("EPSG:4326") # because osm in in wgs84       
        # Merge and cut to the area
        return mf.merge_files(context, osm_files, border, output_file)








################### helper functions ####################

def get_region(context):
    # Bounding Area
    border = context.stage("data.spatial.swiss_border")
    border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
    
    # Collect all geometries to combine
    geometries = [unary_union(border.geometry)]
    
    # outside CH region
    out_region_file = context.config("cross_border_exclude_shapefiles")
    if out_region_file is not None:
        out_region = read_outside_region(out_region_file)
        geometries.append(unary_union(out_region.geometry))
    
    # Combine border + outside regions into one geometry
    combined = unary_union(geometries)
    
    # Apply buffer
    buffer = context.config("border_offset")    
    if buffer <= 0:
        buffer = 100 # just 100 meter to unify geometries if they are not for numerical reasons

    combined = combined.buffer(buffer)
    combined = combined.simplify(min(buffer, 3000), preserve_topology=True)
    
    # Return as GeoDataFrame with single row
    result = gpd.GeoDataFrame(geometry=[combined], crs="EPSG:2056")
    return result


def read_outside_region(out_region_file):    
    if isinstance(out_region_file, (str, Path)):
        out_region_file = [out_region_file]

    if not isinstance(out_region_file, (list, tuple)):
        raise TypeError("cross_border_exclude_shapefiles must be a path or a list of paths.")

    gdfs = []   
    for f in out_region_file:
        suffix = Path(f).suffix.lower()
        if suffix not in {".gpkg", ".shp"}:
            raise TypeError(f"{f} is not a .gpkg or .shp file.")

        gdf = gpd.read_file(f).to_crs(epsg=2056)
        gdfs.append(gdf)

    exclude_region = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:2056")
    return exclude_region
