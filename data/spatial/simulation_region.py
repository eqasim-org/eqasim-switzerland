from shapely.ops import unary_union
from data.spatial.utils import read_outside_region

def configure(context):
    context.stage("data.spatial.swiss_border")

    # we include the network of this region, i don't know if this is the right config param to use, to check later!
    context.config("cross_border_exclude_shapefiles", default=None)
    context.config("include_external_population", default = True)    

def execute(context):
    """
    This method retrieves the geometry of the area that is simulated
    """
    # Bounding Area
    border = context.stage("data.spatial.swiss_border")
    border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
    
    # Collect all geometries to combine
    geometries = [unary_union(border.geometry)]
    
    # outside CH region
    out_region_file = context.config("cross_border_exclude_shapefiles")
    include_external_population = context.config("include_external_population")
    if out_region_file is not None and include_external_population:
        out_region = read_outside_region(out_region_file)
        geometries.append(unary_union(out_region.geometry))
    
    # Combine border + outside regions into one geometry
    combined = unary_union(geometries)
    
    return combined