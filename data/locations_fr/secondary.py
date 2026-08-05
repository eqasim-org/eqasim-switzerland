from shapely import set_precision
from shapely.validation import make_valid
from shapely.ops import unary_union
import geopandas as gpd


def configure(context):
    context.stage("data.locations_fr.bpe.cleaned")
    context.stage("data.spatial.swiss_border")
    context.config("outbound_flows_perimeter", 20000) # Can be a number (= buffer diameter around the Swiss border in meters) or a list of .gpkg or shapefiles


def execute(context):

    df_locations = context.stage("data.locations_fr.bpe.cleaned")[[
        "enterprise_id", "x", "y", "activity_type", "commune_id", "geometry", "weight"
    ]].copy()

    # enterprise_id is already a canonical id, disjoint from STATENT's by
    # construction, so no offset is needed here.
    df_locations = df_locations.rename(columns = {"enterprise_id": "destination_id"})

    # Attach attributes for activity types
    df_locations["offers_leisure"] = df_locations["activity_type"] == "leisure"
    df_locations["offers_shop"]    = df_locations["activity_type"] == "shop"
    df_locations["offers_other"]   = ~(df_locations["offers_leisure"] | df_locations["offers_shop"])

    del df_locations["activity_type"]

    df_locations["ovgk"]     = "C"
    df_locations["geometry"] = df_locations["geometry"].apply(lambda geom: set_precision(geom, grid_size = 1.0))

    df_locations["destination_x"] = df_locations["geometry"].x.astype(int)
    df_locations["destination_y"] = df_locations["geometry"].y.astype(int)

    df_locations["offers_work"]                = False
    df_locations["offers_education"]           = False
    df_locations["offers_work_secondary"]      = False
    df_locations["offers_education_secondary"] = False 
    df_locations["offers_home_secondary"]      = False

    # At this point we have no ideas abot the number of employees in each location, but the weight here is the capacity which is the maximum number of people that can enter the facility. We can use it as a proxy for the number of employees, but we need to scale it down to avoid having locations with an unrealistically high number of employees. 
    # The factor of 0.5 is arbitrary and can be adjusted based on the desired distribution of employees across locations.
    # TODO: find a better way to estimate the number of employees in each location, for example by using the distribution of employees in the statent dataset and applying it to the locations in the bpe dataset based on their activity type and weight.
    df_locations["number_employees"] = df_locations["weight"].fillna(1).astype(int)*0.5
    del df_locations["weight"]
    
    perimeter    = context.config("outbound_flows_perimeter")
    swiss_border = context.stage("data.spatial.swiss_border").iloc[0]

    if isinstance(perimeter, (int, float)):
        target_region = swiss_border.buffer(perimeter)

    elif isinstance(perimeter, str):
        if perimeter.endswith(".shp") or perimeter.endswith(".gpkg"):
            target_region = gpd.read_file(perimeter).to_crs("epsg:2056").geometry.apply(make_valid).unary_union
        else:
            raise ValueError("Unsupported file format: %s" % perimeter)

    elif isinstance(perimeter, list):
        geometries = []
        for path in perimeter:
            if path.endswith(".shp") or path.endswith(".gpkg"):
                geometries.append(gpd.read_file(path).to_crs("epsg:2056").geometry.apply(make_valid).unary_union)
            else:
                raise ValueError("Unsupported file format: %s" % path)
        target_region = unary_union(geometries)

    else:
        raise ValueError(
            "outbound_flows_perimeter must be a number, a file path, or a list of file paths. Got: %s" % type(perimeter)
        )

    df_locations = df_locations[df_locations.within(target_region)]


    df_locations = df_locations[["destination_id", "number_employees", "destination_x", "destination_y", 
                                 "offers_work", "offers_education",
                                 "offers_leisure", "offers_shop", "offers_other",
                                 "offers_work_secondary", "offers_education_secondary", "offers_home_secondary",
                                 "ovgk", "geometry"]]

    return df_locations