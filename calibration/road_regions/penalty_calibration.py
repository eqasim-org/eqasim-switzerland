from io import StringIO
import os
import json
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import GeometryCollection, MultiPolygon, mapping
from shapely.ops import transform, polygonize, unary_union
from pyproj import Transformer
from data.osm.clean import read_outside_region
from shapely.validation import make_valid
 
 
def configure(context):
    # we need to include the network of this french region, 
    # I don't know if this is the right config param to use, to check later!
    context.config("cross_border_exclude_shapefiles", default=None)
    context.config("include_external_population", default = False)
    context.stage("data.external_population.constants")
 
 
def execute(context):
    df = pd.read_csv(StringIO(REGIONS_CSV), skipinitialspace=True)
    
    # whether to transform the coordinates from WGS84 to LV95 (EPSG:2056) or not
    df["transform_coordinates"] = True
 
    # include the french part as a separet calibration region
    out_region_file = context.config("cross_border_exclude_shapefiles")
    include_external_population = context.config("include_external_population")
    if out_region_file is not None and include_external_population:
        out_region = read_outside_region(out_region_file).to_crs("EPSG:4326")
        valid_geometries = [make_valid(geom) if not geom.is_valid else geom for geom in out_region.geometry]
        valid_gdf = gpd.GeoDataFrame(geometry=valid_geometries, crs="EPSG:4326")
        out_region_geometry = unary_union(valid_gdf.geometry)
 
        # Existing regions take precedence over the outside region (all in WGS84).
        existing_regions = unary_union([make_valid(wkt.loads(value)) for value in df["WKT"]])
        out_region_geometry = out_region_geometry.difference(existing_regions)
        if not out_region_geometry.is_empty:
            out_region_wkt = multipolygone_to_polygone_wkt(out_region_geometry)
            df = pd.concat([df,
                            pd.DataFrame({"WKT":[out_region_wkt], 
                                          "nom":["France"], 
                                          "description":["French part of the cross-border region"], 
                                          "transform_coordinates":[True]}),
                            ],
                            ignore_index=True)
 
    paths = [] 
    for i, row in df.iterrows():
        name = row["nom"]
        description = row.get("description", "")
        raw_wkt = row["WKT"]
        transform_coordinates = row.get("transform_coordinates", True)
        
        geometry = reproject_to_lv95(raw_wkt, transform_coordinates)
   
        region = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "name": name,
                        "description": description if pd.notna(description) else ""
                    },
                    "geometry": geometry
                }
            ]
        }
        
        file_name = f"region_{name}.json".lower().replace(" ", "_").replace("+", "_")
        path_i = os.path.join(context.path(), file_name)
        with open(path_i, "w", encoding="utf-8") as f:
            json.dump(region, f, indent=2, ensure_ascii=False)
 
        paths.append(path_i)
 
    return ";".join(paths)
 
 
 
# ---------------------------------------------------------------------------
# Data — exported from Google My Maps (WKT, WGS84, lon lat order)
#
# HOW TO REFRESH:
#   Google My Maps → ⋮ → Export to csv → copy and paste here (including header)
# ---------------------------------------------------------------------------

REGIONS_CSV = """
WKT,nom,description
"POLYGON ((6.2433552 46.3033983, 6.2473252 46.3596963, 6.3758652 46.4142593, 6.6248428 46.4669706, 6.7119539 46.4736943, 6.7359422 46.5144334, 6.6899369 46.540983, 6.660844 46.5654114, 6.6092026 46.5809203, 6.5589249 46.5756388, 6.5201981 46.5446669, 6.3872636 46.5004446, 6.2737611 46.4481443, 6.1923935 46.3962671, 6.1666381 46.3668324, 6.1018187 46.285276, 6.12489 46.2526193, 6.1023681 46.2385634, 6.0880858 46.2465415, 6.0677611 46.2427426, 6.0628173 46.2476812, 6.0463378 46.232864, 6.0326049 46.2389433, 5.9732787 46.2134817, 5.9666869 46.1959938, 5.9864087 46.1815968, 5.9605909 46.1462144, 5.9671827 46.1382217, 5.956059 46.133654, 5.9566083 46.1294665, 5.9783063 46.1334636, 5.9821515 46.14336, 6.0354352 46.1355572, 6.0538373 46.151352, 6.1348045 46.1403315, 6.1880882 46.1669684, 6.1856162 46.1791409, 6.2325828 46.2055689, 6.2504963 46.2069231, 6.2927937 46.2228874, 6.3087238 46.2422664, 6.3065266 46.2567012, 6.2944416 46.2642969, 6.2801594 46.2536626, 6.2642292 46.2498641, 6.2493977 46.2654362, 6.237862 46.2768276, 6.2439045 46.2851798, 6.2521442 46.2889758, 6.2521442 46.295808, 6.2433552 46.3033983))",GenevaLausanneRegion,
"POLYGON ((6.9058383 46.4469953, 6.6585 46.2726269, 6.6310342 45.8495359, 7.1693643 45.692439, 7.6692422 45.7116208, 8.1910928 45.8801373, 8.6799844 45.7346303, 9.0645059 45.7921127, 9.4050821 46.0290783, 9.7676309 46.0138207, 10.3663858 46.1814224, 10.8223184 46.5453311, 10.8442911 46.9218258, 10.3993448 47.0155388, 9.8115762 47.3515468, 9.536918 47.3515468, 9.2128213 47.2882416, 8.9271768 47.1651383, 8.6635049 47.0604628, 8.3738864 46.9143052, 8.060776 46.8354483, 7.8218234 46.7828126, 7.749039 46.7273014, 7.6240695 46.6990535, 7.5210727 46.742361, 7.4112094 46.7254187, 7.2931064 46.6896342, 7.1461642 46.6349701, 6.9772494 46.5717547, 6.9058383 46.4469953))",Alpes,
"""


########### FUNCTIONS ############
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True) 
 
def reproject_to_lv95(wkt_wgs84: str, transform_coordinates: bool = True) -> dict:
    """
    Parse a WKT polygon in WGS84 and return a GeoJSON-like geometry dict
    in EPSG:2056 (LV95).
 
    Raises ValueError if the WKT is invalid or the reprojected geometry
    is not valid (e.g. self-intersecting).
    """
    geom_wgs84 = wkt.loads(wkt_wgs84)
    if not geom_wgs84.is_valid:
        raise ValueError(f"Input geometry is not valid: {wkt_wgs84[:80]}…")
 
    if transform_coordinates:
        geom_lv95 = transform(_TRANSFORMER.transform, geom_wgs84)
    else:
        geom_lv95 = geom_wgs84
 
    if not geom_lv95.is_valid:
        raise ValueError(
            f"Reprojected geometry is not valid for: {wkt_wgs84[:80]}…"
        )
 
    # Return as a GeoJSON geometry object (coordinates are [easting, northing])
    return {
        **mapping(geom_lv95),
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::2056"},
        },        
    }
 
 
def _extract_polygons(geometry):
    """
    Recursively extract all Polygon parts from any Shapely geometry
    (Polygon, MultiPolygon, GeometryCollection, or nested thereof).
 
    Returns:
        list[Polygon]: all polygon sub-geometries found
    """
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        polygons = []
        for geom in geometry.geoms:
            polygons.extend(_extract_polygons(geom))
        return polygons
    else:
        # Point, LineString, etc. — not useful for our purpose
        return []
 
 
def multipolygone_to_polygone_wkt(geometry):
    """
    Convert any Shapely geometry to a single Polygon WKT by extracting
    all polygon parts and returning the largest one.
 
    Handles Polygon, MultiPolygon, and GeometryCollection (including
    nested collections that unary_union can produce on complex inputs).
 
    Args:
        geometry: A Shapely geometry of any type
 
    Returns:
        str: WKT representation of the largest Polygon found
 
    Raises:
        ValueError: if no polygon parts can be extracted from the geometry
    """
    polygons = _extract_polygons(geometry)
 
    if not polygons:
        raise ValueError(
            f"No polygon parts found in geometry of type {geometry.geom_type}"
        )
 
    # Return the largest polygon by area
    largest = max(polygons, key=lambda p: p.area)
    return largest.wkt
 