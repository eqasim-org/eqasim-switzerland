import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def configure(context):
    context.config("number_points_per_uspat_zone")
    context.config("data_path")

    context.stage("data.statent.statent")
    context.stage("pt_preparation.pt_pricing.uspat_zones")


def sample_points_in_polygon(polygon, n):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < n:
        random_points = [
            Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            for _ in range(n * 2)  
        ]
        inside = [p for p in random_points if polygon.contains(p)]
        points.extend(inside)
    return points[:n]


def weighted_sample(group, n):
    # If the zone has fewer than n points, return them all
    n = min(n, len(group))
    return group.sample(n=n, weights=group["number_employees"], replace=False)


def execute(context):
    upsat_zones = context.stage("pt_preparation.pt_pricing.uspat_zones")

    # statent points
    locations  = context.stage("data.statent.statent").copy()[["x", "y", "number_employees"]]
    locations = gpd.GeoDataFrame(
        locations,
        geometry=gpd.points_from_xy(locations.x, locations.y),
        crs="EPSG:2056"
    )

    # UPSAT zones
    data_path  = context.config("data_path")
    lakes_path = f"{data_path}/spatial/lakes/g1s18.shp"
    lakes      = gpd.read_file(lakes_path)

    if upsat_zones.crs != lakes.crs:
        lakes = lakes.to_crs(upsat_zones.crs)

    upsat_zones = gpd.overlay(upsat_zones, lakes, how="difference")
    upsat_zones = upsat_zones.to_crs("EPSG:2056")

    # Join statent locations to upsat
    points_in_zones = gpd.sjoin(locations, upsat_zones, how="inner", predicate="within")

    N = context.config("number_points_per_uspat_zone")

    sampled = points_in_zones.groupby("zone_id", group_keys=False).apply(weighted_sample, n=N)

    print(f"Sampled {len(sampled)} in the study area.")
    print(f"This will lead to {len(sampled)**2} requests.")

    return sampled