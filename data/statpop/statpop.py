import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.distance import distance
from geopy.point import Point

import data.spatial.cantons
import data.spatial.municipality_types
import data.spatial.ovgk
import data.spatial.utils
import data.spatial.zones
import data.statpop.density
import data.statpop.head_of_household
import data.utils


def configure(context):
    context.stage("data.statpop.persons")
    context.stage("data.statpop.households")
    context.stage("data.statpop.link")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipality_types")
    context.stage("data.statpop.density")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.ovgk")
    context.stage("data.constants")


def add_wgs84_coordinates(gdf, drop_geometry=True):
    gdf = gdf.copy()
    
    # convert geometry to WGS84
    wgs84_geom = gdf.geometry.to_crs(epsg=4326)
    
    gdf['longitude'] = wgs84_geom.x
    gdf['latitude'] = wgs84_geom.y
    
    if drop_geometry:
        gdf = gdf.drop(columns=['geometry'])
    
    return gdf

def geo_indistinguishability_noise(epsilon, sensitivity=1.0):
    # Sample radius from exponential distribution
    r = np.random.exponential(scale=sensitivity / epsilon)

    # Sample angle uniformly
    theta = np.random.uniform(0, 2 * np.pi)

    # Convert polar to Cartesian
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy

def add_geo_noise(lat, lon, epsilon):
    dx, dy = geo_indistinguishability_noise(epsilon)
    
    # Convert dx/dy in kilometers
    noisy_point = distance(kilometers=dy).destination(Point(lat, lon), bearing=0)
    noisy_point = distance(kilometers=dx).destination(noisy_point, bearing=90)
    
    return noisy_point.latitude, noisy_point.longitude

def geo_indistinguishability_noise_meters(epsilon, sensitivity=1000.0):
    """
    Generate noise dx, dy in meters (Swiss coordinate units)
    sensitivity defaults to 1000 meters (1 km) here to scale properly.
    """
    # Sample radius from exponential distribution (in meters)
    r = np.random.exponential(scale=sensitivity / epsilon)

    # Sample angle uniformly
    theta = np.random.uniform(0, 2 * np.pi)

    # Convert polar to Cartesian offsets in meters
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    return dx, dy

def execute(context):
    df_persons    = context.stage("data.statpop.persons")
    df_households = context.stage("data.statpop.households")
    df_link       = context.stage("data.statpop.link")
    c             = context.stage("data.constants")

    # Filter non-main residence
    df_persons = df_persons[df_persons["type_of_residence"] == 1]

    # Only allow people with a building ID
    df_persons = df_persons[df_persons["federal_building_id"] < 999990000]

    # Only allow permanent residents
    df_persons = df_persons[df_persons["population_type"] == 1]

    # Merge STATPOP persons and households into a list of persons with houeshold attributes
    df = pd.merge(df_persons, df_link, on=("person_id", "municipality_id"))
    df = pd.merge(df, df_households, on="household_id")

    # Impute the houeshold size for each STATPOP person
    df_size = df.groupby("household_id").size().reset_index(name="household_size")
    df = pd.merge(df, df_size, on="household_id")

    # Only allow plausible households
    df = df[df["plausible"] == 1]

    # Only allow houesholds under a certain size
    df = df[df["household_size"] <= c.MAXIMUM_HOUSEHOLD_SIZE]

    # Remove all households where ALL persons are under a certain age
    df_filter = df[["household_id", "age"]].groupby("household_id").max().reset_index()
    df_filter.loc[:, "all_under_age"] = df_filter["age"] < c.MINIMUM_AGE_PER_HOUSEHOLD

    df = pd.merge(df, df_filter[["household_id", "all_under_age"]], on="household_id")
    df = df[~df["all_under_age"]]

    # This mapping comes from KM
    for from_value, to_value in zip((1, 2, 3, 4, 5, 6, 7, -9), (
            c.MARITAL_STATUS_SINGLE, c.MARITAL_STATUS_MARRIED,
            c.MARITAL_STATUS_SEPARATE, c.MARITAL_STATUS_SEPARATE,
            c.MARITAL_STATUS_SINGLE, c.MARITAL_STATUS_MARRIED,
            c.MARITAL_STATUS_SEPARATE, c.MARITAL_STATUS_SINGLE
    )):
        df.loc[df["marital_status"] == from_value, "marital_status_new"] = to_value

    df["marital_status"] = df["marital_status_new"]
    del df["marital_status_new"]

    # Some adjustments from KM
    data.utils.fix_marital_status(df, c)
    data.utils.assign_household_class(df, c)

    # Turn sex and nationality into an actual 0-based class
    df["sex"] -= 1
    df["nationality"] -= 1

    # Get the age class
    df["age_class"] = np.digitize(df["age"], c.AGE_CLASS_UPPER_BOUNDS)

    # Impute spatial information
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_zones = context.stage("data.spatial.zones")
    df_municipality_types = context.stage("data.spatial.municipality_types")
    df_quarters = context.stage("data.spatial.quarters")
    df_cantons = context.stage("data.spatial.cantons")

    df_spatial = pd.DataFrame(df[["person_id", "home_x", "home_y"]])
    df_spatial = data.spatial.utils.to_gpd(context, df_spatial, "home_x", "home_y", coord_type="home")

    # add WGS84 coordinates (longitude/latitude)
    df_spatial = add_wgs84_coordinates(df_spatial, drop_geometry=False)

    # copy original coordinates for comparison
    df_spatial['orig_lat'] = df_spatial['latitude']
    df_spatial['orig_lon'] = df_spatial['longitude']

    # Add geo noise to coordinates (epsilon=1.0 for moderate privacy protection)
    epsilon = 1.0
    noisy_lats, noisy_lons = zip(*[
        add_geo_noise(lat, lon, epsilon) 
        for lat, lon in zip(df_spatial['latitude'], df_spatial['longitude'])
    ])
    df_spatial['latitude'] = noisy_lats
    df_spatial['longitude'] = noisy_lons

    # Compute and display the distance change in kilometers for verification
    df_spatial["delta_km"] = [
        distance((orig_lat, orig_lon), (noisy_lat, noisy_lon)).km
        for orig_lat, orig_lon, noisy_lat, noisy_lon in zip(
            df_spatial["orig_lat"], df_spatial["orig_lon"],
            df_spatial["latitude"], df_spatial["longitude"]
        )
    ]

    # Print some basic stats
    print("=== Geo Noise Effect Summary ===")
    print(df_spatial["delta_km"].describe())
    print("Max delta (km):", df_spatial["delta_km"].max())
    print("Median delta (km):", df_spatial["delta_km"].median())

    # Optionally: remove delta/temporary columns later if not needed
    df_spatial.drop(columns=["delta_km", "orig_lat", "orig_lon"], inplace=True)

    # Convert noisy WGS84 coordinates back to Swiss coordinates
    points_df = gpd.GeoDataFrame(
        df_spatial,
        geometry=gpd.points_from_xy(df_spatial['longitude'], df_spatial['latitude'], crs="EPSG:4326")
    )
    points_df = points_df.to_crs("EPSG:2056")
    df_spatial['home_x'] = points_df.geometry.x
    df_spatial['home_y'] = points_df.geometry.y
    
    # Update df_spatial geometry to match Swiss coordinates
    df_spatial = gpd.GeoDataFrame(
        df_spatial,
        geometry=points_df.geometry,
        crs="EPSG:2056"
    )

    # Impute municipalities
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_municipalities, "person_id", "municipality_id",
                                            zone_type="municipality", point_type="home")[
        ["person_id", "municipality_id", "geometry"]])
    df_spatial["municipality_id"] = df_spatial["municipality_id"].astype(np.int)

    # Impute quarters
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_quarters, "person_id", "quarter_id",
                                            fix_by_distance=False, zone_type="quarter", point_type="home")[
        ["person_id", "municipality_id", "quarter_id", "geometry"]])

    # Impute cantons
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_cantons, "person_id", "canton_id",
                                            zone_type="canton", point_type="home")[
        ["person_id", "municipality_id", "quarter_id", "canton_id", "geometry"]])

    # Impute municipality types
    df_spatial = data.spatial.municipality_types.impute(df_spatial, df_municipality_types)

    # Impute zones
    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)

    assert (len(df) == len(df_spatial))

    del df["municipality_id"]
    df = pd.merge(
        df, df_spatial[["person_id", "zone_id", "municipality_type", "municipality_id", "quarter_id", "canton_id"]],
        on="person_id"
    )

    df["home_zone_id"] = df["zone_id"]
    df["home_municipality_id"] = df["municipality_id"]
    df["home_quarter_id"] = df["quarter_id"]

    # Impute SP region
    df = data.spatial.cantons.impute_sp_region(df)

    # Impute population density
    data.statpop.density.impute(
        context, 
        context.stage("data.statpop.density"), df, 
        "home_x", "home_y", 
        radius = c.POPULATION_DENSITY_RADIUS,
        chunk_size=1e5,
        point_type="home")

    # Impute OV Guteklasse
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df_spatial, ["person_id"], chunk_size=1e3, point_type="home")
    df = pd.merge(df, df_spatial[["person_id", "ovgk"]], on=["person_id"], how="left")

    # Save original statpop person and household ids
    df["statpop_person_id"] = df["person_id"].astype(int)
    df["statpop_household_id"] = df["household_id"].astype(int)

    # Identify households with children
    children_columns = []
    for upper_age in [3, 6, 12, 18]:
        col_name          = "N_children_under_"+str(upper_age)
        children          = df[df["age"]<upper_age]
        hhl_with_children = np.unique(children["household_id"].values.tolist())
        df.loc[:, col_name] = df["household_id"].isin(hhl_with_children)

        children_columns.append(col_name)

    # Wrap everything up
    df = df[[
        "person_id", "household_id",
        "sex", "age",
        "home_x", "home_y",
        "marital_status", "nationality",
        "household_size",
        "age_class", "household_size_class", "home_zone_id", "municipality_type",
        "home_municipality_id", "home_quarter_id", "canton_id", "population_density", "sp_region", "ovgk",
        "statpop_person_id", "statpop_household_id"]+children_columns]

    df = data.statpop.head_of_household.impute(df, c)

    return df
