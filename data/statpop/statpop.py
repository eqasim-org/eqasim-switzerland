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
    context.config("threads")


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
    initial_count = len(df_persons)
    df_persons = df_persons[df_persons["type_of_residence"] == 1]
    final_count = len(df_persons)
    print(f"{initial_count - final_count} persons were filtered out based on the type of residence.")
    
    # Only allow permanent residents
    initial_count = len(df_persons)
    df_persons = df_persons[df_persons["population_type"] == 1]
    final_count = len(df_persons)
    print(f"{initial_count - final_count} persons without permanent residence filtered out.")


    # Merge STATPOP persons and households into a list of persons with household attributes
    df = pd.merge(df_persons, df_link, on=("person_id", "municipality_id"))
    df = pd.merge(df, df_households, on="household_id")

    # Impute the houeshold size for each STATPOP person
    df_size = df.groupby("household_id").size().reset_index(name="household_size")
    df = pd.merge(df, df_size, on="household_id")

    # Only allow houesholds under a certain size
    df = df[df["household_size"] <= c.MAXIMUM_HOUSEHOLD_SIZE]

    # Remove all households where ALL persons are under a certain age
    df_filter = df[["household_id", "age"]].groupby("household_id").max().reset_index()
    df_filter.loc[:, "all_under_age"] = df_filter["age"] < c.MINIMUM_AGE_PER_HOUSEHOLD

    df = pd.merge(df, df_filter[["household_id", "all_under_age"]], on="household_id")
    df = df[~df["all_under_age"]]

    # This mapping comes Strukturerhebung (as it is the same as in STATPOP) Codelist
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

    # ============= Adding geo noise =============

    epsilon = 1.0

    # Ensure CRS is EPSG:2056
    assert df_spatial.crs.to_epsg() == 2056, "Input GeoDataFrame is not in EPSG:2056"
    
    #  save original coordinates and geometry for comparison
    df_spatial['orig_x'] = df_spatial['home_x']
    df_spatial['orig_y'] = df_spatial['home_y']
    df_spatial['orig_geometry'] = df_spatial.geometry

    # apply geo noise to home_x and home_y
    dx_dy = np.array([geo_indistinguishability_noise_meters(epsilon) for _ in range(len(df_spatial))])
    dxs, dys = dx_dy[:, 0], dx_dy[:, 1]

    # apply noise
    df_spatial['home_x'] += dxs
    df_spatial['home_y'] += dys

    # recreate geometry 
    df_spatial['geometry'] = gpd.points_from_xy(df_spatial['home_x'], df_spatial['home_y'], crs="EPSG:2056")
    df_spatial = gpd.GeoDataFrame(df_spatial, geometry='geometry', crs="EPSG:2056")

    # verification
    df_spatial['final_geometry'] = df_spatial.geometry

    # Compute Euclidean distance in meters between original and final positions
    df_spatial['true_distance'] = df_spatial.apply(
        lambda row: row['orig_geometry'].distance(row['final_geometry']),
        axis=1
    )

    print("=== True Distance Moved (Post-Imputation) ===")
    print(df_spatial['true_distance'].describe())
    print("Number of points actually moved > 1 km:", (df_spatial['true_distance'] > 1000).sum())

    # =======================================================================

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
    # overwrite this with the new home coordinates

    print(" The columns of DF originally", list(df.columns))
    print(" The columns of DF SPATIAL", list(df_spatial.columns))
    
    # Extract coordinates from geometry
    df["home_x"] = df_spatial.geometry.x
    df["home_y"] = df_spatial.geometry.y
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
    data.statpop.density.impute_parallel(
        context, 
        context.stage("data.statpop.density"), df, 
        "home_x", "home_y", 
        radius = c.POPULATION_DENSITY_RADIUS,
        chunk_size=10000, # it looks that 10k is better than 1k, did not test more
        point_type="home",
        n_jobs=context.config("threads"))

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
