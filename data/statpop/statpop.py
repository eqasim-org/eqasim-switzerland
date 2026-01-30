import numpy as np
import pandas as pd
import gc

from data.spatial.cantons import impute_sp_region
from data.spatial.municipality_types import impute as impute_municipality_types
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
    context.stage("data.spatial.districts")
    context.stage("data.spatial.ovgk")
    context.stage("data.constants")
    context.config("threads")


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
    df = pd.merge(df, df_households, on="household_id") # This removes the great majority of collective housing residents, there are only ~700 left at this point.

    # Impute the houeshold size for each STATPOP person
    df_size = df.groupby("household_id").size().reset_index(name="household_size")
    df = pd.merge(df, df_size, on="household_id")

    # Only allow houesholds under a certain size
    initial_count = len(df)
    df = df[df["household_size"] <= c.MAXIMUM_HOUSEHOLD_SIZE]
    final_count = len(df)
    print(f"{initial_count - final_count} persons were filtered out based on the the household size max constraint.")

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
    df["nationality"] -= 1 # 0:swiss 1:rest

    # Get the age class
    df["age_class"] = np.digitize(df["age"], c.AGE_CLASS_UPPER_BOUNDS)

    # Impute spatial information
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_zones = context.stage("data.spatial.zones")
    df_municipality_types = context.stage("data.spatial.municipality_types")
    df_quarters = context.stage("data.spatial.quarters")
    df_cantons = context.stage("data.spatial.cantons")
    df_districts = context.stage("data.spatial.districts")

    df_spatial = pd.DataFrame(df[["person_id", "home_x", "home_y"]])
    df_spatial = data.spatial.utils.to_gpd(context, df_spatial, "home_x", "home_y", coord_type="home")

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
    
    # Impute districts
    df_spatial = (data.spatial.utils.impute(context, df_spatial, df_districts, "person_id", "district_id",
                                            zone_type="district", point_type="home")[
        ["person_id", "municipality_id", "quarter_id", "district_id", "canton_id", "geometry"]])

    # Impute municipality types
    df_spatial = impute_municipality_types(df_spatial, df_municipality_types)

    # Impute zones
    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)

    assert (len(df) == len(df_spatial))

    del df["municipality_id"]
    df = pd.merge(
        df, df_spatial[["person_id", "zone_id", "municipality_type", "municipality_id", "quarter_id", "canton_id", "district_id"]],
        on="person_id"
    )

    df["home_zone_id"] = df["zone_id"]
    df["home_municipality_id"] = df["municipality_id"]
    df["home_quarter_id"] = df["quarter_id"]

    # Impute SP region
    df = impute_sp_region(df)

    # Impute population density
    data.statpop.density.impute_parallel(
        context, 
        context.stage("data.statpop.density"), df, 
        "home_x", "home_y", 
        radius = c.POPULATION_DENSITY_RADIUS,
        chunk_size = 10000, # it looks that 10k is better than 1k, did not test more
        point_type = "home",
        n_jobs = context.config("threads"))

    # Impute OV Guteklasse
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df_spatial, ["person_id"], chunk_size=1e3, point_type="home")
    df = pd.merge(df, df_spatial[["person_id", "ovgk"]], on=["person_id"], how="left")

    # Save original statpop person and household ids
    df["statpop_person_id"]    = df["person_id"].astype(int)
    df["statpop_household_id"] = df["household_id"].astype(int)

    # Identify households with children
    children_columns = []
    for upper_age in [3, 6, 12, 18]:
        col_name          = "N_children_under_" + str(upper_age)
        children          = df[df["age"] < upper_age]
        hhl_with_children = np.unique(children["household_id"].values.tolist())
        df.loc[:, col_name] = df["household_id"].isin(hhl_with_children)

        children_columns.append(col_name)

    # Wrap everything up
    df = df[[
        "person_id", "household_id",
        "sex", "age",
        "home_x", "home_y",
        "marital_status",
        "household_size",
        "age_class", "household_size_class", "home_zone_id", "municipality_type",
        "home_municipality_id", "home_quarter_id", "canton_id", "district_id", "population_density", "sp_region", "ovgk",
        "collective_housing_resident", "nationality"] + children_columns]

    df = data.statpop.head_of_household.impute(df, c)

    return df
