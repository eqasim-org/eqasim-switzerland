import numpy as np
import pandas as pd

import data.constants as c
import data.spatial.cantons
import data.spatial.municipalities
import data.spatial.municipality_types
import data.spatial.ovgk
import data.spatial.utils
import data.spatial.zones
import data.statpop.density
import data.statpop.head_of_household
import data.utils
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


def execute(context):
    df_persons = context.stage("data.statpop.persons")
    df_households = context.stage("data.statpop.households")
    df_link = context.stage("data.statpop.link")

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

    # Only allow houesholds under a certian size
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
    data.utils.fix_marital_status(df)
    data.utils.assign_household_class(df)

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
    data.statpop.density.impute(context.stage("data.statpop.density"), df, "home_x", "home_y")

    # Impute OV Guteklasse
    print("Imputing ÖV Güteklasse ...")
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df_spatial, ["person_id"])
    df = pd.merge(df, df_spatial[["person_id", "ovgk"]], on=["person_id"], how="left")

    # Save original statpop person and household ids
    df["statpop_person_id"] = df["person_id"].astype(int)
    df["statpop_household_id"] = df["household_id"].astype(int)

    # Wrap everything up
    df = df[[
        "person_id", "household_id",
        "sex", "age",
        "home_x", "home_y",
        "marital_status", "nationality",
        "household_size",
        "age_class", "household_size_class", "home_zone_id", "municipality_type",
        "home_municipality_id", "home_quarter_id", "canton_id", "population_density", "sp_region", "ovgk",
        "statpop_person_id", "statpop_household_id"]]

    df = data.statpop.head_of_household.impute(df)

    return df
