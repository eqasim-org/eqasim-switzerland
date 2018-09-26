import data.utils
import pandas as pd
import numpy as np
import data.constants as c
import data.statpop.head_of_household
import data.statpop.home_structure

def configure(context, require):
    require.stage("data.statpop.persons")
    require.stage("data.statpop.households")
    require.stage("data.statpop.link")
    require.stage("data.microcensus.households")

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
    df = pd.merge(df_persons, df_link, on = ("person_id", "municipality_id"))
    df = pd.merge(df, df_households, on = "household_id")

    # Impute the houeshold size for each STATPOP person
    df_size = df.groupby("household_id").size().reset_index(name = "household_size")
    df = pd.merge(df, df_size, on = "household_id")

    # Only allow plausible households
    df = df[df["plausible"] == 1]

    # Only allow houesholds under a certian size
    df = df[df["household_size"] <= c.MAXIMUM_HOUSEHOLD_SIZE]

    # Remove all households where ALL persons are under a certain age
    df_filter = df[["household_id", "age"]].groupby("household_id").max().reset_index()
    df_filter.loc[:, "all_under_age"] = df_filter["age"] < c.MINIMUM_AGE_PER_HOUSEHOLD

    df = pd.merge(df, df_filter[["household_id", "all_under_age"]], on = "household_id")
    df = df[~df["all_under_age"]]

    # This mapping comes from KM
    for from_value, to_value in zip( (1, 2, 3, 4, 5, 6, 7, -9), (
        c.MARITAL_STATUS_SINGLE, c.MARITAL_STATUS_MARRIED,
        c.MARITAL_STATUS_SEPARATE, c.MARITAL_STATUS_SEPARATE,
        c.MARITAL_STATUS_SINGLE, c.MARITAL_STATUS_MARRIED,
        c.MARITAL_STATUS_SEPARATE, c.MARITAL_STATUS_SINGLE
    ) ):
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

    # Wrap everything up
    df = df[[
        "person_id", "household_id",
        "sex", "age",
        "home_x", "home_y",
        "marital_status", "nationality",
        "household_size",
        "age_class", "household_size_class"]]

    df = data.statpop.head_of_household.impute(df)
    df = data.statpop.home_structure.impute(df, context.stage("data.microcensus.households"))
    return df
