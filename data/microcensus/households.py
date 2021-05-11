import numpy as np
import pandas as pd
import pyproj

import data.constants as c
import data.spatial.cantons
import data.spatial.municipalities
import data.spatial.municipality_types
import data.spatial.ovgk
import data.spatial.utils
import data.spatial.zones
import data.utils
import data.utils


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipality_types")
    context.stage("data.statpop.density")
    context.stage("data.spatial.ovgk")


def execute(context):
    data_path = context.config("data_path")

    df_mz_households = pd.read_csv(
        "%s/microcensus/haushalte.csv" % data_path, sep=",", encoding="latin1")

    # Simple attributes
    df_mz_households["home_structure"] = df_mz_households["W_STRUKTUR_AGG_2000"]
    df_mz_households["household_size"] = df_mz_households["hhgr"]
    df_mz_households["number_of_cars"] = np.maximum(0, df_mz_households["f30100"])
    df_mz_households["number_of_bikes"] = df_mz_households["f32200a"]
    df_mz_households["person_id"] = df_mz_households["HHNR"]
    df_mz_households["household_weight"] = df_mz_households["WM"]

    # Income
    df_mz_households["income_class"] = df_mz_households["F20601"] - 1  # Turn into zero-based class
    df_mz_households["income_class"] = np.maximum(-1, df_mz_households["income_class"])  # Make all "invalid" entries -1

    # Convert coordinates to LV95
    coords = df_mz_households[["W_X_CH1903", "W_Y_CH1903"]].values
    transformer = pyproj.Transformer.from_crs(c.CH1903, c.CH1903_PLUS)
    x, y = transformer.transform(coords[:, 0], coords[:, 1])
    df_mz_households.loc[:, "home_x"] = x
    df_mz_households.loc[:, "home_y"] = y

    # Class variable for number of cars
    df_mz_households["number_of_cars_class"] = 0
    df_mz_households.loc[df_mz_households["number_of_cars"] > 0, "number_of_cars_class"] = np.minimum(
        c.MAX_NUMBER_OF_CARS_CLASS, df_mz_households["number_of_cars"])

    # Bike availability depends on household size. (TODO: Would it make sense to use the same concept for cars?)
    df_mz_households["number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_NONE
    df_mz_households.loc[
        df_mz_households["number_of_bikes"] > 0, "number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_SOME
    df_mz_households.loc[
        df_mz_households["number_of_bikes"] >= df_mz_households["household_size"],
        "number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_ALL

    # Household size class
    data.utils.assign_household_class(df_mz_households)

    # Region information
    # (acc. to Analyse der SP-Befragung 2015 zur Verkehrsmodus- und Routenwahl)
    df_mz_households["canton_id"] = df_mz_households["W_KANTON"]
    df_mz_households = data.spatial.cantons.impute_sp_region(df_mz_households)

    # Impute spatial information
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_zones = context.stage("data.spatial.zones")
    df_municipality_types = context.stage("data.spatial.municipality_types")

    df_spatial = pd.DataFrame(df_mz_households[["person_id", "home_x", "home_y"]])
    df_spatial = data.spatial.utils.to_gpd(context, df_spatial, "home_x", "home_y")
    df_spatial = data.spatial.utils.impute(context, df_spatial, df_municipalities, "person_id", "municipality_id")
    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)
    df_spatial = data.spatial.municipality_types.impute(df_spatial, df_municipality_types)

    df_mz_households = pd.merge(
        df_mz_households, df_spatial[["person_id", "zone_id", "municipality_type"]],
        on="person_id"
    )

    df_mz_households["home_zone_id"] = df_mz_households["zone_id"]

    # Impute density
    data.statpop.density.impute(context.stage("data.statpop.density"), df_mz_households, "home_x", "home_y")

    # Impute OV Guteklasse
    print("Imputing ÖV Güteklasse ...")
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df_spatial, ["person_id"])
    df_mz_households = pd.merge(df_mz_households, df_spatial[["person_id", "ovgk"]], on=["person_id"], how="left")

    # Wrap it up
    return df_mz_households[[
        "person_id", "household_size", "number_of_cars", "number_of_bikes", "income_class",
        "home_x", "home_y", "household_size_class", "number_of_cars_class", "number_of_bikes_class", "household_weight",
        "home_zone_id", "municipality_type", "sp_region", "population_density", "canton_id", "ovgk"
    ]]
