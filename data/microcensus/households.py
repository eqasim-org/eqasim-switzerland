import pandas as pd
import numpy as np
import data.utils
import data.constants as c
import pyproj

def configure(context, require):
    require.config("raw_data_path")
    # require.cache = False

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df_mz_households = pd.read_csv(
        "%s/microcensus/haushalte.csv" % raw_data_path, sep = ",", encoding = "latin1")

    # Simple attributes
    df_mz_households["home_structure"] = df_mz_households["W_STRUKTUR_AGG_2000"]
    df_mz_households["household_size"] = df_mz_households["hhgr"]
    df_mz_households["number_of_cars"] = df_mz_households["f30100"]
    df_mz_households["number_of_bikes"] = df_mz_households["f32200a"]
    df_mz_households["person_id"] = df_mz_households["HHNR"]
    df_mz_households["household_weight"] = df_mz_households["WM"]

    # Income
    df_mz_households["income_class"] = df_mz_households["F20601"] - 1 # Turn into zero-based class
    df_mz_households["income_class"] = np.maximum(-1, df_mz_households["income_class"]) # Make all "invalid" entries -1

    # Convert coordinates to LV95
    coords = df_mz_households[["W_X_CH1903", "W_Y_CH1903"]].values
    x, y = pyproj.transform(c.CH1903, c.CH1903_PLUS, coords[:,0], coords[:,1])
    df_mz_households.loc[:, "home_x"] = x
    df_mz_households.loc[:, "home_y"] = y

    # Class variable for number of cars
    df_mz_households["number_of_cars_class"] = 0
    df_mz_households.loc[df_mz_households["number_of_cars"] > 0, "number_of_cars_class"] = np.minimum(
        c.MAX_NUMBER_OF_CARS_CLASS, df_mz_households["number_of_cars"])

    # Bike availability depends on household size. (TODO: Would it make sense to use the same concept for cars?)
    df_mz_households["number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_NONE
    df_mz_households.loc[df_mz_households["number_of_bikes"] > 0, "number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_SOME
    df_mz_households.loc[
        df_mz_households["number_of_bikes"] >= df_mz_households["household_size"],
        "number_of_bikes_class"] = c.BIKE_AVAILABILITY_FOR_ALL

    # Houeshold size class
    data.utils.assign_household_class(df_mz_households)

    # Wrap it up
    return df_mz_households[[
        "person_id", "household_size", "number_of_cars", "number_of_bikes", "income_class",
        "home_x", "home_y", "household_size_class", "number_of_cars_class", "number_of_bikes_class", "household_weight"
    ]]
