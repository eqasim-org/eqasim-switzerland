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

    df_mz_trips = pd.read_csv("%s/microcensus/wege.csv" % raw_data_path, encoding = "latin1")

    df_mz_trips = df_mz_trips[[
        "HHNR", "WEGNR", "f51100", "f51400", "wzweck1", "wzweck2", "wmittel",
        "S_X_CH1903", "S_Y_CH1903", "Z_X_CH1903", "Z_Y_CH1903"
    ]]

    # First, adjust the modes
    df_mz_trips.loc[df_mz_trips["wmittel"] == -99, "mode"] = "unknown" # Pseudo stage
    df_mz_trips.loc[df_mz_trips["wmittel"] == 1, "mode"] = "pt" # Plane
    df_mz_trips.loc[df_mz_trips["wmittel"] == 2, "mode"] = "pt" # Train
    df_mz_trips.loc[df_mz_trips["wmittel"] == 3, "mode"] = "pt" # Postauto
    df_mz_trips.loc[df_mz_trips["wmittel"] == 4, "mode"] = "pt" # Ship
    df_mz_trips.loc[df_mz_trips["wmittel"] == 5, "mode"] = "pt" # Tram
    df_mz_trips.loc[df_mz_trips["wmittel"] == 6, "mode"] = "pt" # Bus
    df_mz_trips.loc[df_mz_trips["wmittel"] == 7, "mode"] = "pt" # other PT
    df_mz_trips.loc[df_mz_trips["wmittel"] == 8, "mode"] = "pt" # Reisecar -> I think this is a coach in Swiss German?
    df_mz_trips.loc[df_mz_trips["wmittel"] == 9, "mode"] = "car" # Car
    df_mz_trips.loc[df_mz_trips["wmittel"] == 10, "mode"] = "car" # Truck
    df_mz_trips.loc[df_mz_trips["wmittel"] == 11, "mode"] = "pt" # Taxi
    df_mz_trips.loc[df_mz_trips["wmittel"] == 12, "mode"] = "car" # Motorbike
    df_mz_trips.loc[df_mz_trips["wmittel"] == 13, "mode"] = "car" # Mofa
    df_mz_trips.loc[df_mz_trips["wmittel"] == 14, "mode"] = "bike" # Biciycle / E-bike
    df_mz_trips.loc[df_mz_trips["wmittel"] == 15, "mode"] = "walk" # Walking
    df_mz_trips.loc[df_mz_trips["wmittel"] == 16, "mode"] = "car" # "Machines similar to a vehicle"
    df_mz_trips.loc[df_mz_trips["wmittel"] == 17, "mode"] = "unknown" # Other / don't know

    # Put a flag if this agent is using a Flugi
    df_mz_trips.loc[:, "uses_plane"] = df_mz_trips["wmittel"] == 1

    # Second, adjust the purposes
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -99, "purpose"] = "unknown" # Pseudo stage
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -98, "purpose"] = "unknown" # No answer
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -97, "purpose"] = "unknown" # Don't know
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 1, "purpose"] = "interaction" # Transfer, change of mode, park car
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 2, "purpose"] = "work" # Work
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 3, "purpose"] = "education" # Education
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 4, "purpose"] = "shop" # Shopping
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 5, "purpose"] = "service" # Chores, use of public services
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 6, "purpose"] = "work" # Business activity
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 7, "purpose"] = "work" # Business trip
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 8, "purpose"] = "leisure" # Leisure
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 9, "purpose"] = "service" # Bring children
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 10, "purpose"] = "service" # Bring others (disabled, ...)
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 11, "purpose"] = "home" # Return home
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 12, "purpose"] = "unknown" # Other
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 13, "purpose"] = "border" # Going out of country

    # Adjust trips back home
    df_mz_trips.loc[df_mz_trips["wzweck2"] > 1, "purpose"] = "home"

    # Adjust times
    df_mz_trips.loc[:, "departure_time"] = df_mz_trips["f51100"] * 60
    df_mz_trips.loc[:, "arrival_time"] = df_mz_trips["f51400"] * 60

    # Adjust id
    df_mz_trips.loc[:, "person_id"] = df_mz_trips["HHNR"]
    df_mz_trips.loc[:, "trip_id"] = df_mz_trips["WEGNR"]

    # Adjust coordinates
    coords = df_mz_trips[["Z_X_CH1903", "Z_Y_CH1903"]].values
    x, y = pyproj.transform(c.CH1903, c.CH1903_PLUS, coords[:,0], coords[:,1])
    df_mz_trips.loc[:, "destination_x"] = x
    df_mz_trips.loc[:, "destination_y"] = y

    return df_mz_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "destination_x", "destination_y",
        "uses_plane"
    ]]
