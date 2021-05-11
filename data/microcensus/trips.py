import numpy as np
import pandas as pd
import pyproj

import data.constants as c


def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df_mz_trips = pd.read_csv("%s/microcensus/wege.csv" % data_path, encoding = "latin1")
    df_mz_stages = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")

    df_mz_trips = df_mz_trips[[
        "HHNR", "WEGNR", "f51100", "f51400", "wzweck1", "wzweck2", "wmittel",
        "S_X_CH1903", "S_Y_CH1903", "Z_X_CH1903", "Z_Y_CH1903", "W_X_CH1903", "W_Y_CH1903",
        "w_rdist"
    ]]

    df_mz_stages = df_mz_stages[[
        "HHNR", "WEGNR", "ETNR", "f51300"
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

    df_mz_trips["mode_detailed"] = df_mz_trips["mode"]
    df_mz_trips.loc[df_mz_trips["wmittel"] == 1, "mode_detailed"] = "plane"
    df_mz_trips.loc[df_mz_trips["wmittel"] == 11, "mode_detailed"] = "taxi"

    # Find passenger trips
    df_mz_stages["is_car_passenger"] = df_mz_stages["f51300"] == 8
    df_passengers = df_mz_stages[["HHNR", "WEGNR", "is_car_passenger"]].groupby(["HHNR", "WEGNR"]).sum().reset_index()
    df_mz_trips = pd.merge(df_mz_trips, df_passengers, on = ["HHNR", "WEGNR"], how = "left")
    df_mz_trips.loc[df_mz_trips["is_car_passenger"] > 0, "mode_detailed"] = "car_passenger"
    df_mz_trips.loc[df_mz_trips["is_car_passenger"] > 0, "mode"] = "car_passenger"
    del df_mz_trips["is_car_passenger"]

    # Second, adjust the purposes
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -99, "purpose"] = "unknown" # Pseudo stage
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -98, "purpose"] = "unknown" # No answer
    df_mz_trips.loc[df_mz_trips["wzweck1"] == -97, "purpose"] = "unknown" # Don't know
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 1, "purpose"] = "interaction" # Transfer, change of mode, park car
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 2, "purpose"] = "work" # Work
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 3, "purpose"] = "education" # Education
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 4, "purpose"] = "shop" # Shopping
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 5, "purpose"] = "other" # Chores, use of public services
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 6, "purpose"] = "work" # Business activity
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 7, "purpose"] = "work" # Business trip
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 8, "purpose"] = "leisure" # Leisure
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 9, "purpose"] = "other" # Bring children
    df_mz_trips.loc[df_mz_trips["wzweck1"] == 10, "purpose"] = "other" # Bring others (disabled, ...)
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
    for mz_attribute, df_attribute in [("Z", "destination"), ("S", "origin"), ("W", "home")]:
        coords = df_mz_trips[["%s_X_CH1903" % mz_attribute, "%s_Y_CH1903" % mz_attribute]].values
        transformer = pyproj.Transformer.from_crs(c.CH1903, c.CH1903_PLUS)
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        df_mz_trips.loc[:, "%s_x" % df_attribute] = x
        df_mz_trips.loc[:, "%s_y" % df_attribute] = y

    # Add crowfly distance
    df_mz_trips.loc[:, "crowfly_distance"] = np.sqrt(
        (df_mz_trips["origin_x"] - df_mz_trips["destination_x"])**2 +
        (df_mz_trips["origin_y"] - df_mz_trips["destination_y"])**2)

    # Add activity durations by joining the trips with themselves
    df_mz_trips.loc[:, "previous_trip_id"] = df_mz_trips["trip_id"] -1

    df_durations = pd.merge(
        df_mz_trips[["person_id", "trip_id", "departure_time"]],
        df_mz_trips[["person_id", "previous_trip_id", "arrival_time"]],
        left_on = ["person_id", "trip_id"], right_on = ["person_id", "previous_trip_id"])

    df_durations.loc[:, "activity_duration"] = df_durations["arrival_time"] - df_durations["departure_time"]

    df_mz_trips = pd.merge(
        df_mz_trips, df_durations[["person_id", "trip_id", "activity_duration"]],
        on = ["person_id", "trip_id"], how = "left"
    )

    # Filter persons for which we do not have sufficient information
    unknown_ids = set(df_mz_trips[
        (df_mz_trips["mode"] == "unknown") | (df_mz_trips["purpose"] == "unknown")
    ]["person_id"])

    print("  Removed %d persons with trips with unknown mode or unknown purpose" % len(unknown_ids))
    df_mz_trips = df_mz_trips[~df_mz_trips["person_id"].isin(unknown_ids)]

    # Filter persons which do not start or end with "home"
    df_end = df_mz_trips[["person_id", "trip_id", "purpose"]].sort_values("trip_id", ascending = False).drop_duplicates("person_id")
    df_end = df_end[df_end["purpose"] != "home"]

    before_length = len(np.unique(df_mz_trips["person_id"]))
    df_mz_trips = df_mz_trips[~df_mz_trips["person_id"].isin(df_end["person_id"])]
    after_length = len(np.unique(df_mz_trips["person_id"]))
    print("  Removed %d persons with trips not ending with 'home'" % (before_length - after_length,))

    df_start = df_mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "home_x", "home_y"]]
    df_start = df_start[
        (df_start["trip_id"] == 1) & ((df_start["origin_x"] != df_start["home_x"]) |
        (df_start["origin_y"] != df_start["home_y"]))
    ]

    before_length = len(np.unique(df_mz_trips["person_id"]))
    df_mz_trips = df_mz_trips[~df_mz_trips["person_id"].isin(df_start["person_id"])]
    after_length = len(np.unique(df_mz_trips["person_id"]))
    print("  Removed %d persons with trips not starting at home location" % (before_length - after_length,))

    # Parking cost
    df_mz_stages = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")

    df_cost = pd.DataFrame(df_mz_stages[["HHNR", "WEGNR", "f51330"]], copy = True)
    df_cost.columns = ["person_id", "trip_id", "parking_cost"]
    df_cost["parking_cost"] = np.maximum(0, df_cost["parking_cost"])
    df_cost = df_cost.groupby(["person_id", "trip_id"]).sum().reset_index()

    df_mz_trips = pd.merge(df_mz_trips, df_cost, on = ["person_id", "trip_id"], how = "left")
    assert(not np.any(np.isnan(df_mz_trips["parking_cost"])))

    # Network distance
    df_mz_trips["network_distance"] = df_mz_trips["w_rdist"] * 1000.0

    return df_mz_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "destination_x", "destination_y", "origin_x", "origin_y",
        "activity_duration", "crowfly_distance", "parking_cost", "network_distance",
        "mode_detailed"
    ]]
