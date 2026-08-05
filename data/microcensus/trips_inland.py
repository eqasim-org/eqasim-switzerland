import numpy as np
import pandas as pd
import pyproj
import geopandas as gpd
from shapely.geometry import LineString
import warnings

try:
    from pandas.errors import SettingWithCopyWarning
except ImportError:
    from pandas.errors import ChainedAssignmentError as SettingWithCopyWarning
import logging

logger = logging.getLogger("synpp")
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def configure(context):
    context.config("data_path")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")

    context.stage("data.microcensus.trips")
    context.config("output_path")


def execute(context):
    data_path = context.config("data_path")
    c         = context.stage("data.constants")

    _, _, _, crossborder_person_ids = context.stage("data.microcensus.trips")

    df_mz_trips   = pd.read_csv("%s/microcensus/wegeinland.csv" % data_path, encoding = "latin1")
    df_mz_stages  = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")
    df_mz_persons = pd.read_csv("%s/microcensus/zielpersonen.csv" % data_path, sep = ",", encoding = "latin1", parse_dates = ["USTag"])

    df_mz_trips = df_mz_trips[[
        "HHNR", "WEGNR", "f51100", "f51400", "wzweck1", "wzweck2", "wmittel",
        "S_X_CH1903", "S_Y_CH1903", "Z_X_CH1903", "Z_Y_CH1903", "W_X_CH1903", "W_Y_CH1903",
        "w_rdist"
    ]]

    df_mz_trips = df_mz_trips[df_mz_trips["HHNR"].isin(crossborder_person_ids)]
    output_path = context.config("output_path")

    df_mz_stages = df_mz_stages[[
        "HHNR", "WEGNR", "ETNR", "f51300"
    ]]

    df_mz_persons_work = df_mz_persons[["HHNR", "A_X_CH1903", "A_Y_CH1903", "AU_X_CH1903", "AU_Y_CH1903"]]

    df_mz_trips = df_mz_trips.merge(df_mz_persons_work, on = "HHNR", how = "left")

    del df_mz_persons_work
    del df_mz_persons

    print(len(df_mz_trips))

    # First, adjust the modes
    mode_map = {
        -99: "unknown",  # Pseudo stage
        1: "pt",         # Plane
        2: "pt",         # Train
        3: "pt",         # Postauto
        4: "pt",         # Ship
        5: "pt",         # Tram
        6: "pt",         # Bus
        7: "pt",         # Other PT
        8: "pt",         # Reisecar (coach)
        9: "car",        # Car
        10: "car",       # Truck
        11: "pt",        # Taxi
        12: "car",       # Motorbike
        13: "car",       # Mofa
        14: "bike",      # Bicycle / E-bike
        15: "walk",      # Walking
        16: "bike",      # Machines similar to a vehicle
        17: "unknown"    # Other / don't know
    }

    df_mz_trips["mode"] = df_mz_trips["wmittel"].map(mode_map)

    df_mz_trips["mode_detailed"] = df_mz_trips["mode"]
    df_mz_trips.loc[df_mz_trips["wmittel"] == 1, "mode_detailed"]  = "plane"
    df_mz_trips.loc[df_mz_trips["wmittel"] == 11, "mode_detailed"] = "taxi"

    # Find passenger trips
    df_mz_stages["is_car_passenger"] = df_mz_stages["f51300"] == 8
    df_passengers = df_mz_stages[["HHNR", "WEGNR", "is_car_passenger"]].groupby(["HHNR", "WEGNR"]).sum().reset_index()
    df_mz_trips = pd.merge(df_mz_trips, df_passengers, on = ["HHNR", "WEGNR"], how = "left")
    df_mz_trips.loc[df_mz_trips["is_car_passenger"] > 0, "mode_detailed"] = "car_passenger"
    df_mz_trips.loc[df_mz_trips["is_car_passenger"] > 0, "mode"]          = "car_passenger"
    del df_mz_trips["is_car_passenger"]

    # Second, adjust the purposes
    purpose_map = {
        -99: "unknown",     # Pseudo stage
        -98: "unknown",     # No answer
        -97: "unknown",     # Don't know
        1: "interaction",   # Transfer, change of mode, park car
        2: "work",          # Work
        3: "education",     # Education
        4: "shop",          # Shopping
        5: "other",         # Chores, use of public services
        6: "work_secondary",          # Business activity
        7: "work_secondary",          # Business trip
        8: "leisure",       # Leisure
        9: "other",         # Bring children
        10: "other",        # Bring others (disabled, etc.)
        11: "home",         # Return home
        12: "unknown",      # Other
        13: "border"        # Going out of country
    }

    df_mz_trips["purpose"] = df_mz_trips["wzweck1"].map(purpose_map)

    # Adjust trips back home
    df_mz_trips.loc[df_mz_trips["wzweck2"] > 1, "purpose"] = "home"

    # Adjust times
    df_mz_trips.loc[:, "departure_time"] = df_mz_trips["f51100"] * 60
    df_mz_trips.loc[:, "arrival_time"] = df_mz_trips["f51400"] * 60

    # Adjust id
    df_mz_trips.loc[:, "person_id"] = df_mz_trips["HHNR"]
    df_mz_trips.loc[:, "trip_id"] = df_mz_trips["WEGNR"]

    # Adjust coordinates
    for mz_attribute, df_attribute in [("Z", "destination"), ("S", "origin"), ("W", "home"), ("A", "work"), ("AU", "education")]:
        coords = df_mz_trips[["%s_X_CH1903" % mz_attribute, "%s_Y_CH1903" % mz_attribute]].values
        transformer = pyproj.Transformer.from_crs(c.CH1903, c.CH1903_PLUS)
        x, y = transformer.transform(coords[:, 0], coords[:, 1])
        df_mz_trips.loc[:, "%s_x" % df_attribute] = x
        df_mz_trips.loc[:, "%s_y" % df_attribute] = y

    # Add crowfly distance
    df_mz_trips.loc[:, "crowfly_distance"] = np.sqrt(
        (df_mz_trips["origin_x"] - df_mz_trips["destination_x"])**2 +
        (df_mz_trips["origin_y"] - df_mz_trips["destination_y"])**2)
    
    df_mz_trips.to_csv(f"{output_path}/wegeinland_raw.csv", index = False)

    # Filter persons for which we do not have sufficient information
    unknown_ids = set(df_mz_trips[
        (df_mz_trips["mode"] == "unknown") | (df_mz_trips["purpose"] == "unknown")
    ]["person_id"])

    print("  Removed %d persons with trips with unknown mode or unknown purpose" % len(unknown_ids))
    df_mz_trips = df_mz_trips[~df_mz_trips["person_id"].isin(unknown_ids)]

    filterout_ids = unknown_ids

    # Fixing the initial purpose (origin purpose of trip with trip_id = 1)
    df_start = df_mz_trips[df_mz_trips["trip_id"]==1]

    df_start.loc[(df_start["origin_x"] == df_start["home_x"]) & (df_start["origin_y"] == df_start["home_y"]), "origin_purpose"] = "home"
    df_start.loc[(df_start["origin_x"] == df_start["work_x"]) & (df_start["origin_y"] == df_start["work_y"]), "origin_purpose"] = "work" # Should have classified it as "work_main" but this creates issues later
    df_start.loc[(df_start["origin_x"] == df_start["education_x"]) & (df_start["origin_y"] == df_start["education_x"]), "origin_purpose"] = "education" # Same

    start_from_home        = df_start[df_start["origin_purpose"]=="home"]
    start_from_work        = df_start[df_start["origin_purpose"]=="work"]
    start_from_education   = df_start[df_start["origin_purpose"]=="education"]
    missing_origin_purpose = df_start[df_start["origin_purpose"].isna()]
    logger.info(f"  Assigned origin_purpose of initial trip to home to {len(start_from_home)} persons")
    logger.info(f"  Assigned origin_purpose of initial trip to work to {len(start_from_work)} persons")
    logger.info(f"  Assigned origin_purpose of initial trip to education to {len(start_from_education)} persons")
    logger.info(f"  Number of agents with missing initial purpose: {len(missing_origin_purpose)}")

    if len(missing_origin_purpose) > 0:
        logger.info(f"   Investigating stages for these {len(missing_origin_purpose)} agents.")
        del missing_origin_purpose["origin_purpose"]
        df_mz_stages                = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")
        df_mz_stages_origin         = df_mz_stages[["HHNR", "WEGNR", "ETNR", "f52950"]]
        df_mz_stages_origin         = df_mz_stages_origin[(df_mz_stages_origin["HHNR"].isin(missing_origin_purpose["person_id"])) & (df_mz_stages_origin["WEGNR"]==1)]
        df_mz_stages_origin.columns = ["person_id", "trip_id", "stage_id", "f52950"]
        df_mz_stages_origin         = df_mz_stages_origin.groupby(["person_id", "trip_id"])["f52950"].agg(lambda x: "-".join(x.astype(str))).reset_index()

        def clean_f52950(val):
            parts = str(val).split("-")
            clean_parts = [p for p in parts if p.strip() != "-99" and p.strip() != "99" and p.strip() != ""]
            return "-".join(clean_parts)

        df_mz_stages_origin["f52950"] = df_mz_stages_origin["f52950"].astype(str).apply(clean_f52950)
        df_mz_stages_origin.loc[df_mz_stages_origin["f52950"] == "", "f52950"] = np.nan

        # Clean purposes
        df_mz_stages_origin["f52950"] = df_mz_stages_origin["f52950"].astype("Int64")

        origin_map = {
            -98: "unknown",         # No answer
            -97: "unknown",         # Don't know
            97: "unknown",          # Don't know
            98: "unknown",          # No answer
            1: "interaction",       # Transfer, change of mode, park car
            2: "work",              # Work
            3: "education",         # Education
            4: "shop",              # Shopping
            5: "other",             # Chores, use of public services
            6: "work_secondary",    # Business activity
            7: "work_secondary",    # Business trip
            8: "leisure",           # Leisure
            9: "other",             # Bring children
            10: "other",            # Bring others (disabled, ...)
            11: "home",             # Return home
            12: "unknown",          # Other
            13: "border"            # Going out of country
        }

        df_mz_stages_origin["origin_purpose"] = df_mz_stages_origin["f52950"].map(origin_map).fillna("unknown")

        # 9.799% of week-end daily plans (with at least one trip) start from a location which is at this point classified as "unknown".
        # This share is 6.156% for work day daily plans. We decided not to care about this issue and to classify the origin purpose
        # of all these trips as "other". TODO for later - if someone has time -: figure out a better way to assign the origin purpose.

        df_mz_stages_origin.loc[df_mz_stages_origin["origin_purpose"]=="unknown", "origin_purpose"] = "other"
        
        del df_mz_stages_origin["f52950"]

        stages_origin_purpose = missing_origin_purpose.merge(df_mz_stages_origin[["person_id", "origin_purpose"]], on = "person_id", how = "left")
        vc = stages_origin_purpose['origin_purpose'].value_counts(dropna=False)

        logger.info("   Found distribution of origin purposes from the stages:")
        for value, count in vc.items():
            logger.info(f"     {value}: {count}")

        df_start_purpose = pd.concat([start_from_home, start_from_work, start_from_education, stages_origin_purpose])

        assert len(df_start_purpose) == len(df_start)

    df_start_purpose = df_start_purpose[["person_id", "trip_id", "origin_purpose"]]
    df_mz_trips      = df_mz_trips.merge(df_start_purpose, on = ["person_id", "trip_id"], how = "left")
    df_mz_trips      = df_mz_trips.rename(columns = {"origin_purpose": "origin_purpose_start"})

    # Create the trip origin_purpose column
    df_mz_trips["previous_trip_id"] = df_mz_trips["trip_id"] - 1

    df_mz_trips = df_mz_trips.merge(df_mz_trips[["person_id", "trip_id", "purpose"]].rename(columns={"purpose":"origin_purpose"}), 
                                left_on = ["person_id", "previous_trip_id"], right_on = ["person_id", "trip_id"], suffixes=("", "_"), how = "left")
    
    del df_mz_trips["trip_id_"]
    df_mz_trips.loc[df_mz_trips["origin_purpose"].isna(), "origin_purpose"] = df_mz_trips[df_mz_trips["origin_purpose"].isna()]["origin_purpose_start"]
    del df_mz_trips["origin_purpose_start"]

    # Now fix the activity durations
    df_durations = pd.merge(
        df_mz_trips[["person_id", "trip_id", "arrival_time"]],
        df_mz_trips[["person_id", "previous_trip_id", "departure_time"]],
        left_on = ["person_id", "trip_id"], right_on = ["person_id", "previous_trip_id"]
    )

    df_durations.loc[:, "activity_duration"] = df_durations["departure_time"] - df_durations["arrival_time"]

    df_mz_trips = df_mz_trips.merge(df_durations[["person_id", "trip_id", "activity_duration"]], on = ["person_id", "trip_id"], how = "left")

    # Fix the activity duration for the last activity of the day
    df_mz_trips.loc[df_mz_trips["activity_duration"].isna(), "activity_duration"] = [30*3600-arr_time for arr_time in df_mz_trips[df_mz_trips["activity_duration"].isna()]["arrival_time"]]

    # Parking cost
    df_mz_stages = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")

    df_cost = pd.DataFrame(df_mz_stages[["HHNR", "WEGNR", "f51330"]], copy = True)
    df_cost.columns = ["person_id", "trip_id", "parking_cost"]
    df_cost["parking_cost"] = np.maximum(0, df_cost["parking_cost"])
    df_cost = df_cost.groupby(["person_id", "trip_id"]).sum().reset_index()

    df_mz_trips = pd.merge(df_mz_trips, df_cost, on = ["person_id", "trip_id"], how = "left")
    assert(not np.any(np.isnan(df_mz_trips["parking_cost"])))

    # Identify and fix activity chains for the individuals with consecutive "work" or "education" activities
    df_act_chains = df_mz_trips.copy()
    df_act_chains = df_act_chains.sort_values(["person_id", "trip_id"])

    def build_activity_chain(trips):
        parts = [str(trips.iloc[0]["origin_purpose"])]
        parts += trips["purpose"].astype(str).tolist()
        parts = [p for p in parts if p not in ("nan", "None", "")]

        return "-".join(parts)

    activity_chains = (
        df_act_chains
        .groupby("person_id")
        .apply(build_activity_chain)
        .reset_index(name="activity_chain")
    )

    # Identify people with multiple "work" or "education" in a row - excluding work_secondary
    def has_consecutive_work_or_edu(chain):
        activities = chain.split("-")
        i = 0
        while i < len(activities) -1:
            if activities[i] in ["work", "education", "work_main", "education_main"] and activities[i+1] in ["work", "education", "work_main", "education_main"]:
                if activities[i] == activities[i+1] or activities[i].split("_")[0] == activities[i+1].split("_")[0]:
                    return True
                else:
                    i = i+1
            else:
                i = i+1
        return False

    activity_chains["has_consecutive_work_or_edu"] = activity_chains["activity_chain"].apply(has_consecutive_work_or_edu)
    have_consecutive_work_or_edu                   = activity_chains[activity_chains["has_consecutive_work_or_edu"]]["person_id"].values

    if len(have_consecutive_work_or_edu) > 0:
        logger.info(f"INFO fixing consecutive work and education activities for {len(have_consecutive_work_or_edu)} agents.")

        def consecutive_work_or_edu_indices(chain):
            activities = chain.split("-")
            indices = []
            i = 0
            while i < len(activities) - 1:
                if activities[i] in ["work", "education", "work_main", "education_main"] and activities[i+1] in ["work", "education", "work_main", "education_main"]:
                    if activities[i] == activities[i+1] or activities[i].split("_")[0] == activities[i+1].split("_")[0]:
                        start = i
                        while i + 1 < len(activities) and activities[i+1] in ["work", "education",  "work_main", "education_main"]:
                            i += 1
                        end = i
                        indices.append(range(start, end + 1))
                    else:
                        i = i+1
                else:
                    i += 1
            return indices
        
        consecutive_actchains = activity_chains[activity_chains["has_consecutive_work_or_edu"]]
        consecutive_actchains["consecutive_indices"] = consecutive_actchains["activity_chain"].apply(consecutive_work_or_edu_indices)

        df_mz_trips_no_issues = df_mz_trips[~df_mz_trips["person_id"].isin(have_consecutive_work_or_edu)].copy()
        fixed_df_mz_trips = []

        for person_id in have_consecutive_work_or_edu:
            df_person    = df_mz_trips[df_mz_trips["person_id"]==person_id].sort_values(by="trip_id")
            trip_indices = consecutive_actchains[consecutive_actchains["person_id"]==person_id]["consecutive_indices"].values
            
            for group in trip_indices:
                flat_group = [i for r in group for i in r]
                trips = df_person[df_person["trip_id"].isin(flat_group)]
                df_person.loc[df_person["trip_id"].isin(flat_group), "purpose"] = df_person["purpose"] + "_secondary"
        
                if not trips.empty:
                    if 0 in flat_group:
                        # Handle special case with trip_id == 0 (initial activity)
                        # Get departure time of trip_id 1 as proxy for activity duration before first trip
                        trip1 = df_person[df_person["trip_id"] == 1]
                        initial_duration = trip1["departure_time"].values[0] if not trip1.empty else 0
            
                        # Get all trips with valid activity_duration
                        activity_durations = trips.set_index("trip_id")["activity_duration"].copy()
                        activity_durations.loc[0] = initial_duration
            
                        # Find max
                        trip_with_max = activity_durations.idxmax()
            
                        if trip_with_max == 0:
                            # Initial activity is the main one
                            df_person.loc[df_person["trip_id"] == 1, "origin_purpose"] = \
                                df_person.loc[df_person["trip_id"] == 1, "origin_purpose"] + "_main"
                            
                        else:
                            # Usual case
                            df_person.loc[df_person["trip_id"] == trip_with_max, "purpose"] = \
                                df_person.loc[df_person["trip_id"] == trip_with_max, "purpose"].replace("_secondary", "_main")
                    else:
                        idx_max = trips["activity_duration"].idxmax()
                    
                        # Update that one to _main
                        df_person.loc[idx_max, "purpose"] = df_person.loc[idx_max, "purpose"].replace("_secondary", "_main")
        
            df_person = df_person.sort_values(by="trip_id")
            df_person.loc[df_person["origin_purpose"]=="work_main_main", "origin_purpose"] = "work_main"
            df_person.loc[df_person["purpose"]=="work_main_main", "purpose"]               = "work_main"
            df_person.loc[df_person["purpose"]=="work_main_secondary", "purpose"]          = "work_secondary"
            df_person.loc[df_person["purpose"]=="education_main_secondary", "purpose"]     = "education_secondary"
            df_person.loc[df_person["purpose"]=="education_main_main", "purpose"]          = "education_main"
            
            shifted_purpose = df_person["purpose"].shift(1)
        
            mask = df_person["trip_id"] != 1
            df_person.loc[mask, "origin_purpose"] = shifted_purpose.loc[mask].values
        
            if df_person.loc[df_person["trip_id"]==1]["origin_purpose"].values[0] == df_person.loc[df_person["trip_id"]==1]["purpose"].values[0]:
                df_person.loc[df_person["trip_id"] == 1, "origin_purpose"] = df_person.loc[df_person["trip_id"]==1]["origin_purpose"].replace("main", "secondary")
        
        
            fixed_df_mz_trips.append(df_person)
        
        fixed_df_mz_trips = pd.concat(fixed_df_mz_trips)
        df_mz_trips_fixed = pd.concat([df_mz_trips_no_issues, fixed_df_mz_trips])
        df_mz_trips_fixed = df_mz_trips_fixed.sort_values(by=["person_id", "trip_id"])
        
        assert len(df_mz_trips) == len(df_mz_trips_fixed)

        df_mz_trips = df_mz_trips_fixed
        del df_mz_trips_fixed

        df_mz_trips["purpose"] = df_mz_trips["purpose"].replace("work_main", "work")
        df_mz_trips["origin_purpose"] = df_mz_trips["origin_purpose"].replace("work_main", "work")
        df_mz_trips["purpose"] = df_mz_trips["purpose"].replace("education_main", "education")
        df_mz_trips["origin_purpose"] = df_mz_trips["origin_purpose"].replace("education_main", "education")
        
        # Check that everything is correct now
        df_act_chains = df_mz_trips.copy()
        df_act_chains = df_act_chains.sort_values(["person_id", "trip_id"])
        
        # Apply it per person
        activity_chains = (
            df_act_chains
            .drop(columns="person_id")  
            .groupby(df_act_chains["person_id"])
            .apply(build_activity_chain)
            .reset_index(name="activity_chain") 
        )
        activity_chains.columns = ["person_id", "activity_chain"]

        #assert len(list(activity_chains[activity_chains["activity_chain"].str.contains("work-work-")]["activity_chain"].unique())) == 0
        #assert len(list(activity_chains[activity_chains["activity_chain"].str.contains("education-education-")]["activity_chain"].unique())) == 0

    # Match observed trip coordinates with reported home coordinates for home trips.
    for col in ["origin_x", "origin_y", "destination_x", "destination_y", "home_x", "home_y"]:
        df_mz_trips[col] = df_mz_trips[col].astype(int)

    print(len(df_mz_trips))

    df_purpose_home        = df_mz_trips[df_mz_trips["purpose"]=="home"][["person_id", "destination_x", "destination_y", "home_x", "home_y"]]
    df_origin_purpose_home = df_mz_trips[df_mz_trips["origin_purpose"]=="home"][["person_id", "origin_x", "origin_y", "home_x", "home_y"]]

    df_purpose_home.columns        = ["person_id", "x", "y", "home_x", "home_y"]
    df_origin_purpose_home.columns = ["person_id", "x", "y", "home_x", "home_y"]

    # Finding all home coordinates reported for each agent
    df_home_coordinates = pd.concat([df_purpose_home.drop_duplicates(), df_origin_purpose_home.drop_duplicates()]).sort_values(by="person_id", ascending=True).drop_duplicates()

    df_home_coordinates["home_location_count"] = df_home_coordinates.groupby("person_id")["person_id"].transform("count")
    df_home_coordinates["home_id"]             = df_home_coordinates.groupby("person_id").cumcount() + 1
    df_home_coordinates["is_reported_home"]    = (df_home_coordinates["x"] == df_home_coordinates["home_x"]) & (df_home_coordinates["y"] == df_home_coordinates["home_y"])

    person_summary = df_home_coordinates.groupby("person_id").agg(
        home_location_count=("person_id", "count"),
        has_home_match=("is_reported_home", "any")
    ).reset_index()

    df_new_list = []

    # 1st case: one home location found, corresponds to reported home
    cond1  = (person_summary["home_location_count"] == 1) & (person_summary["has_home_match"])
    share1 = cond1.sum() / len(person_summary) * 100
    logger.info(f"  INFO for {round(share1, 2)}% of the agents, only one home location was found and it corresponds to the reported home location.")
    logger.info(f"    Nothing to do for them!")

    df_home_1stcase     = df_home_coordinates[(df_home_coordinates["home_location_count"]==1) & (df_home_coordinates["is_reported_home"])]
    df_home_1stcase_ids = df_home_1stcase["person_id"].values
    df_new_list.append(df_mz_trips[df_mz_trips["person_id"].isin(df_home_1stcase_ids)])

    # 2nd case: one home location found, does not correspond to the reported home
    cond2  = (person_summary["home_location_count"] == 1) & (~person_summary["has_home_match"])
    share2 = cond2.sum() / len(person_summary) * 100
    logger.info(f"  INFO for {round(share2, 2)}% of the agents, only one home location was found and it does not correspond to the reported home location.")
    logger.info(f"    Check the origin_purpose of the initial trip and adjust the purpose to home if needed.")

    df_home_2ndcase     = df_home_coordinates[(df_home_coordinates["home_location_count"]==1) & (~df_home_coordinates["is_reported_home"])]
    df_home_2ndcase_ids = df_home_2ndcase["person_id"]

    cpt = 0
    for person_id in df_home_2ndcase_ids:
        df_person  = df_mz_trips[df_mz_trips["person_id"]==person_id]
        home_coord = df_home_2ndcase[df_home_2ndcase["person_id"]==person_id][["x", "y"]]
        x          = home_coord["x"].values[0]
        y          = home_coord["y"].values[0]

        trip1 = df_person[df_person["trip_id"] == 1]
        if trip1.empty:
            continue
        ox = trip1["origin_x"].iloc[0]
        oy = trip1["origin_y"].iloc[0]

        if pd.notna(ox) and pd.notna(oy) and abs(ox - x) < 1e-6 and abs(oy - y) < 1e-6:
            cpt += 1
            df_person.loc[df_person["trip_id"] == 1, "origin_purpose"] = "home"
        df_new_list.append(df_person)

    logger.info(f"    INFO fixed origin purpose to home for {cpt} activity chains.")

    # 3rd case: multiple home locations found, one corresponds to the reported home.
    cond3  = (person_summary["home_location_count"] > 1) & (person_summary["has_home_match"])
    share3 = cond3.sum() / len(person_summary) * 100
    print(f"  INFO for {round(share3, 2)}% of the agents, multiple home locations were found. One corresponds to the reported home location.")
    print(f"    For these agents, create home_secondary activities.")

    df_home_3rdcase     = person_summary[cond3]
    df_home_3rdcase_ids = df_home_3rdcase["person_id"]

    for person_id in df_home_3rdcase_ids:
        df_person       = df_mz_trips[df_mz_trips["person_id"]==person_id]
        main_home_coord = df_home_coordinates[(df_home_coordinates["person_id"]==person_id) & (df_home_coordinates["is_reported_home"])][["x", "y"]]
        assert len(main_home_coord["x"]) == 1
        other_home_coord = df_home_coordinates[(df_home_coordinates["person_id"]==person_id) & (~df_home_coordinates["is_reported_home"])]
        other_home_coord["home_id"] = "home_secondary" + other_home_coord["home_id"].astype(str)

        secondary_home_coords = {
            (round(row["x"]), round(row["y"])): row["home_id"]
            for _, row in other_home_coord.iterrows()
        }

        def match_secondary_home(x, y):
            return secondary_home_coords.get((int(x), int(y)), None)
        
        df_person["purpose"] = df_person.apply(
            lambda row: match_secondary_home(row["destination_x"], row["destination_y"]) 
                        if match_secondary_home(row["destination_x"], row["destination_y"]) and row["purpose"]=="home" 
                        else row["purpose"],
            axis=1
        )

        df_person["origin_purpose"] = df_person.apply(
            lambda row: match_secondary_home(row["origin_x"], row["origin_y"]) 
                        if match_secondary_home(row["origin_x"], row["origin_y"]) and row["origin_purpose"]=="home" 
                        else row["origin_purpose"],
            axis=1
        )    
        df_new_list.append(df_person) 


    # 4th case: multiple home locations found, none corresponds to the reported home.
    # 0.09% of the cases
    cond4  = (person_summary["home_location_count"] > 1) & (~person_summary["has_home_match"])
    share4 = cond4.sum() / len(person_summary) * 100
    logger.info(f"  INFO for {round(share4, 2)}% of the agents, multiple home locations were found. None of them corresponds to the reported home location.")
    logger.info(f"    For these agents, create home_secondary activities after identifying the main home location.")

    df_home_4thcase     = person_summary[cond4]
    df_home_4thcase_ids = df_home_4thcase["person_id"]

    for person_id in df_home_4thcase_ids:
        df_person             = df_mz_trips[df_mz_trips["person_id"]==person_id]
        home_coord            = df_home_coordinates[(df_home_coordinates["person_id"]==person_id) & (~df_home_coordinates["is_reported_home"])]
        home_coord["home_id"] = "home_secondary" + home_coord["home_id"].astype(str)

        secondary_home_coords = {
            (round(row["x"]), round(row["y"])): row["home_id"]
            for _, row in home_coord.iterrows()
        }

        def match_secondary_home(x, y):
            return secondary_home_coords.get((int(x), int(y)), None)
        
        df_person["purpose"] = df_person.apply(
            lambda row: match_secondary_home(row["destination_x"], row["destination_y"]) 
                        if match_secondary_home(row["destination_x"], row["destination_y"]) 
                        else row["purpose"],
            axis=1
        )

        df_person["origin_purpose"] = df_person.apply(
            lambda row: match_secondary_home(row["origin_x"], row["origin_y"]) 
                        if match_secondary_home(row["origin_x"], row["origin_y"]) 
                        else row["origin_purpose"],
            axis=1
        ) 

        home_locs = []
        for col_x, col_y in [("origin_x", "origin_y"), ("destination_x", "destination_y")]:
            home_rows = df_person[df_person["purpose"].str.startswith("home") | df_person["origin_purpose"].str.startswith("home")]
            home_locs.extend(list(zip(home_rows[col_x], home_rows[col_y])))   

        unique_home_locs = set(home_locs)
        home_time = {}
        for x, y in unique_home_locs:
            mask = ((df_person["origin_x"] == x) & (df_person["origin_y"] == y)) | \
                ((df_person["destination_x"] == x) & (df_person["destination_y"] == y))
            home_time[(x, y)] = df_person.loc[mask, "activity_duration"].sum()

        first_trip = df_person.iloc[0]
        if first_trip["origin_purpose"].startswith("home"):
            loc = (first_trip["origin_x"], first_trip["origin_y"])
            home_time[loc] += first_trip["activity_duration"]

        main_home = max(home_time, key=home_time.get)

        home_label_map = {}
        for idx, loc in enumerate(sorted(home_time, key=home_time.get, reverse=True)):
            if loc == main_home:
                home_label_map[loc] = "home"
            else:
                home_label_map[loc] = f"home_secondary{idx}"

        def get_home_label(row):
            if "home" in row["purpose"]:
                return home_label_map.get((row["destination_x"], row["destination_y"]), row["purpose"])
            return row["purpose"]
        
        def get_home_label_origin(row):
            if "home" in row["origin_purpose"]:
                return home_label_map.get((row["origin_x"], row["origin_y"]), row["origin_purpose"])
            return row["origin_purpose"]
        
        df_person["purpose"]        = df_person.apply(get_home_label, axis=1)
        df_person["origin_purpose"] = df_person.apply(get_home_label_origin, axis=1)

        df_new_list.append(df_person) 
        
    df_mz_trips = pd.concat(df_new_list, ignore_index=True)

    df_purpose_home        = df_mz_trips[df_mz_trips["purpose"].str.contains("home")][["person_id", "destination_x", "destination_y"]]
    df_origin_purpose_home = df_mz_trips[df_mz_trips["origin_purpose"].str.contains("home", na=False)][["person_id", "origin_x", "origin_y"]]

    df_purpose_home.columns        = ["person_id", "x", "y"]
    df_origin_purpose_home.columns = ["person_id", "x", "y"]

    # Finding all home coordinates reported for each agent
    df_home_coordinates = pd.concat([df_purpose_home.drop_duplicates(), df_origin_purpose_home.drop_duplicates()]).sort_values(by="person_id", ascending=True).drop_duplicates()

    df_home_coordinates["home_location_count"] = df_home_coordinates.groupby("person_id")["person_id"].transform("count")
    df_home_coordinates["home_id"]             = df_home_coordinates.groupby("person_id").cumcount() + 1
    df_home_coordinates["home_id"]             = df_home_coordinates["home_id"].astype(str)

    def merge_close_home_locations(df, threshold=30):
        merged_rows = []
        coord_map   = {}  # Maps (person_id, old_x, old_y) -> (person_id, kept_x, kept_y)
        label_map   = {}  # Maps (person_id, old_x, old_y) -> home_id label

        for person_id, group in df.groupby("person_id"):
            coords  = group[["x", "y"]].values
            keep    = np.ones(len(coords), dtype=bool)
            rep_idx = np.arange(len(coords))  # index of representative for each point

            for i in range(len(coords)):
                if not keep[i]:
                    continue
                for j in range(i+1, len(coords)):
                    dist = np.sqrt((coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2)
                    if dist <= threshold:
                        keep[j] = False
                        rep_idx[j] = i 

            merged = group.iloc[keep].copy()
            merged["home_location_count"] = keep.sum()
            # Assign home_id labels: first kept is "home", others are "home_secondaryX"
            merged = merged.reset_index(drop=True)
            for idx, row in merged.iterrows():
                if idx == 0:
                    home_id = "home"
                else:
                    home_id = f"home_secondary{idx}"
                merged.at[idx, "home_id"] = home_id

            merged_rows.append(merged)

            # Build mapping for all original points to their representative and label
            for pos, row in enumerate(group.itertuples(index=False)):
                rep = rep_idx[pos]
                rep_x, rep_y = coords[rep]
                rep_label = merged.loc[merged["x"].eq(rep_x) & merged["y"].eq(rep_y), "home_id"].values[0]
                coord_map[(person_id, row.x, row.y)] = (person_id, rep_x, rep_y)
                label_map[(person_id, row.x, row.y)] = rep_label

        merged_df = pd.concat(merged_rows, ignore_index=True)
        return merged_df, coord_map, label_map

    single_homes   = df_home_coordinates[df_home_coordinates["home_location_count"]==1]
    multiple_homes = df_home_coordinates[df_home_coordinates["home_location_count"]>1]

    multiple_homes, coord_map, label_map = merge_close_home_locations(multiple_homes)

    df_home_coordinates = pd.concat([single_homes, multiple_homes]).sort_values(by="person_id")

    def map_home_label_and_coords(row, x_col, y_col, purpose_col):
        key = (row["person_id"], row[x_col], row[y_col])
        purpose = row[purpose_col]
        purpose_str = "" if pd.isna(purpose) else str(purpose)

        # Only adjust if this is a home purpose and the key exists in the mapping
        if purpose_str.startswith("home") and key in label_map:
            # Update purpose
            new_purpose = label_map[key]
            # Update coordinates
            _, new_x, new_y = coord_map[key]
            row[x_col] = new_x
            row[y_col] = new_y
            return pd.Series([new_purpose, new_x, new_y])
        
        # No change
        return pd.Series([row[purpose_col], row[x_col], row[y_col]])

    # Update purposes for all trips
    df_mz_trips[["purpose", "destination_x", "destination_y"]] = df_mz_trips.apply(
        lambda row: map_home_label_and_coords(row, "destination_x", "destination_y", "purpose"), axis=1
    )

    df_mz_trips[["origin_purpose", "origin_x", "origin_y"]] = df_mz_trips.apply(
        lambda row: map_home_label_and_coords(row, "origin_x", "origin_y", "origin_purpose"), axis=1
    )

    # Now fix the home->home trips
    loops = df_mz_trips[df_mz_trips["origin_purpose"].str.startswith("home") & df_mz_trips["purpose"].str.startswith("home") & (df_mz_trips["origin_purpose"]==df_mz_trips["purpose"])].copy()

    stages  = pd.read_csv("%s/microcensus/etappen.csv" % data_path, encoding = "latin1")
    stages  = stages[["HHNR", "WEGNR", "WP", "ETNR", "DMOD", "f51300", "f52900", "f52950", "f51100time", "f51400time", "S_X_CH1903", "S_Y_CH1903",
                             "Z_X_CH1903", "Z_Y_CH1903"]]
    stages.columns = ["person_id", "trip_id", "weight_person", "stage_id", "module", "mode", "purpose", "origin_purpose",
                            "departure_time", "arrival_time", "origin_x", "origin_y", "destination_x", "destination_y"]

    loops["person_trip_id"]  = loops["person_id"].astype(str) + "_" + loops["trip_id"].astype(str)
    stages["person_trip_id"] = stages["person_id"].astype(str) + "_" + stages["trip_id"].astype(str)

    stages = stages[stages["person_trip_id"].isin(loops["person_trip_id"])]

    stages["stage_count"] = stages.groupby("person_trip_id")["stage_id"].transform("count")
    multi_stages  = stages[stages["stage_count"]>1]
    single_stages = stages[stages["stage_count"]==1]

    loops_singlestage = loops[loops["person_trip_id"].isin(single_stages["person_trip_id"])]
    loops_singlestage["mode"] = loops_singlestage["mode"] + "_loop"

    loops_multistage = loops[loops["person_trip_id"].isin(multi_stages["person_trip_id"])]

    def process_useful_stages(multi_stages, loops_multistage):

        logger.info("INFO starting to fix home to home trips")

        mode_map = {
            -99: "unknown",
            -98: "unknown",
            95: "other",
            1: "walk",
            2: "bike",
            3: "motorcycle",
            4: "motorcycle",
            5: "motorcycle",
            6: "motorcycle_passenger",
            7: "car",
            8: "car_passenger",
            9: "pt",
            10: "pt",
            11: "pt",
            12: "pt",
            13: "car",
            14: "bike",
            15: "truck",
            16: "pt",
            17: "pt",
            18: "pt",
            19: "other",
            20: "bike",
            21: "bike"
        }

        purpose_map = {
            1: "other", # connection
            2: "work",
            3: "education",
            4: "shop",
            5: "other",
            6: "work_secondary",
            7: "work_secondary",
            8: "leisure",
            9: "other", # accompanying kids
            10: "other", # accompanying others
            11: "home",
            12: "other",
            13: "other",
            -99: "other",
            -98: "unknown",
            -97: "unknown"
        }

        multi_stages.replace({"purpose": purpose_map, "origin_purpose": purpose_map, "mode": mode_map}, inplace=True)

        for attribute in ["origin", "destination"]:
            coords = multi_stages[[attribute + "_x", attribute + "_y"]].values
            transformer = pyproj.Transformer.from_crs("epsg:21781", "epsg:2056")
            x, y = transformer.transform(coords[:, 0], coords[:, 1])
            multi_stages.loc[:, attribute + "_x"] = [int(xelem) for xelem in x]
            multi_stages.loc[:, attribute + "_y"] = [int(yelem) for yelem in y]
            
        def time_to_seconds(t):
            if t == "24:00:00":
                return 24 * 3600  # 86400 seconds
            h, m, s = map(int, t.split(":"))
            return h * 3600 + m * 60 + s

        cpt_ok  = 0
        cpt_all = 0
        for person_trip_id, group in multi_stages.groupby("person_trip_id"):
            cpt_all += 1

            origin_counts = group[["origin_x", "origin_y"]].nunique()
            origin_counts["multiple_origins"] = (origin_counts["origin_x"] > 1) | (origin_counts["origin_y"] > 1)

            destination_counts = group[["destination_x", "destination_y"]].nunique()
            destination_counts["multiple_destinations"] = (destination_counts["destination_x"] > 1) | (destination_counts["destination_y"] > 1)

            if origin_counts["multiple_origins"] | destination_counts["multiple_destinations"]:
                cpt_ok += 1
                group["arrival_time"]   = [time_to_seconds(t) for t in group["arrival_time"]]
                group["departure_time"] = [time_to_seconds(t) for t in group["departure_time"]]
                group = group.sort_values(["person_trip_id", "stage_id"])
                group["next_departure_time"] = group.groupby("person_trip_id")["departure_time"].shift(-1)
                group["activity_duration"] = (group["next_departure_time"] - group["arrival_time"])

                valid = group["activity_duration"].notna()
                idx_longest_time =  group.loc[valid, "activity_duration"].idxmax()            

                unique_list = list(set(group["purpose"]).union(set(group["origin_purpose"])))
                unique_list = [p for p in unique_list if p not in ["pseudostage", "connection"]]

                x, y      = group.loc[idx_longest_time, ["destination_x", "destination_y"]]
                arrival   = group.loc[idx_longest_time, "arrival_time"]
                departure = group.loc[idx_longest_time + 1, "departure_time"]
                purpose   = group.loc[idx_longest_time, "purpose"]

                base_index = loops_multistage[loops_multistage["person_trip_id"]==person_trip_id].index

                new_row = loops_multistage.loc[base_index].copy()

                loops_multistage.loc[base_index, "destination_x"] = int(x)
                loops_multistage.loc[base_index, "destination_y"] = int(y)
                loops_multistage.loc[base_index, "arrival_time"]  = int(arrival)
                loops_multistage.loc[base_index, "purpose"]       = purpose

                new_row["origin_x"]       = int(x)
                new_row["origin_y"]       = int(y)
                new_row["departure_time"] = int(departure)
                new_row["origin_purpose"] = purpose

                loops_multistage = pd.concat(
                    [loops_multistage, pd.DataFrame(new_row)],
                    ignore_index=True
                )
                loops_multistage = loops_multistage.sort_values(by=["person_id", "trip_id", "departure_time"], ascending = True)
                
            else:
                loops_multistage.loc[loops_multistage["person_trip_id"]==person_trip_id, "mode"] =  loops_multistage[loops_multistage["person_trip_id"]==person_trip_id]["mode"] + "_loop"

        logger.info("  done!")
        return loops_multistage


    for col in ["origin_x", "origin_y", "destination_x", "destination_y"]:
        multi_stages[col] = multi_stages[col].astype(int)

    for col in ["origin_x", "origin_y", "destination_x", "destination_y",  "departure_time", "arrival_time"]:
        loops_multistage[col] = loops_multistage[col].astype(int)

    loops_multistage = process_useful_stages(multi_stages.copy(), loops_multistage.copy())

    loops    = pd.concat([loops_singlestage, loops_multistage])
    nonloops = df_mz_trips[~df_mz_trips["origin_purpose"].str.startswith("home", na=False) | ~df_mz_trips["purpose"].str.startswith("home", na=False) | ~(df_mz_trips["origin_purpose"]==df_mz_trips["purpose"])].copy()

    df_mz_trips = pd.concat([loops, nonloops])
    df_mz_trips = df_mz_trips.sort_values(by=["person_id", "trip_id", "departure_time"])
    df_mz_trips["trip_id"] = df_mz_trips.groupby(["person_id"]).cumcount() + 1

    df_mz_trips[["origin_purpose", "purpose"]] = df_mz_trips[["origin_purpose", "purpose"]].replace(["home_secondary1", "home_secondary2"], "home_secondary")

    # Network distance
    df_mz_trips["network_distance"] = df_mz_trips["w_rdist"] * 1000.0

    # Re-compute crowfly distance
    df_mz_trips.loc[:, "crowfly_distance"] = np.sqrt(
        (df_mz_trips["origin_x"] - df_mz_trips["destination_x"])**2 +
        (df_mz_trips["origin_y"] - df_mz_trips["destination_y"])**2)
    
    # Identify activity chains completely outside of Switzerland
    swiss_border = context.stage("data.spatial.swiss_border").copy().unary_union

    origins = gpd.GeoDataFrame(df_mz_trips,
                               geometry = gpd.points_from_xy(df_mz_trips["origin_x"], df_mz_trips["origin_y"]),
                               crs="epsg:2056")

    destinations = gpd.GeoDataFrame(df_mz_trips,
                                    geometry = gpd.points_from_xy(df_mz_trips["destination_x"], df_mz_trips["destination_y"]),
                                    crs = "epsg:2056")
    
    df_mz_trips["origin_in_ch"] = origins.geometry.within(swiss_border)
    df_mz_trips["dest_in_ch"]   = destinations.geometry.within(swiss_border)

    df_mz_trips["trip_outside_ch"] = (~df_mz_trips["origin_in_ch"]) & (~df_mz_trips["dest_in_ch"])

    persons_all_outside = df_mz_trips.groupby("person_id")["trip_outside_ch"].all()
    persons_outside_ch  = persons_all_outside[persons_all_outside].index

    df_mz_trips["trip_crossing_border"] = ((df_mz_trips["origin_in_ch"]) & (~df_mz_trips["dest_in_ch"])) | ((~df_mz_trips["origin_in_ch"]) & (df_mz_trips["dest_in_ch"]))
    persons_crossing_border             = df_mz_trips.groupby("person_id")["trip_crossing_border"].any()
    persons_crossing_the_border         = persons_crossing_border[persons_crossing_border].index

    print(len(df_mz_trips))

    df_mz_trips["geometry"] = df_mz_trips.apply(
        lambda r: LineString([
            (r["origin_x"], r["origin_y"]),
            (r["destination_x"], r["destination_y"])
        ]),
        axis=1
    )

    gdf = gpd.GeoDataFrame(df_mz_trips, geometry="geometry", crs="EPSG:2056")
    
    output_path = context.config("output_path")
    gdf.to_file(f"{output_path}/wegeinland_processed.gpkg", driver = "gpkg")

    return df_mz_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "origin_purpose", "purpose", 
        "origin_x", "origin_y", "destination_x", "destination_y", 
        "activity_duration", "crowfly_distance", "parking_cost", "network_distance",
        "mode_detailed",
        "trip_outside_ch"
    ]], filterout_ids, persons_outside_ch, persons_crossing_the_border
