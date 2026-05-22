import pandas as pd
import numpy as np
import geopandas as gpd
import os
from shapely import wkt
from shapely.geometry import Point
import logging

logger = logging.getLogger(__name__)

PERSON_FIELDS = ["person_id", "age", "car_availability", "employed", "driving_license", "sex", 
                 "home_x", "home_y",
                 "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", "subscriptions_strecke",
                 "household_id", "is_car_passenger", 
                 "statpop_person_id", "statpop_household_id", "mz_person_id", "mz_head_id", 
                 "has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip",
                 "income_class",
                 "number_of_cars_class", "number_of_bikes_class"]


def configure(context):
    context.config("include_external_population", default = False)

    if context.config("include_external_population"):
        context.config("external_population_folder")
        context.stage("data.constants")
        context.stage("synthesis.population.destinations")
        context.stage("synthesis.population.enriched")
        context.config("fr_sample_rate", default = 1.0)
        context.config("input_downsampling")


def execute(context):
    if not context.config("include_external_population"):
        return

    folder = context.config("external_population_folder")

    assert any(f.endswith("_persons.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_households.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_homes.gpkg") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_trips.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"
    assert any(f.endswith("_activities.csv") for f in os.listdir(folder)), f"No *_persons.csv file found in {folder}"

    persons_file = next(f for f in os.listdir(folder) if f.endswith("_persons.csv"))
    persons      = pd.read_csv(os.path.join(folder, persons_file), sep = ";")[["person_id", "household_id", "age", "sex", "employed", "has_driving_license", "has_pt_subscription", "census_person_id", "hts_id"]]

    households_file = next(f for f in os.listdir(folder) if f.endswith("_households.csv"))
    households      = pd.read_csv(os.path.join(folder, households_file), sep = ";")[["household_id", "car_availability", "bike_availability", "number_of_vehicles", "number_of_bikes", "income"]]

    trips_file = next(f for f in os.listdir(folder) if f.endswith("_trips.csv"))
    trips      = pd.read_csv(os.path.join(folder, trips_file), sep = ";")[["person_id", "preceding_activity_index", "mode"]]

    acts_file = next(f for f in os.listdir(folder) if f.endswith("_activities.csv"))
    acts      = pd.read_csv(os.path.join(folder, acts_file), sep = ";")[["person_id", "activity_index", "start_time", "end_time",
                                                                          "is_first", "is_last", "purpose", "location_id", "geometry"]]
    acts["geometry"] = acts["geometry"].apply(wkt.loads)
    acts = gpd.GeoDataFrame(acts, geometry="geometry", crs="EPSG:2154")
    acts = acts.to_crs("EPSG:2056")

    acts["destination_x"] = acts["geometry"].apply(lambda g: g.x)
    acts["destination_y"] = acts["geometry"].apply(lambda g: g.y)

    vehicles_file = next(f for f in os.listdir(folder) if f.endswith("_vehicles.csv"))
    vehicles      = pd.read_csv(os.path.join(folder, vehicles_file), sep = ";")
    
    homes_file = next(f for f in os.listdir(folder) if f.endswith("_homes.gpkg"))
    homes      = gpd.read_file(os.path.join(folder, homes_file))[["household_id", "geometry"]]
    homes.crs  = "EPSG:2154"
    homes      = homes.to_crs("EPSG:2056")

    homes["home_x"] = homes.geometry.x
    homes["home_y"] = homes.geometry.y

    households = households.merge(homes[["household_id", "home_x", "home_y"]], on = "household_id", how = "left")
    households.loc[:, "car_availability"]  = households["car_availability"].map({"none": "never", "all": "always", "some": "always"})
    households.loc[:, "bike_availability"] = households["bike_availability"].map({"none": "never", "all": "always", "some": "always"})

    bins   = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, float("inf")]
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    households["income_class"] = pd.cut(households["income"], bins=bins, labels=labels, right=True)
    households["number_of_cars_class"]  = households["number_of_vehicles"]
    households["number_of_bikes_class"] = households["number_of_bikes"]

    persons = persons.merge(households, on = "household_id", how = "left")

    car_passenger_ids = set(trips.loc[trips["mode"] == "car_passenger", "person_id"])
    persons["is_car_passenger"] = persons["person_id"].isin(car_passenger_ids)

    persons["statpop_person_id"]    = persons["census_person_id"]
    persons["statpop_household_id"] = persons["census_person_id"]

    persons["mz_person_id"] = persons["hts_id"]
    persons["mz_head_id"]   = persons["hts_id"]

    persons["driving_license"]       = persons["has_driving_license"]
    persons["subscriptions_ga"]      = False
    persons["subscriptions_halbtax"] = False
    persons["subscriptions_strecke"] = False
    persons["subscriptions_verbund"] = persons["has_pt_subscription"]

    for column in ["has_walk_loop_trip", "has_car_loop_trip", "has_car_passenger_loop_trip", "has_pt_loop_trip", "has_bike_loop_trip"]:
        persons[column] = False

    persons = persons[PERSON_FIELDS]

    acts.loc[acts["is_first"], "start_time"] = 0
    acts.loc[acts["is_last"], "end_time"]    = 30*3600
    acts.loc[:, "duration"] = acts["end_time"] - acts["start_time"]
    acts = acts.merge(trips.rename(columns = {"preceding_activity_index": "activity_index", "mode": "following_mode"}),  on = ["person_id", "activity_index"], how = "left")

    valid_ids = acts.groupby("person_id")["geometry"].apply(
        lambda g: g.notna().all()
    )
    valid_ids = valid_ids[valid_ids].index

    persons = persons[persons["person_id"].isin(valid_ids)]

    acts["destination_id"]    = acts["location_id"].astype(str).str.split("_").str[-1].astype(int)
    acts["municipality_id"]   = 0
    acts["municipality_type"] = 0

    # Adjust IDS
    id_person_max    = np.max(context.stage("synthesis.population.enriched").copy()["person_id"].values)
    id_household_max = np.max(context.stage("synthesis.population.enriched").copy()["household_id"].values)
    id_person_max    = max(id_person_max, id_household_max)  # just in case person_id and household_id are not on the same scale
    N                = id_person_max + 1

    # 1. Adjust person_id
    persons["new_person_id"] = range(N, N + len(persons), 1)
    person_id_map            = persons.set_index("person_id")["new_person_id"]

    persons["person_id"]    = persons["new_person_id"].values
    persons["household_id"] = persons["new_person_id"].values

    vehicles["owner_id"]    = vehicles["owner_id"].map(person_id_map).fillna(vehicles["owner_id"])
    vehicles["vehicle_id"]   = vehicles["owner_id"].astype(str) + ":" + vehicles["mode"]
    vehicles = vehicles[["owner_id", "vehicle_id", "age", "euro", "mode"]]

    acts["person_id"] = acts["person_id"].map(person_id_map).fillna(acts["person_id"])

    # 2. Destination id

    homes = acts[acts["purpose"] == "home"]

    homes["destination_id"] = ["home" + str(person_id) for person_id in homes["person_id"].values.tolist()]

    home_coords = persons.groupby("person_id")[["home_x", "home_y"]].first()
    homes["destination_x"] = homes["person_id"].map(home_coords["home_x"])
    homes["destination_y"] = homes["person_id"].map(home_coords["home_y"])

    acts_not_home = acts[acts["purpose"]!="home"]
    unique_ids    = acts_not_home["destination_id"].astype(int).unique()
    max_id        = np.max(context.stage("synthesis.population.destinations").copy()["destination_id"].values.tolist())
    correspondence = {old: max_id + i + 1 for i, old in enumerate(unique_ids)}

    acts_not_home["destination_id"] = acts_not_home["destination_id"].map(correspondence)
    facility_locations = acts_not_home.groupby("destination_id")[["destination_x", "destination_y"]].first()
    acts_not_home["destination_x"] = acts_not_home["destination_id"].map(facility_locations["destination_x"])
    acts_not_home["destination_y"] = acts_not_home["destination_id"].map(facility_locations["destination_y"])

    acts = pd.concat([homes, acts_not_home])
    acts = acts.sort_values(by = ["person_id", "activity_index"])

    acts["geometry"] = acts.apply(
        lambda r: Point(r["destination_x"], r["destination_y"]), axis=1
    )

    #acts.to_csv("/cluster/project/cmdp/asallard/theacts.csv", index=False)

    # Fix missing vehicles
    modes = ["car", "car_passenger", "bike"]
    all_persons = persons["person_id"].unique()

    df_required = pd.MultiIndex.from_product(
        [all_persons, modes], names=["owner_id", "mode"]
    ).to_frame(index=False)

    df_required["vehicle_id"] = df_required["owner_id"].astype(str) + ":" + df_required["mode"]

    # Find missing
    existing = set(zip(vehicles["owner_id"], vehicles["mode"]))
    df_missing = df_required[~df_required.apply(
        lambda r: (r["owner_id"], r["mode"]) in existing, axis=1
    )]

    # Add missing vehicles (age=0 or whatever default)
    df_missing["age"]  = 0
    df_missing["euro"] = 6

    vehicles = pd.concat([vehicles, df_missing], ignore_index=True)

    fr_sample_rate = context.config("fr_sample_rate")
    ch_sample_rate = context.config("input_downsampling")
    ratio          = ch_sample_rate / fr_sample_rate

    if ratio > 1:
        logger.warning("The requested sample size for the Swiss population exceeds the sample size used for the generation of the French population. We might find a solution for this at some point but as of now we are keeping the French population unchanged.")

    elif ratio < 1:
        print(f"FR sample rate: {fr_sample_rate}. CH sample rate: {ch_sample_rate}.")
        print(f"Downsampling with a ratio of {round(ratio, 2)}.")

        person_ids  = persons["person_id"].values.tolist()
        sampled_ids = np.random.choice(person_ids, size = int(len(person_ids) * ratio), replace = False).tolist()

        persons    = persons[persons["person_id"].isin(sampled_ids)]
        acts       = acts[acts["person_id"].isin(sampled_ids)]
        vehicles   = vehicles[vehicles["owner_id"].isin(sampled_ids)]

    return persons, acts, vehicles




