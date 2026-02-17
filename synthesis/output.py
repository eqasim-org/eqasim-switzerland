import geopandas as gpd
import pandas as pd
import shapely.geometry as geo
import os, datetime, json
import sqlite3
import math

def configure(context):
    context.stage("synthesis.population.enriched")

    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.trips")

    #context.stage("synthesis.population.spatial.locations")

    context.config("output_path")
    context.config("output_prefix", "switzerland_")

def validate(context):
    output_path = context.config("output_path")

    if not os.path.isdir(output_path):
        raise RuntimeError("Output directory must exist: %s" % output_path)

def clean_gpkg(path):
    '''
    Make GPKG files time and OS independent.

    In GeoPackage metadata:
    - replace last_change date with a placeholder, and
    - round coordinates.

    This allow for comparison of output digests between runs and between OS.
    '''
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for table_name, min_x, min_y, max_x, max_y in cur.execute(
        "SELECT table_name, min_x, min_y, max_x, max_y FROM gpkg_contents"
    ):
        cur.execute(
            "UPDATE gpkg_contents " +
            "SET last_change='2000-01-01T00:00:00Z', min_x=?, min_y=?, max_x=?, max_y=? " +
            "WHERE table_name=?",
            (math.floor(min_x), math.floor(min_y), math.ceil(max_x), math.ceil(max_y), table_name)
        )
    conn.commit()
    conn.close()

def execute(context):
    output_path = context.config("output_path")
    output_prefix = context.config("output_prefix")

    # Prepare households
    df_households = context.stage("synthesis.population.enriched").rename(
        columns = { "income_class": "income"}
    ).drop_duplicates("household_id")
    df_households = df_households[[
        "household_id",
        "number_of_cars_class", "number_of_bikes_class",
        "ovgk",
        "income",
        "statpop_household_id"
    ]]

    df_households.to_csv("%s/%shouseholds.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")

    # Prepare persons
    df_persons = context.stage("synthesis.population.enriched").rename(
        columns = { "driving_license": "has_driving_license" }
    )

    df_persons = df_persons[[
        "person_id", "household_id",
        "age", "employed", "sex",
        "has_driving_license",
        "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund", 
        "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior",
        "subscriptions_other", "subscriptions_ga_class",
        "subscriptions_verbund_class", "subscriptions_strecke_class",
        "statpop_person_id", "mz_person_id", "mz_head_id", "canton_id"
    ]]

    df_persons.to_csv("%s/%spersons.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")

    # Prepare activities
    df_activities = context.stage("synthesis.population.activities")
    df_activities["following_trip_index"] = df_activities["activity_index"]
    
    df_activities = pd.merge(
        df_activities, df_persons[["person_id", "household_id"]], on = "person_id")

    df_activities["preceding_trip_index"] = df_activities["following_trip_index"].shift(1)
    df_activities.loc[df_activities["activity_index"]==0, "preceding_trip_index"] = -1    
    # I add filna(-1) in the next link, because something very rare can happen, which is: one person without any trips (only one activity) 
    # is located in the first row of the dataframe. this person would have Nan value as proceding_trip_index and thus through an error.
    df_activities["preceding_trip_index"] = df_activities["preceding_trip_index"].fillna(-1).astype(int) 

    df_activities = df_activities[[
        "person_id", "household_id", "activity_index",
        "preceding_trip_index", "following_trip_index",
        "purpose", "start_time", "end_time",
        "is_last"
    ]]

    df_activities.to_csv("%s/%sactivities.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")
    print("Finished writing activities!")
    # Prepare trips
    df_trips = context.stage("synthesis.population.trips")
    #df_locat = context.stage("synthesis.population.spatial.locations")


    df_trips["preceding_activity_index"] = df_trips["trip_index"]
    df_trips["following_activity_index"] = df_trips["trip_index"] + 1

    df_trips = df_trips[[
        "person_id", "trip_index",
        "preceding_activity_index", "following_activity_index",
        "departure_time", "arrival_time", "mode",
        "preceding_purpose", "following_purpose"
    ]]

    df_trips.to_csv("%s/%strips.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")
    print("Starting to create activities gpkg data.")
    # Prepare spatial data sets
    #df_locations = context.stage("synthesis.population.spatial.locations")[[
    #    "person_id", "activity_index", "geometry"
    #]]
    # Set a MultiIndex on both (avoids reshuffling all columns)
    #df_locations = df_locations.set_index(["person_id", "activity_index"])
    #df_activities = df_activities.set_index(["person_id", "activity_index"])

    # Fast alignment of the one column you need
    #df_activities["geometry"] = df_locations["geometry"]
    #df_activities = df_activities.reset_index()
    #df_locations = df_locations.reset_index()
    # Write spatial activities
    #df_spatial = gpd.GeoDataFrame(
    #    df_activities,
    #    geometry="geometry",
    #    crs="EPSG:2056"
    #)
    # df_spatial["purpose"] = df_spatial["purpose"].astype(str)
    # path = "%s/%sactivities.gpkg" % (output_path, output_prefix)
    # print("Starting to write activities gpkg data.")
    # df_spatial.to_file(path, driver = "GPKG", engine="pyogrio")
    # print("Finished writing activities gpkg data.")
    # clean_gpkg(path)

    # # Write spatial homes
    # path = "%s/%shomes.gpkg" % (output_path, output_prefix)
    # df_spatial[
    #     df_spatial["purpose"] == "home"
    # ].drop_duplicates("household_id")[[
    #     "household_id", "geometry"
    # ]].to_file(path, driver = "GPKG")
    # clean_gpkg(path)

    # # Write spatial commutes
    # df_spatial = pd.merge(
    #     df_spatial[df_spatial["purpose"] == "home"].drop_duplicates("person_id")[["person_id", "geometry"]].rename(columns = { "geometry": "home_geometry" }),
    #     df_spatial[df_spatial["purpose"] == "work"].drop_duplicates("person_id")[["person_id", "geometry"]].rename(columns = { "geometry": "work_geometry" })
    # )

    # df_spatial["geometry"] = [
    #     geo.LineString(od)
    #     for od in zip(df_spatial["home_geometry"], df_spatial["work_geometry"])
    # ]

    # df_spatial = df_spatial.drop(columns = ["home_geometry", "work_geometry"])
    # path = "%s/%scommutes.gpkg" % (output_path, output_prefix)
    # df_spatial.to_file(path, driver = "GPKG")
    # clean_gpkg(path)

    # # Write spatial trips
    # df_spatial = pd.merge(df_trips, df_locations[[
    #     "person_id", "activity_index", "geometry"
    # ]].rename(columns = {
    #     "activity_index": "preceding_activity_index",
    #     "geometry": "preceding_geometry"
    # }), how = "left", on = ["person_id", "preceding_activity_index"])

    # df_spatial = pd.merge(df_spatial, df_locations[[
    #     "person_id", "activity_index", "geometry"
    # ]].rename(columns = {
    #     "activity_index": "following_activity_index",
    #     "geometry": "following_geometry"
    # }), how = "left", on = ["person_id", "following_activity_index"])

    # df_spatial["geometry"] = [
    #     geo.LineString(od)
    #     for od in zip(df_spatial["preceding_geometry"], df_spatial["following_geometry"])
    # ]

    # df_spatial = df_spatial.drop(columns = ["preceding_geometry", "following_geometry"])

    # df_spatial = gpd.GeoDataFrame(df_spatial, crs = "EPSG:2056")
    # df_spatial["following_purpose"] = df_spatial["following_purpose"].astype(str)
    # df_spatial["preceding_purpose"] = df_spatial["preceding_purpose"].astype(str)
    # df_spatial["mode"] = df_spatial["mode"].astype(str)

    # df_spatial["crowfly_distance"] = df_spatial.geometry.length
    # df_trips = pd.merge(df_trips, df_spatial[["crowfly_distance", "person_id", "following_activity_index"]], how="left", 
    #                     on=["person_id", "following_activity_index"])
    # df_trips = df_trips.drop(columns = ["following_activity_index"])
    # df_trips.to_csv("%s/%strips.csv" % (output_path, output_prefix), sep = ";", index = None, lineterminator = "\n")

    # path = "%s/%strips.gpkg" % (output_path, output_prefix)
    # df_spatial.to_file(path, driver = "GPKG")
    # clean_gpkg(path)


