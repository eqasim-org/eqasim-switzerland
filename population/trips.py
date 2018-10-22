import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.work_locations")
    require.stage("population.education_locations")
    require.stage("data.microcensus.trips")
    require.stage("population.commute")

# TODO: Probably it would make sense to split this up in "trips" and "primary_locations"

def execute(context):
    df_persons = context.stage("population.sociodemographics")

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy = True)
    df_trips["mz_person_id"] = df_trips["person_id"]
    df_trips["mz_x"] = df_trips["destination_x"]
    df_trips["mz_y"] = df_trips["destination_y"]

    df_trips = df_trips[[
        "mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose", "crowfly_distance",
        "mz_x", "mz_y"
    ]]

    df_trips = pd.merge(df_persons[["person_id", "mz_person_id"]], df_trips, how = "inner")

    df_home_location = pd.DataFrame(df_persons[["person_id", "home_x", "home_y"]], copy = True)
    df_home_location.loc[:, "location_id"] = np.nan
    df_home_location.loc[:, "location_purpose"] = "home"
    df_home_location.columns = ["person_id", "location_x", "location_y", "location_id", "location_purpose"]

    df_work_location = pd.DataFrame(context.stage("population.work_locations")[[
        "person_id", "work_x", "work_y", "work_enterprise_id"
    ]], copy = True)
    df_work_location.loc[:, "location_purpose"] = "work"
    df_work_location.columns = ["person_id", "location_x", "location_y", "location_id", "location_purpose"]

    df_education_location = pd.DataFrame(context.stage("population.education_locations")[[
        "person_id", "education_x", "education_y", "education_enterprise_id"
    ]], copy = True)
    df_education_location.loc[:, "location_purpose"] = "education"
    df_education_location.columns = ["person_id", "location_x", "location_y", "location_id", "location_purpose"]

    df_location = pd.concat([df_home_location, df_work_location, df_education_location])

    df_trips = pd.merge(
        df_trips, df_location,
        left_on = ["person_id", "purpose"], right_on = ["person_id", "location_purpose"],
        how = "left"
    )

    # Make sure we reset locations for all non-commute work or education trips
    df_commute = pd.DataFrame(context.stage("population.commute")[["person_id", "commute_trip_id", "commute_purpose", "commute_x", "commute_y"]], copy = True)
    df_commute["mz_person_id"] = df_commute["person_id"]
    del df_commute["person_id"]

    df_trips = pd.merge(
        df_trips, df_commute, how = "left",
        left_on = ["mz_person_id", "trip_id", "purpose"],
        right_on = ["mz_person_id", "commute_trip_id", "commute_purpose"])

    for purpose in ["work", "education"]:
        f = np.isnan(df_trips["commute_trip_id"]) & (df_trips["purpose"] == purpose)
        df_trips.loc[f, "location_x"] = np.nan
        df_trips.loc[f, "location_y"] = np.nan
        df_trips.loc[f, "location_id"] = np.nan

    df_trips.loc[:, "is_commute_trip"] = ~np.isnan(df_trips["commute_trip_id"])
    df_trips.loc[:, "reference_distance"] = df_trips.loc[:, "crowfly_distance"]

    df_trips = df_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose",
        "location_x", "location_y", "location_id", "reference_distance", "is_commute_trip", "mz_x", "mz_y"
    ]]

    # If there are primary activities that are not the commute activity, but
    # are at the same location, we also fix the location already here.
    df_commute_location = df_trips[df_trips["is_commute_trip"]][[
        "person_id", "location_x", "location_y", "location_id", "mz_x", "mz_y", "purpose"
    ]]
    df_commute_location.columns = ["person_id", "commute_location_x", "commute_location_y", "commute_location_id", "commute_x", "commute_y", "commute_purpose"]
    assert(len(df_commute_location) == len(np.unique(df_commute_location["person_id"])))

    df_trips = pd.merge(df_trips, df_commute_location, on = "person_id", how = "left")

    same_mz_coord_as_commute_trip = (
        df_trips["mz_x"] == df_trips["commute_x"]
    ) & (
        df_trips["mz_y"] == df_trips["commute_y"]
    ) & (
        df_trips["purpose"] == df_trips["commute_purpose"]
    )

    df_trips.loc[same_mz_coord_as_commute_trip, "location_x"] = df_trips.loc[same_mz_coord_as_commute_trip, "commute_location_x"]
    df_trips.loc[same_mz_coord_as_commute_trip, "location_y"] = df_trips.loc[same_mz_coord_as_commute_trip, "commute_location_y"]
    df_trips.loc[same_mz_coord_as_commute_trip, "location_id"] = df_trips.loc[same_mz_coord_as_commute_trip, "commute_location_id"]

    df_trips = df_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose",
        "location_x", "location_y", "location_id", "reference_distance", "is_commute_trip"
    ]]

    return df_trips
