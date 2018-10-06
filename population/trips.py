import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.work_locations")
    require.stage("population.education_locations")
    require.stage("data.microcensus.trips")

def execute(context):
    df_persons = context.stage("population.sociodemographics")

    df_trips = pd.DataFrame(context.stage("data.microcensus.trips"), copy = True)
    df_trips["mz_person_id"] = df_trips["person_id"]
    del df_trips["person_id"]

    df_trips = df_trips[[
        "mz_person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose"
    ]]

    df_trips = pd.merge(df_persons[["person_id", "mz_person_id"]], df_trips, how = "left")

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

    df_trips = df_trips[[
        "person_id", "trip_id", "departure_time", "arrival_time", "mode", "purpose",
        "location_x", "location_y", "location_id"
    ]]

    return df_trips
