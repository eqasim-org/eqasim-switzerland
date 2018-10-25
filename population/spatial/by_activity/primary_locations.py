import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("population.spatial.by_person.work_locations")
    require.stage("population.spatial.by_person.education_locations")
    require.stage("population.activities")

def execute(context):
    df_activities = context.stage("population.activities")

    expected_work_locations = np.count_nonzero(df_activities["is_commute"] & (df_activities["purpose"] == "work"))
    expected_education_locations = np.count_nonzero(df_activities["is_commute"] & (df_activities["purpose"] == "education"))

    # Merge in home locations
    df_home_locations = pd.DataFrame(context.stage("population.sociodemographics")[[
        "person_id", "home_x", "home_y"
    ]], copy = True)
    df_home_locations.loc[:, "location_id"] = np.nan
    df_home_locations.columns = ["person_id", "location_x", "location_y", "location_id"]

    df_home_activities = df_activities[df_activities["purpose"] == "home"]
    df_home_locations = pd.merge(df_home_activities, df_home_locations, on = "person_id")

    # Merge in work locations
    df_work_locations = pd.DataFrame(context.stage("population.spatial.by_person.work_locations")[[
        "person_id", "work_x", "work_y", "work_location_id"
    ]], copy = True)
    df_work_locations.columns = ["person_id", "location_x", "location_y", "location_id"]

    df_work_activities = df_activities[df_activities["is_commute"] & (df_activities["purpose"] == "work")]
    df_work_locations = pd.merge(df_work_activities, df_work_locations, on = "person_id")
    assert(expected_work_locations == len(df_work_locations))

    # Merge in education locations
    df_education_locations = pd.DataFrame(context.stage("population.spatial.by_person.education_locations")[[
        "person_id", "education_x", "education_y", "education_location_id"
    ]], copy = True)
    df_education_locations.columns = ["person_id", "location_x", "location_y", "location_id"]

    df_education_activities = df_activities[df_activities["is_commute"] & (df_activities["purpose"] == "education")]
    df_education_locations = pd.merge(df_education_activities, df_education_locations, on = "person_id")
    assert(expected_education_locations == len(df_education_locations))

    df_primary_locations = pd.concat([
        df_home_locations,
        df_work_locations,
        df_education_locations
    ])

    df_primary_locations = df_primary_locations[[
        "person_id", "activity_id", "location_x", "location_y", "location_id"
    ]]

    df_primary_locations = df_primary_locations.sort_values(by = ["person_id", "activity_id"])
    return df_primary_locations
