import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np
import data.spatial.ov_guteklasse

def configure(context, require):
    require.stage("population.activities")
    require.stage("population.spatial.by_activity.primary_locations")
    require.stage("population.spatial.by_activity.subprimary_locations")
    require.stage("data.spatial.ov_guteklasse")

def execute(context):
    df_activities = context.stage("population.activities")

    df_locations = pd.concat([
        context.stage("population.spatial.by_activity.primary_locations"),
        context.stage("population.spatial.by_activity.subprimary_locations")
    ])

    df_locations = pd.merge(
        df_activities, df_locations,
        on = ["person_id", "activity_id"], how = "left"
    )

    df_locations = df_locations[[
        "person_id", "activity_id", "location_x", "location_y", "location_id"
    ]]

    # Impute OV Guteklasse
    print("Imputing ÖV Güteklasse ...")

    df_ov_guteklasse = context.stage("data.spatial.ov_guteklasse")

    df_impute = data.spatial.utils.to_gpd(pd.DataFrame(df_locations[["person_id", "activity_id", "location_x", "location_y"]], copy = True), "location_x", "location_y")
    df_impute = data.spatial.ov_guteklasse.impute(df_ov_guteklasse, df_impute, ["person_id", "activity_id"])
    df_locations = pd.merge(df_locations, df_impute, on = ["person_id", "activity_id"], how = "left")

    return df_locations
