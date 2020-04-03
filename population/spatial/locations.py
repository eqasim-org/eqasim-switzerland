import gzip
from tqdm import tqdm
import pandas as pd
import numpy as np

def configure(context):
    context.stage("population.activities")
    context.stage("population.spatial.by_activity.primary_locations")
    context.stage("population.spatial.by_activity.subprimary_locations")

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

    return df_locations
