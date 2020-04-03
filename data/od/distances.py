import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.spatial.zones")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.municipalities")

COUNRY_DISTANCE = 800 * 1e3

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_quarters = context.stage("data.spatial.quarters")
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    df_quarters = pd.merge(
        df_quarters[["quarter_id", "geometry"]],
        df_zones[df_zones["zone_level"] == "quarter"][[
            "zone_level_id", "zone_id", "zone_level"
        ]],
        right_on = "zone_level_id", left_on = "quarter_id")
    df_quarters["geometry"] = df_quarters["geometry"].centroid

    df_municipalities = pd.merge(
        df_municipalities[["municipality_id", "geometry"]],
        df_zones[df_zones["zone_level"] == "municipality"][[
            "zone_level_id", "zone_id", "zone_level"
        ]],
        right_on = "zone_level_id", left_on = "municipality_id")
    df_municipalities["geometry"] = df_municipalities["geometry"].centroid

    # TODO: What do we do with countries? Right now, we just set a long distance
    # later on.

    df = pd.concat([
        df_quarters[["zone_id", "zone_level", "geometry"]],
        df_municipalities[["zone_id", "zone_level", "geometry"]]
    ])

    assert(len(df) == np.sum(df_zones["zone_level"].isin(["quarter", "municipality"])))

    zone_ids = list(df_zones["zone_id"])

    # We get the coordinates and indices in the zonal data set for the zones that have a geometry
    coordinates = np.vstack([df["geometry"].centroid.x, df["geometry"].centroid.y]).T
    df_indices = [zone_ids.index(zone_id) for zone_id in df["zone_id"]]

    # Set the default distance to COUNTRY_DISTANCE. All others will be overridden.
    distances = np.ones((len(df_zones), len(df_zones))) * COUNRY_DISTANCE

    # Here we juggle around with the indices to create a distance matrix
    for coord_index, (df_index, coordinate) in enumerate(zip(df_indices, coordinates)):
        distances[df_index,df_indices] = np.sqrt(np.sum((coordinates - coordinates[coord_index,:])**2, axis = 1))

    return distances
