import numpy as np
import pandas as pd

import data.spatial.utils as spatial_utils
import data.spatial.zone_shapes


def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.zone_shapes")
    context.stage("synthesis.population.spatial.primary.work.zones")


def execute(context):
    df = context.stage("synthesis.population.spatial.primary.work.zones")
    df_statent = context.stage("data.statent.statent")

    df_zones = context.stage("data.spatial.zones")
    df_zones["work_zone_id"] = df_zones["zone_id"]

    df_demand = df.groupby("work_zone_id").size().reset_index(name="count")
    df_demand = pd.merge(df_demand, df_zones[["work_zone_id", "zone_level"]])

    # First handle the national commuters
    df_national = df_demand[df_demand["zone_level"].isin(("municipality", "quarter"))]
    empty_zones = []

    for zone_id, count in context.progress(zip(df_national["work_zone_id"], df_national["count"]),
                                           label="Assigning national locations ...", total=len(df_demand)):
        indices = np.where(df_statent["zone_id"] == zone_id)[0]
        weights = df_statent.iloc[indices]["number_employees"]
        weights /= np.sum(weights)

        if len(indices) > 0:
            indices = np.repeat(indices, np.random.multinomial(count, weights))

            f = df["work_zone_id"] == zone_id
            df.loc[f, "work_x"] = df_statent.iloc[indices]["x"].values
            df.loc[f, "work_y"] = df_statent.iloc[indices]["y"].values
            df.loc[f, "work_location_id"] = df_statent.iloc[indices]["enterprise_id"].values
        else:
            empty_zones.append(zone_id)

    print("Found %d zones which do not have any samples in STATENT" % len(empty_zones))

    # There are some empty zones (mainly border zones to Italy, which are under shared administration)
    # There, we just sample a random location inside of the zone

    df_shapes = context.stage("data.spatial.zone_shapes")

    for zone_id in context.progress(empty_zones, label="Assigning national locations for empty zones ..."):
        count = df_national[df_national["work_zone_id"] == zone_id]["count"].iloc[0]
        row = df_shapes[df_shapes["zone_id"] == zone_id].iloc[0]
        coordinates = data.spatial.zone_shapes.sample_coordinates(row, count)
        df.loc[df["work_zone_id"] == zone_id, "work_x"] = coordinates[:, 0]
        df.loc[df["work_zone_id"] == zone_id, "work_y"] = coordinates[:, 1]

    # Second, handle the international commuters
    print("TODO: We do not handle commuter traffic at the moment.")

    # For now, make sure that we do not have any international traffic
    df_international = df_demand[df_demand["zone_level"] == "country"]

    assert (len(df_international) == 0)
    assert (len(df) == len(df.dropna()))

    df = df[["person_id",
             "work_x", "work_y",
             "work_location_id"]].rename({"work_x": "x",
                                          "work_y": "y",
                                          "work_location_id": "destination_id"},
                                         axis=1)

    df = spatial_utils.to_gpd(context, df, coord_type="work")

    return df[["person_id", "destination_id", "geometry"]]
