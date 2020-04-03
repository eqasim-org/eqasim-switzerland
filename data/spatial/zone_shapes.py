import pandas as pd


def configure(context):
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")

def execute(context):
    df_zones = context.stage("data.spatial.zones")
    df_municipalities = context.stage("data.spatial.municipalities")[0]
    df_quarters = context.stage("data.spatial.quarters")

    df_municipalities = pd.merge(
        df_municipalities, df_zones[df_zones["zone_level"] == "municipality"],
        right_on = "zone_level_id", left_on = "municipality_id"
    )[["zone_id", "zone_level_id", "zone_level", "geometry"]]

    df_quarters = pd.merge(
        df_quarters, df_zones[df_zones["zone_level"] == "quarter"],
        right_on = "zone_level_id", left_on = "quarter_id"
    )[["zone_id", "zone_level_id", "zone_level", "geometry"]]

    df = pd.concat([df_municipalities, df_quarters])

    #df["zone_level"] = df["zone_level"].astype("str")
    #df.to_file("/home/sebastian/zones.shp")

    return df
