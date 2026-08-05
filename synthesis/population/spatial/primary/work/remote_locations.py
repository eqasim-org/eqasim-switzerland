import data.spatial.utils as spatial_utils
import data.spatial.ovgk
import geopandas as gpd
import logging
logger = logging.getLogger("synpp")


def configure(context):
    context.stage("synthesis.population.spatial.primary.work.work_remotly", alias="remote_locations")
    context.stage("data.spatial.ovgk")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")

    
def execute(context):
    df = context.stage("remote_locations")
    
    df["destination_x"] = df["x"]
    df["destination_y"] = df["y"]
    df = df[["destination_id", "destination_x", "destination_y"]]
    df = df.drop_duplicates(subset=["destination_id"]).reset_index(drop=True)

    df.loc[:, "offers_work"]  = False
    df.loc[:, "offers_other"] = False
    df.loc[:, "offers_work_secondary"] = False
    df.loc[:, "offers_home_secondary"] = False
    df.loc[:, "offers_education"] = False
    df.loc[:, "offers_education_secondary"] = False
    df.loc[:, "offers_leisure"] = False
    df.loc[:, "offers_shop"] = False
    df.loc[:, "number_employees"] = 0

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")

    # impute ovgk
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df, ["destination_id"], chunk_size=1e3, point_type="facility")
    df = df.merge(df_spatial[["destination_id", "ovgk"]], how="left", on="destination_id")

    # impute municipality types
    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities,_ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type","municipality_id", "geometry"]]
    assert df.crs == df_municipalities.crs
    df = gpd.sjoin_nearest(df, df_municipalities, how="left").drop(columns=["index_right"])

    return df[["destination_id", "number_employees", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "offers_work_secondary", "offers_education_secondary", "offers_home_secondary", "ovgk",
               "municipality_id", "municipality_type", "geometry"]]