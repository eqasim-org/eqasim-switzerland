import data.spatial.utils as spatial_utils
import data.spatial.ovgk
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
    context.stage("data.spatial.ovgk")
    
def configure(context):
    df = context.stage("work_locations")
    df = df[df["work_remotly"] == True]
    df["destination_x"] = df.geometry.x
    df["destination_y"] = df.geometry.y
    df = df[["destination_id", "destination_x", "destination_y"]]
    df = df.drop_duplicates(subset=["destination_id"]).reset_index(drop=True)

    df.loc[:, "offers_work"]  = True
    df.loc[:, "offers_other"] = False
    df.loc[:, "offers_work_secondary"] = False
    df.loc[:, "offers_home_secondary"] = False
    df.loc[:, "offers_education"] = False
    df.loc[:, "offers_education_secondary"] = False
    df.loc[:, "offers_leisure"] = False
    df.loc[:, "offers_shop"] = False

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")

    # impute ovgk
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df, ["destination_id"], chunk_size=1e3, point_type="facility")
    df = df.merge(df_spatial[["destination_id", "ovgk"]], how="left", on="destination_id")

    return df[["destination_id", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "offers_work_secondary", "offers_education_secondary", "offers_home_secondary", "ovgk",
               "geometry"]]