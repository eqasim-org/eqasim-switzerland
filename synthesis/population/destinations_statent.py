import pandas as pd

import data.spatial.utils as spatial_utils
import data.spatial.ovgk

def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.spatial.ovgk")


def execute(context):
    df = pd.DataFrame(context.stage("data.statent.statent")[["enterprise_id", "x", "y", "noga"]],
                                    copy=True)
    df.columns = ["destination_id", "destination_x", "destination_y", "noga"]

    df.loc[:, "offers_work"]  = True
    df.loc[:, "offers_other"] = True
    df.loc[:, "offers_work_secondary"] = True
    df.loc[:, "offers_home_secondary"] = True

    # 85 = education
    df.loc[:, "offers_education"] = df["noga"].str.startswith("85")
    df.loc[:, "offers_education_secondary"] = df["noga"].str.startswith("85")

    # 90 = arts, entertainment, leisure; 56 = gastronomy
    df.loc[:, "offers_leisure"] = df["noga"].str.startswith("90") | df[
        "noga"].str.startswith("56")

    # 47 = retail
    df.loc[:, "offers_shop"] = df["noga"].str.startswith("47")

    del df["noga"]

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")

    # impute ovgk
    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = data.spatial.ovgk.impute(context, df_ovgk, df, ["destination_id"], chunk_size=1e3, point_type="facility")
    df = df.merge(df_spatial[["destination_id", "ovgk"]], how="left", on="destination_id")

    return df[["destination_id", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "offers_work_secondary", "offers_education_secondary", "offers_home_secondary", "ovgk",
               "geometry"]]
