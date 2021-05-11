import pandas as pd

import data.spatial.utils as spatial_utils


def configure(context):
    context.stage("data.statent.statent")


def execute(context):
    df = pd.DataFrame(context.stage("data.statent.statent")[["enterprise_id", "x", "y", "noga"]],
                                    copy=True)
    df.columns = ["destination_id", "destination_x", "destination_y", "noga"]

    df.loc[:, "offers_work"] = True
    df.loc[:, "offers_other"] = True

    # 85 = education
    df.loc[:, "offers_education"] = df["noga"].str.startswith("85")

    # 90 = arts, entertainment, leisure; 56 = gastronomy
    df.loc[:, "offers_leisure"] = df["noga"].str.startswith("90") | df[
        "noga"].str.startswith("56")

    # 47 = retail
    df.loc[:, "offers_shop"] = df["noga"].str.startswith("47")

    del df["noga"]

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")

    return df[["destination_id", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "geometry"]]
