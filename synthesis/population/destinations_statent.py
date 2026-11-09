import hashlib
from pathlib import Path

import pandas as pd
import geopandas as gpd
import data.activity_purposes as activity_purposes
import data.spatial.utils as spatial_utils
from data.spatial.ovgk import impute as impute_ovgk
from data.activity_purposes import statent_activity_masks

def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.statent.education_institutions")
    context.stage("data.spatial.ovgk")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")

def execute(context):

    # Prepare destination purposes and coordinates.

    df = prepare_destinations(context.stage("data.statent.statent"),
                              context.stage("data.statent.education_institutions"))

    df = spatial_utils.to_gpd(context, df, x="destination_x", y="destination_y", coord_type="facility")

    # Impute public transport quality.

    df_ovgk = context.stage("data.spatial.ovgk")
    df_spatial = impute_ovgk(context, df_ovgk, df, ["destination_id"], chunk_size=1e3, point_type="facility")
    df = df.merge(df_spatial[["destination_id", "ovgk"]], how="left", on="destination_id")

    # Impute municipality types.

    df_municipality_type = context.stage("data.spatial.municipality_types")
    df_municipalities,_ = context.stage("data.spatial.municipalities")
    df_municipalities = df_municipalities.merge(df_municipality_type)[["municipality_type","municipality_id", "geometry"]]
    assert df.crs == df_municipalities.crs
    df = gpd.sjoin_nearest(df, df_municipalities, how="left").drop(columns=["index_right"])

    # Return the standard destination catalogue.

    return df[["destination_id", "number_employees", "destination_x", "destination_y",
               "offers_work", "offers_education", "offers_leisure", "offers_shop", "offers_other",
               "offers_work_secondary", "offers_education_secondary", "offers_home_secondary",
               "ovgk", "municipality_id", "municipality_type",
               "geometry"]]


def prepare_destinations(df_statent, registry):
    df = pd.DataFrame(df_statent[["enterprise_id", "x", "y", "noga", "number_employees"]], copy=True)
    df.columns = ["destination_id", "destination_x", "destination_y", "noga", "number_employees"]

    df.loc[:, "offers_work"] = True
    df.loc[:, "offers_other"] = True
    df.loc[:, "offers_work_secondary"] = True
    df.loc[:, "offers_home_secondary"] = True

    # 85 = education; doctoral destinations can include research establishments.
    df.loc[:, "offers_education"] = df["noga"].str.startswith("85")
    compatibility = registry["compatibility"]
    doctoral_ids = compatibility.loc[compatibility["study_level"].eq("doctorate_habilitation"), "enterprise_id"]
    df.loc[df["destination_id"].isin(doctoral_ids), "offers_education"] = True
    df.loc[:, "offers_education_secondary"] = df["noga"].str.startswith("85")

    # Match microcensus shop/leisure purposes using the shared NOGA 2008 map.
    df["offers_shop"], df["offers_leisure"] = statent_activity_masks(df["noga"])
    return df.drop(columns="noga")


def validate(context):
    return hashlib.sha256(Path(activity_purposes.__file__).read_bytes()).hexdigest()
