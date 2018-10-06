import pandas as pd
import numpy as np
import data.constants as c
from tqdm import tqdm
import data.spatial.zones
import data.utils
import data.spatial.municipalities
import data.spatial.quarters
import data.spatial.utils

def configure(context, require):
    require.config("raw_data_path")
    require.stage("data.spatial.zones")
    require.stage("data.spatial.municipalities")
    require.stage("data.spatial.quarters")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df = pd.DataFrame(pd.read_csv(
        "%s/statent/QUERY_FOR_2014_DEC_STATENT_LOC.csv" % raw_data_path,
        encoding = "latin1", sep = ";"))

    df = pd.DataFrame(df[["METER_X", "METER_Y", "NOGA08", "EMPTOT"]])
    df.columns = ["x", "y", "noga", "number_employees"]
    df.loc[:, "noga"] = df["noga"].astype(np.str)
    df.loc[:, "enterprise_id"] = np.arange(len(df))

    df.loc[df["noga"].str.startswith("851"), "education_type"] = "kindergarten"
    df.loc[df["noga"].str.startswith("852"), "education_type"] = "primary"
    df.loc[df["noga"].str.startswith("853"), "education_type"] = "secondary"
    df.loc[df["noga"].str.startswith("854"), "education_type"] = "tertiary"
    df["education_type"] = df["education_type"].astype("category")

    # For now we don't do anything with the NOGA category.
    # (but need to do later for the education locations)

    # Impute zones
    df_zones = context.stage("data.spatial.zones")
    df_quarters = context.stage("data.spatial.quarters")
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    df_spatial = pd.DataFrame(df[["enterprise_id", "x", "y"]])
    df_spatial = data.spatial.utils.to_gpd(df_spatial, "x", "y")

    df_spatial = data.spatial.utils.impute(df_spatial, df_municipalities, "enterprise_id", "municipality_id")[[
        "enterprise_id", "municipality_id", "geometry"
    ]]

    df_spatial = data.spatial.utils.impute(df_spatial, df_quarters, "enterprise_id", "quarter_id", fix_by_distance = False)[[
        "enterprise_id", "municipality_id", "quarter_id", "geometry"
    ]]

    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)[[
        "enterprise_id", "zone_id", "zone_level", "municipality_id", "quarter_id"
    ]]

    assert(len(df) == len(df_spatial))
    assert(len(df_spatial) == len(df_spatial["zone_id"].dropna()))

    df = pd.merge(
        df, df_spatial[["enterprise_id", "zone_id", "municipality_id", "quarter_id"]],
        on = "enterprise_id"
    )
    df["zone_id"] = df["zone_id"].astype(np.int)

    return df
