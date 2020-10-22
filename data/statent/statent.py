import numpy as np
import pandas as pd

import data.spatial.municipalities
import data.spatial.quarters
import data.spatial.utils
import data.spatial.zones
import data.utils


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.zones")
    context.stage("data.spatial.municipalities")
    context.stage("data.spatial.quarters")
    context.stage("data.spatial.nuts")
    context.stage("data.spatial.postal_codes")

def execute(context):
    data_path = context.config("data_path")

    df = pd.DataFrame(pd.read_csv(
        "%s/statent/QUERY_FOR_2014_DEC_STATENT_LOC.csv" % data_path,
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
    df_nuts = context.stage("data.spatial.nuts")
    df_postal_codes = context.stage("data.spatial.postal_codes")

    df_spatial = pd.DataFrame(df[["enterprise_id", "x", "y"]])

    df_spatial = data.spatial.utils.to_gpd(context, df_spatial, "x", "y")
    df_spatial = df_spatial.drop(["x", "y"], axis=1)

    columns = ["enterprise_id"]

    # impute municipalities
    df_spatial = data.spatial.utils.impute(context, df_spatial, df_municipalities, "enterprise_id", "municipality_id")
    columns.extend(["municipality_id"])

    # impute quarters
    df_spatial = data.spatial.utils.impute(context, df_spatial, df_quarters, "enterprise_id", "quarter_id", fix_by_distance = False)
    columns.extend(["quarter_id"])

    # impute NUTS
    df_nuts = df_nuts[df_nuts["nuts_id"].str.contains("CH")]
    for level in df_nuts["nuts_level"].unique():
        df_spatial = data.spatial.utils.impute(context, df_spatial, df_nuts[df_nuts["nuts_level"] == level],
                                               "enterprise_id", "nuts_id", fix_by_distance=False).rename(
            {"nuts_id": ("nuts_id_level_" + str(level))}, axis=1)
        columns.extend([("nuts_id_level_" + str(level))])

    # impute postal codes
    df_spatial = data.spatial.utils.impute(context, df_spatial, df_postal_codes, "enterprise_id", "postal_code", fix_by_distance=False)
    columns.extend(["postal_code"])

    # clean up columns
    df_spatial = df_spatial[columns]

    # impute zones
    df_spatial = data.spatial.zones.impute(df_spatial, df_zones)

    assert(len(df) == len(df_spatial))
    assert(len(df_spatial) == len(df_spatial["zone_id"].dropna()))

    columns.extend(["zone_id"])

    df = pd.merge(
        df, df_spatial[columns],
        on = "enterprise_id"
    )
    df["zone_id"] = df["zone_id"].astype(np.int)

    return df
