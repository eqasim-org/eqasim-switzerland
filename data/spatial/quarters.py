import geopandas as gpd
import numpy as np
import pandas as pd


def configure(context):
    context.config("data_path")

def execute(context):
    data_path = context.config("data_path")

    df = gpd.read_file(
        "%s/statistical_quarter_borders/shp/quart17.shp" % data_path,
        encoding = "latin1"
    ).to_crs("epsg:2056")

    df.crs = "epsg:2056"

    df["quarter_id"] = df["GMDEQNR"]
    df["quarter_name"] = df["NAME"]
    df = df[["quarter_id", "quarter_name", "geometry"]]

    return df

def update_quarter_ids(df, df_quarters, remove_unknown = False):
    assert("quarter_id" in df.columns)

    df["deprecated_quarter_id"] = df["quarter_id"]
    del df["quarter_id"]

    df_join = pd.merge(
        df[["deprecated_quarter_id"]], df_quarters,
        left_on = "deprecated_quarter_id", right_on = "quarter_id", how = "left"
    )

    df.loc[:, "quarter_id"] = df_join.loc[:, "quarter_id"].values

    if remove_unknown:
        return df[~np.isnan(df["quarter_id"])]
    else:
        return df
