import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    raw_data_path = context.config["raw_data_path"]

    df = gpd.read_file(
        "%s/statistical_quarter_borders/shp/quart17.shp" % raw_data_path,
        encoding = "latin1"
    ).to_crs({'init': 'EPSG:2056'})

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














#
