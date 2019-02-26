import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    # Load data
    raw_data_path = context.config["raw_data_path"]

    df_cantons = gpd.read_file("%s/municipality_types/ARE_GemTyp00_9.shp" % raw_data_path)
    df_cantons["municipality_id"] = df_cantons["BFS_NO"]
    df_cantons["canton_id"] = df_cantons["KT_NO"]
    df_cantons = df_cantons[["municipality_id", "canton_id"]]

    return df_cantons

def impute(df_cantons, df):
    assert("municipality_id" in df.columns)
    return pd.merge(df, df_cantons, on = "municipality_id", how = "left")

SP_REGION_1 = [25, 12, 13, 1, 2, 14, 9]
SP_REGION_2 = [21, 26, 15, 16, 22, 11, 24, 3, 6, 7]
SP_REGION_3 = [17, 19, 10, 23, 20, 5, 18, 4, 8]

def impute_sp_region(df):
    assert("canton_id" in df.columns)
    assert("sp_region" not in df.columns)

    df["sp_region"] = 0
    df.loc[df["canton_id"].isin(SP_REGION_1), "sp_region"] = 1
    df.loc[df["canton_id"].isin(SP_REGION_2), "sp_region"] = 2
    df.loc[df["canton_id"].isin(SP_REGION_3), "sp_region"] = 3

    # TODO: There are some municipalities that are not included in the shape
    # file above. Hence, they get region 0. Should be fixed in the future.
    # Especially, we need a consistent spatial system. It probably would make
    # more sense to impute the SP region in another way

    #assert(not np.any(df["sp_region"] == 0))
    return df
