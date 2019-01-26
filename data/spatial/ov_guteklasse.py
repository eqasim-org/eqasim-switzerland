import pandas as pd
import numpy as np
import geopandas as gpd

def configure(context, require):
    require.config("raw_data_path")

def execute(context):
    input_path = "%s/ov_guteklasse/LV95/Oev_Gueteklassen_ARE.shp" % context.config["raw_data_path"]
    df = gpd.read_file(input_path)
    df.crs = {"init" : "EPSG:2056"}
    df = df[["KLASSE", "geometry"]].rename({"KLASSE" : "ov_guteklasse"}, axis = 1)
    return df

def impute(df_ov_guteklasse, df, on):
    df_join = gpd.sjoin(df, df_ov_guteklasse, op = "within")[on + ["ov_guteklasse"]]
    df_join = pd.merge(df, df_join, on = on, how = "left")
    df_join.loc[df_join["ov_guteklasse"].isna(), "ov_guteklasse"] = "None"
    df_join["ov_guteklasse"] = df_join["ov_guteklasse"].astype("category")
    return df_join[on + ["ov_guteklasse"]]
