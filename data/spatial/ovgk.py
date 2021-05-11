import geopandas as gpd
import numpy as np
import pandas as pd


def configure(context):
    context.config("data_path")
    context.config("threads")


def execute(context):
    input_path = "%s/ov_guteklasse/LV95/Oev_Gueteklassen_ARE.shp" % context.config("data_path")
    df = gpd.read_file(input_path)
    df.crs = "epsg:2056"
    df = df[["KLASSE", "geometry"]].rename({"KLASSE": "ovgk"}, axis=1)
    return df


def impute(context, df_ovgk, df, on):
    indices = np.array_split(np.arange(len(df)), 100)
    df_join = []

    for chunk in context.progress(indices, label="Imputing ÖV Güteklasse"):
        df_join.append(gpd.sjoin(df.iloc[chunk], df_ovgk, op="within")[on + ["ovgk"]])

    df_join = pd.concat(df_join)
    df_join = pd.merge(df, df_join, on=on, how="left")
    df_join.loc[df_join["ovgk"].isna(), "ovgk"] = "None"
    df_join["ovgk"] = df_join["ovgk"].astype("category")

    return df_join[on + ["ovgk"]]
