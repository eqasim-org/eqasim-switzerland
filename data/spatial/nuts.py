import geopandas as gpd
import numpy as np
import pandas as pd


def configure(context):
    context.config("data_path")


SHAPEFILES = [
    (2024, "spatial/nuts_borders//NUTS_RG_01M_2024_4326.shp", "NUTS_ID",
     "NUTS_NAME", "LEVL_CODE"),
    (2021, "spatial/nuts_borders//NUTS_RG_01M_2021_4326.shp", "NUTS_ID",
     "NUTS_NAME", "LEVL_CODE"),
]


def execute(context):
    data_path = context.config("data_path")

    df_all = []
    all_ids = set()

    # Load all the shape files, only add the NUTS zones that haven't been found before
    for year, shapefile, id_field, name_field, level_field in context.progress(SHAPEFILES,
                                                                               label="Reading NUTS shape files"):
        df = gpd.read_file(
            "%s/%s" % (data_path, shapefile),
            encoding="utf-8"
        )
        df.crs = "epsg:4326"
        df = df.to_crs("epsg:2056")

        df.loc[:, "nuts_id"] = df[id_field]
        df.loc[:, "nuts_name"] = df[name_field]
        df["nuts_level"] = df[level_field]
        df.loc[:, "year"] = year

        df_ids = set(np.unique(df["nuts_id"]))
        df_new_ids = df_ids - all_ids

        df_all.append(
            df[df["nuts_id"].isin(df_new_ids)][["nuts_id", "nuts_name", "nuts_level", "year", "geometry"]])
        all_ids |= df_new_ids

    df_all = pd.concat(df_all)
    return df_all
