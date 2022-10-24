import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def configure(context):
    context.config("data_path")


REFERENCE_YEAR = 2018

SHAPEFILES = [
    (2018, "municipality_borders/gd-b-00.03-875-gg18/ggg_2018-LV95/shp/g1g18.shp", "GMDNR", "GMDNAME"),
    (2017, "municipality_borders/gd-b-00.03-875-gg17/ggg_2017/shp/LV95/g1g17.shp", "GMDNR", "GMDNAME"),
    (2016, "municipality_borders/gd-b-00.03-875-gg16/ggg_2016/shp/g1g16.shp", "GMDNR", "GMDNAME"),
    (2015, "municipality_borders/gd-b-00.03-876-gg15/GGG_15_V161025/shp/g1g15.shp", "GMDNR", "GMDNAME"),
    (2014, "municipality_borders/gd-b-00.03-877-gg14/ggg_2014/shp/g1g14.shp", "GMDNR", "GMDNAME"),
    (2013, "municipality_borders/gd-b-00.03-877-gg13_r1/ggg_2013/shp/g1g13.shp", "GMDNR", "GMDNAME"),
    (2012, "municipality_borders/gd-b-00.03-878-gg12/g1g12_shp_121130/G1G12.shp", "GMDE", "NAME"),
    (2011, "municipality_borders/gd-b-00.03-879-gg11/g1g11_shp_121130/G1G11.shp", "GMDE", "NAME"),
    (2010, "municipality_borders/gd-b-00.03-880-gg10/g1g10_shp_121130/G1G10.shp", "GMDE", "NAME"),
    (2009, "municipality_borders/gd-b-00.03-881-gg09g1/g1g09_shp_090626/G1G09.shp", "GMDE", "NAME")
]


def execute(context):
    data_path = context.config("data_path")

    df_all = []
    all_ids = set()

    # Load all the shape files, only add the municipalities that haven't been found before
    for year, shapefile, id_field, name_field in context.progress(SHAPEFILES, label="Reading municipality shape files"):
        df = gpd.read_file(
            "%s/%s" % (data_path, shapefile),
            encoding="latin1"
        ).to_crs("epsg:2056")

        df.crs = "epsg:2056"

        df.loc[:, "municipality_id"] = df[id_field]
        df.loc[:, "municipality_name"] = df[name_field]
        df.loc[:, "year"] = year

        df_ids = set(np.unique(df["municipality_id"]))
        df_new_ids = df_ids - all_ids

        df_all.append(
            df[df["municipality_id"].isin(df_new_ids)][["municipality_id", "municipality_name", "year", "geometry"]])
        all_ids |= df_new_ids

    df_all = pd.concat(df_all)

    df_reference = gpd.GeoDataFrame(df_all[df_all["year"] == REFERENCE_YEAR])
    df_reference.crs = df_all.crs

    df_deprecated = gpd.GeoDataFrame(df_all[df_all["year"] != REFERENCE_YEAR])
    df_deprecated["deprecated_municipality_id"] = df_deprecated["municipality_id"]
    del df_deprecated["municipality_id"]
    df_deprecated["geometry"] = df_deprecated.centroid
    df_deprecated.crs = df_all.crs

    # For each deprecated municipality find the covering reference municipality
    df_mapping = gpd.sjoin(
        df_reference, df_deprecated, op="contains"
    ).reset_index()[["municipality_id", "deprecated_municipality_id"]]

    # Now we are left over with some old municipalities whose centroids
    # are not covered by any new municipality (mainly at the border and
    # close to lakes). Therefore, we do another run and find the current
    # municipality with the closes distance (more expensive operation).

    missing_ids = set(
        np.unique(df_deprecated["deprecated_municipality_id"])
    ) - set(np.unique(df_mapping["deprecated_municipality_id"]))

    df_missing = df_deprecated[
        df_deprecated["deprecated_municipality_id"].isin(missing_ids)
    ][["deprecated_municipality_id", "geometry"]]

    coordinates = np.vstack([df_reference["geometry"].centroid.x, df_reference["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].x, df_missing["geometry"].y]).T
    indices = kd_tree.query(coordinates, return_distance=False).flatten()

    df_missing.loc[:, "municipality_id"] = df_reference.iloc[indices]["municipality_id"].values
    df_missing = df_missing[["municipality_id", "deprecated_municipality_id"]]

    df_existing = pd.DataFrame(df_reference[["municipality_id"]])
    df_existing["deprecated_municipality_id"] = df_existing["municipality_id"]

    df_mapping = pd.concat([df_existing, df_mapping, df_missing])
    df_reference = df_reference[["municipality_id", "municipality_name", "geometry"]]

    return df_reference, df_mapping


def update_municipality_ids(df, df_mapping, remove_unknown=False):
    assert ("municipality_id" in df.columns)

    df["deprecated_municipality_id"] = df["municipality_id"]
    del df["municipality_id"]

    df_join = pd.merge(
        df[["deprecated_municipality_id"]], df_mapping,
        on="deprecated_municipality_id", how="left"
    )

    df.loc[:, "municipality_id"] = df_join.loc[:, "municipality_id"].values

    if remove_unknown:
        return df[~np.isnan(df["municipality_id"])]
    else:
        return df
