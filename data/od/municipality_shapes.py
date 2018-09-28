import pandas as pd
import numpy as np
import data.constants as c
import geopandas as gpd
from tqdm import tqdm
from sklearn.neighbors import KDTree

def configure(context, require):
    require.stage("data.od.raw")
    require.config("raw_data_path")

CURRENT_YEAR = 2018

SHAPEFILES = [
    (2018, "municipality_borders/gd-b-00.03-875-gg18/ggg_2018-LV95/shp/g1g18.shp", "GMDNR"),
    (2017, "municipality_borders/gd-b-00.03-875-gg17/ggg_2017/shp/LV95/g1g17.shp", "GMDNR"),
    (2016, "municipality_borders/gd-b-00.03-875-gg16/ggg_2016/shp/g1g16.shp", "GMDNR"),
    (2015, "municipality_borders/gd-b-00.03-876-gg15/GGG_15_V161025/shp/g1g15.shp", "GMDNR"),
    (2014, "municipality_borders/gd-b-00.03-877-gg14/ggg_2014/shp/g1g14.shp", "GMDNR"),
    (2013, "municipality_borders/gd-b-00.03-877-gg13_r1/ggg_2013/shp/g1g13.shp", "GMDNR"),
    (2012, "municipality_borders/gd-b-00.03-878-gg12/g1g12_shp_121130/G1G12.shp", "GMDE"),
    (2011, "municipality_borders/gd-b-00.03-879-gg11/g1g11_shp_121130/G1G11.shp", "GMDE"),
    (2010, "municipality_borders/gd-b-00.03-880-gg10/g1g10_shp_121130/G1G10.shp", "GMDE"),
    (2009, "municipality_borders/gd-b-00.03-881-gg09g1/g1g09_shp_090626/G1G09.shp", "GMDE")
]

def execute(context):
    raw_data_path = context.config["raw_data_path"]
    df_od = context.stage("data.od.raw")

    requested_municipality_ids = set(np.unique(df_od[
        df_od["home_municipality"] > 0
    ][["home_municipality"]]))

    requested_municipality_ids |= set(np.unique(df_od[
        df_od["work_municipality"] > 0
    ][["work_municipality"]]))

    remaining_municipality_ids = set(requested_municipality_ids)
    number_of_municipality_ids = len(remaining_municipality_ids)

    municipality_dfs = []

    for year, shapefile, id_field in SHAPEFILES:
        shape_df = gpd.read_file(
            "%s/%s" % (raw_data_path, shapefile),
            encoding = "utf-8"
        ).to_crs({'init': 'EPSG:2056'})
        shape_df.loc[:, "zone"] = shape_df[id_field]
        shape_df.loc[:, "year"] = year

        f = shape_df["zone"].isin(remaining_municipality_ids)

        if year == CURRENT_YEAR:
            # For the current year we want all zones (although we may not have rquested all)
            municipality_dfs.append(shape_df[["zone", "year", "geometry"]])
        else:
            municipality_dfs.append(shape_df[f][["zone", "year", "geometry"]])

        remaining_municipality_ids -= set(shape_df.loc[f, "zone"])
        print("Found %d remaining municipalities in %d (%d remaining)" % (sum(f), year, len(remaining_municipality_ids)))

    assert(len(remaining_municipality_ids) == 0)

    df_all = pd.concat(municipality_dfs)

    # Split the data set in current shapes and the ones that ever existed
    df_current = gpd.GeoDataFrame(df_all)
    df_current = df_current[df_current["year"] == CURRENT_YEAR]
    df_current["zone_current"] = df_current["zone"]
    df_current = df_current[["zone_current", "geometry"]]

    df_ever = gpd.GeoDataFrame(df_all)
    df_ever["zone_ever"] = df_ever["zone"]
    df_ever["geometry"] = df_ever["geometry"].centroid
    df_ever = df_ever[["zone_ever", "geometry"]]

    # Find the corresponding current shape using "contains"
    df_contains = gpd.sjoin(
        df_current, df_ever, op = "contains", how = "right"
    ).reset_index()

    # Now we are left over with some old municipalities whose centroids
    # are not covered by any new municipality (mainly at the border and
    # close to lakes). Therefore, we do another run and find the current
    # municipality with the closes distance (more expensive operation).

    df_missing = df_contains[np.isnan(df_contains["zone_current"])][["zone_ever", "geometry"]]

    coordinates = np.vstack([df_current["geometry"].centroid.x, df_current["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].x, df_missing["geometry"].y]).T
    indices = kd_tree.query(coordinates, return_distance = False).flatten()

    df_missing.loc[:, "zone_current"] = df_current.iloc[indices]["zone_current"].values

    df_matching = pd.concat([
        df_contains[~np.isnan(df_contains["zone_current"])][["zone_current", "zone_ever"]],
        df_missing[["zone_current", "zone_ever"]]
    ])

    df_matching["zone_current"] = df_matching["zone_current"].astype(np.int)

    assert(len(df_matching) == len(df_ever))

    df_matching["zone"] = df_matching["zone_current"]
    df_matching["zone_previously"] = df_matching["zone_ever"]
    df_matching = df_matching[["zone", "zone_previously"]]

    df_current["zone"] = df_current["zone_current"]
    df_current = df_current[["zone", "geometry"]]

    return df_current, df_matching
