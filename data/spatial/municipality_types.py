import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


def configure(context):
    context.config("data_path")
    context.stage("data.spatial.municipalities")


def execute(context):
    # Load data
    data_path = context.config("data_path")

    df_types = pd.read_excel("%s/spatial_structure_2018.xlsx" % data_path,
                             names=["municipality_id", "TYP"],
                             usecols=[0, 21],
                             skiprows=6,
                             nrows=2229,
                             )
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    # Rewrite classification
    df_types.loc[df_types["TYP"] == 1, "municipality_type"] = "urban"
    df_types.loc[df_types["TYP"] == 2, "municipality_type"] = "urban"
    df_types.loc[df_types["TYP"] == 3, "municipality_type"] = "suburban"
    df_types.loc[df_types["TYP"] == 4, "municipality_type"] = "urban"
    df_types.loc[df_types["TYP"] == 5, "municipality_type"] = "suburban"
    df_types.loc[df_types["TYP"] == 6, "municipality_type"] = "rural"
    df_types.loc[df_types["TYP"] == 7, "municipality_type"] = "rural"
    df_types.loc[df_types["TYP"] == 8, "municipality_type"] = "rural"
    df_types.loc[df_types["TYP"] == 9, "municipality_type"] = "rural"

    df_types["municipality_type"] = df_types["municipality_type"].astype("category")
    df_types = df_types[["municipality_id", "municipality_type"]]

    # Match by municipality_id
    df_existing = pd.merge(df_municipalities, df_types, on="municipality_id")
    df_existing["imputed_municipality_type"] = False
    df_existing = df_existing[["municipality_id", "municipality_type", "imputed_municipality_type", "geometry"]]

    # Some ids are missing (because they are special zones)
    df_missing = gpd.GeoDataFrame(df_municipalities[
                                      ~df_municipalities["municipality_id"].isin(df_existing["municipality_id"])
                                  ])
    df_missing.crs = df_municipalities.crs
    df_missing = df_missing[["municipality_id", "geometry"]]

    print("Imputing %d spatial types by distance..." % len(df_missing))
    coordinates = np.vstack([df_existing["geometry"].centroid.x, df_existing["geometry"].centroid.y]).T
    kd_tree = KDTree(coordinates)

    coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
    indices = kd_tree.query(coordinates, return_distance=False).flatten()

    df_missing.loc[:, "municipality_type"] = df_existing.iloc[indices]["municipality_type"].values
    df_missing.loc[:, "imputed_municipality_type"] = True
    df_missing = df_missing[["municipality_id", "municipality_type", "imputed_municipality_type", "geometry"]]

    df_mapping = pd.concat((df_existing, df_missing))

    assert (len(df_mapping) == len(df_municipalities))
    assert (set(np.unique(df_mapping["municipality_id"])) == set(np.unique(df_municipalities["municipality_id"])))

    df_mapping = pd.DataFrame(df_mapping[["municipality_id", "municipality_type", "imputed_municipality_type"]])
    df_mapping["municipality_type"] = df_mapping["municipality_type"].astype("category")

    return df_mapping


def impute(df, df_municipality_types, remove_unknown=False):
    assert ("municipality_id" in df.columns)
    df = pd.merge(df, df_municipality_types, on="municipality_id")

    if remove_unknown:
        return df[~np.isnan(df["municipality_type"])]
    else:
        return df
