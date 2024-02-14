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

    df_types = pd.read_excel("%s/urbanisierungsgrad2018.xlsx" % data_path,
                             names=["BFS Gde-nummer", "Urbanisierungsgrad 2011 (DEGURBA) - eurostat"],
                             usecols=[0, 6],
                             skiprows=2,
                             nrows=2225,
                             )
    df_municipalities = context.stage("data.spatial.municipalities")[0]

    df_types.columns = ["municipality_id", "muntype"]

    df_types.loc[df_types["muntype"]==1, "urbanisation_level"] = "high"
    df_types.loc[df_types["muntype"]==2, "urbanisation_level"] = "medium"
    df_types.loc[df_types["muntype"]==3, "urbanisation_level"] = "low"

    df_types["urbanisation_level"] = df_types["urbanisation_level"].astype("category")
    df_types = df_types[["municipality_id", "urbanisation_level"]]

    # Match by municipality_id
    df_existing = pd.merge(df_municipalities, df_types, on="municipality_id")
    df_existing["imputed_urbanisation"] = False
    df_existing = df_existing[["municipality_id", "urbanisation_level", "imputed_urbanisation", "geometry"]]

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

    df_missing.loc[:, "urbanisation_level"] = df_existing.iloc[indices]["urbanisation_level"].values
    df_missing.loc[:, "imputed_urbanisation"] = True
    df_missing = df_missing[["municipality_id", "urbanisation_level", "imputed_urbanisation", "geometry"]]

    df_mapping = pd.concat((df_existing, df_missing))

    assert (len(df_mapping) == len(df_municipalities))
    assert (set(np.unique(df_mapping["municipality_id"])) == set(np.unique(df_municipalities["municipality_id"])))

    df_mapping = pd.DataFrame(df_mapping[["municipality_id", "urbanisation_level", "imputed_urbanisation"]])
    df_mapping["urbanisation_level"] = df_mapping["urbanisation_level"].astype("category")

    return df_mapping


def impute(df, df_municipality_types, remove_unknown=False):
    assert ("municipality_id" in df.columns)
    df = pd.merge(df, df_municipality_types, on="municipality_id")

    if remove_unknown:
        return df[~np.isnan(df["urbanisation_level"])]
    else:
        return df
