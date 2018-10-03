import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm

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

def impute(df, df_quarters, fix_by_distance = True):
    assert(not "quarter_id" in df.columns)

    print("Imputing %d quarters by spatial join..." % len(df))

    result = []
    chunk_count = int(len(df) / 10000)
    for chunk in tqdm(np.array_split(df, chunk_count), total = chunk_count):
        result.append(gpd.sjoin(df_quarters, chunk, op = "contains", how = "right"))
    df = pd.concat(result).reset_index()

    invalid_mask = np.isnan(df["quarter_id"])
    df.loc[~invalid_mask, "quarter_id"] = df.loc[~invalid_mask, "quarter_id"]

    if fix_by_distance and np.any(invalid_mask):
        print("  Fixing %d observations by distance join..." % np.count_nonzero(invalid_mask))
        coordinates = np.vstack([df_quarters["geometry"].centroid.x, df_quarters["geometry"].centroid.y]).T
        kd_tree = KDTree(coordinates)

        df_missing = df[invalid_mask]
        coordinates = np.vstack([df_missing["geometry"].centroid.x, df_missing["geometry"].centroid.y]).T
        indices = kd_tree.query(coordinates, return_distance = False).flatten()

        df.loc[invalid_mask, "quarter_id"] = df_quarters.iloc[indices]["quarter_id"].values

    return df












#
