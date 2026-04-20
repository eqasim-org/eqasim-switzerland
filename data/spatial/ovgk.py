import geopandas as gpd
import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed

logger = logging.getLogger("synpp")

def configure(context):
    context.config("data_path")
    context.config("threads")


def execute(context):
    input_path = "%s/spatial/ov_guteklasse/OeV_Gueteklassen_ARE.gpkg" % context.config("data_path")
    df = gpd.read_file(input_path)
    df.crs = "epsg:2056"
    df = df[["KLASSE", "geometry"]].rename({"KLASSE": "ovgk"}, axis=1)
    return df


def impute(context, df_ovgk, df, on, point_type="", chunk_size=100):
    indices = np.array_split(np.arange(len(df)), chunk_size)
    df_join = []

    logger.info(f"Imputing ÖV Güteklasse for {len(df)} {point_type} coordinates...")
    for chunk in context.progress(indices, total=len(indices), label="Imputing ÖV Güteklasse..."):
        df_join.append(gpd.sjoin(df.iloc[chunk], df_ovgk, predicate="within")[on + ["ovgk"]])

    df_join = pd.concat(df_join)
    df_join = pd.merge(df, df_join, on=on, how="left")
    df_join.loc[df_join["ovgk"].isna(), "ovgk"] = "None"
    df_join["ovgk"] = df_join["ovgk"].astype("category")

    return df_join[on + ["ovgk"]]

def impute_parallel(context, df, x="x", y="y", geometry="geometry", output_column="ovgk", point_type="", chunk_size=5000, n_jobs=8):
    if geometry not in df.columns:
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"df must contain either a {geometry} column or both {x} and {y} columns")
        df[geometry] = gpd.points_from_xy(df[x], df[y], crs="epsg:2056")
        df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:2056")

    
    df_ovgk = context.stage("data.spatial.ovgk")
    df_ovgk.rename(columns={"ovgk": output_column}, inplace=True)
    assert df_ovgk.crs==df.crs, "CRS of df and df_ovgk must match"

    total_points = len(df)
    logger.info("Imputing OeV Gueteklasse for %d %s coordinates in parallel...", total_points, point_type)

    if total_points == 0:
        result = df.copy()
        result[output_column] = pd.Series(dtype="category")
        return result

    if n_jobs is None:
        n_jobs = int(context.config("threads"))

    chunk_count = max(1, int(np.ceil(total_points / chunk_size)))
    row_ids = np.arange(total_points)
    id_chunks = np.array_split(row_ids, chunk_count)

    left = gpd.GeoDataFrame(
        {"__row_id": row_ids, "geometry": df[geometry].values},
        geometry="geometry",
        crs=df.crs
    )
    right = df_ovgk[[output_column, "geometry"]]

    def process_chunk(ids):
        chunk = left.iloc[ids]
        joined = gpd.sjoin_nearest(chunk, right, how="left")
        joined = joined[["__row_id", output_column]].drop_duplicates("__row_id", keep="first")
        return joined

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(ids)
        for ids in context.progress(id_chunks, total=len(id_chunks), label="Imputing OeV Gueteklasse (parallel)...")
    )

    joined = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["__row_id", output_column])

    out = df.copy()
    out["__row_id"] = row_ids
    out = out.merge(joined, on="__row_id", how="left")
    out = out.drop(columns=["__row_id"])

    out.loc[out[output_column].isna(), output_column] = "None"
    out[output_column] = out[output_column].astype("category")
    return out