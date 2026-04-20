import numpy as np
from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.statpop.persons")


def execute(context):
    df_statpop = context.stage("data.statpop.persons")
    density_coordinates = np.vstack([df_statpop["home_x"], df_statpop["home_y"]]).T
    kd_tree = KDTree(density_coordinates)

    return kd_tree


def impute(context, kd_tree, df, x="x", y="y", radius= 2.5 * 1e3, point_type="", chunk_size=1e6):
    logger.info("Imputing population density within %d m of %d %s coordinates...", radius, len(df), point_type)
    counts = []
    chunk_count = max(1, int(len(df) / chunk_size))
    for chunk in context.progress(np.array_split(df, chunk_count), 
                                  total=chunk_count,
                                  label="Imputing population density..."):
        
        coordinates = np.vstack([chunk[x], chunk[y]]).T
        counts.extend(kd_tree.query_radius(coordinates, radius, count_only=True))
    
    df["population_density"] = counts # / (np.pi * c.POPULATION_DENSITY_RADIUS**2)
    return df

def impute_parallel(context, kd_tree, df, x="x", y="y", radius=2.5 * 1e3, point_type="", chunk_size=1000, n_jobs=10):

    total_points = len(df)
    logger.info("Imputing population density within %d m of %d %s coordinates...", radius, total_points, point_type)

    # Split DataFrame into roughly equal chunks
    chunk_count = max(1, int(np.ceil(total_points / chunk_size)))
    df_splits = np.array_split(df, chunk_count)

    def process_chunk(chunk):
        coords = np.vstack([chunk[x], chunk[y]]).T
        return kd_tree.query_radius(coords, radius, count_only=True)

    # Run in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk)
        for chunk in context.progress(df_splits, total=chunk_count, label="Imputing population density...")
    )

    # Flatten list of arrays
    counts = np.concatenate(results)
    df["population_density"] = counts
    return df
