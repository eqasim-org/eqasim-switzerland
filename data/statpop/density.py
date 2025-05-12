import numpy as np
from sklearn.neighbors import KDTree

def configure(context):
    context.stage("data.statpop.persons")


def execute(context):
    df_statpop = context.stage("data.statpop.persons")
    density_coordinates = np.vstack([df_statpop["home_x"], df_statpop["home_y"]]).T
    kd_tree = KDTree(density_coordinates)

    return kd_tree


def impute(context, kd_tree, df, x="x", y="y", radius= 2.5 * 1e3, point_type="", chunk_size=1e6):
    print(f"Imputing population density within {radius} m of {len(df)} {point_type} coordinates...")
    counts = []
    chunk_count = max(1, int(len(df) / chunk_size))
    for chunk in context.progress(np.array_split(df, chunk_count), 
                                  total=chunk_count,
                                  label="Imputing population density..."):
        
        coordinates = np.vstack([chunk[x], chunk[y]]).T
        counts.extend(kd_tree.query_radius(coordinates, radius, count_only=True))
    
    df["population_density"] = counts # / (np.pi * c.POPULATION_DENSITY_RADIUS**2)
