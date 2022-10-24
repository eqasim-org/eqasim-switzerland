import numpy as np
from sklearn.neighbors import KDTree

import data.constants as c


def configure(context):
    context.stage("data.statpop.persons")


def execute(context):
    df_statpop = context.stage("data.statpop.persons")
    density_coordinates = np.vstack([df_statpop["home_x"], df_statpop["home_y"]]).T
    kd_tree = KDTree(density_coordinates)

    return kd_tree


def impute(kd_tree, df, x="x", y="y", radius=c.POPULATION_DENSITY_RADIUS):
    print("Imputing population density ...")
    coordinates = np.vstack([df[x], df[y]]).T
    counts = kd_tree.query_radius(coordinates, radius, count_only=True)
    df["population_density"] = counts  # / (np.pi * c.POPULATION_DENSITY_RADIUS**2)
