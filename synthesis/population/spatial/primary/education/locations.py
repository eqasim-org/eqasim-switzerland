import numpy as np
from sklearn.neighbors import KDTree
import math

import data.spatial.utils as spatial_utils


def configure(context):
    context.stage("data.statent.statent")
    context.stage("synthesis.population.SNN_population")
    context.config("random_seed")


# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In data.microcensus.commute
# we already produce information on education commute.

def execute(context):
    df_persons = context.stage("synthesis.population.SNN_population")

    df_statent = context.stage("data.statent.statent")
    df_statent = df_statent[~df_statent["education_type"].isna()]

    age_bounds = [(-np.inf, 6), (6, 12), (12, 15), (15, 19), (19, np.inf)]
    education_types = ["kindergarten", "primary", "secondary1", "secondary2", "tertiary"]
    query_sizes = (1, 1, 5, 5, 10)
    
    # Set up RNG
    rng = np.random.RandomState(context.config("random_seed"))


    for (lower_bound, upper_bound), type_educ, query_size in zip(age_bounds, education_types, query_sizes):
        f_persons = (df_persons["age"] > lower_bound) & (df_persons["age"] <= upper_bound)
        df_candidates = df_statent[df_statent["education_type"] == type_educ]

        education_coordinates = np.vstack([df_candidates["x"], df_candidates["y"]]).T
        home_coordinates = np.vstack([df_persons.loc[f_persons, "home_x"], df_persons.loc[f_persons, "home_y"]]).T

        tree = KDTree(education_coordinates)
        distances, indices = tree.query(home_coordinates, query_size, return_distance=True)
        
        selector = rng.randint(query_size, size=(indices.shape[0],))
        indices  = np.choose(selector, indices.T)

        df_persons.loc[f_persons, "education_x"] = df_candidates.iloc[indices]["x"].values
        df_persons.loc[f_persons, "education_y"] = df_candidates.iloc[indices]["y"].values
        df_persons.loc[f_persons, "education_location_id"] = df_candidates.iloc[indices]["enterprise_id"].values

        print("  %s (%d persons, %d locations)" % (type_educ, np.count_nonzero(f_persons), len(df_candidates)))

    df_persons = df_persons[["person_id",
                             "education_x", "education_y",
                             "education_location_id"]].rename({"education_location_id": "destination_id",
                                                               "education_x": "x",
                                                               "education_y": "y"},
                                                              axis=1)

    df_persons = spatial_utils.to_gpd(context, df_persons, coord_type="education")

    return df_persons[["person_id", "destination_id", "geometry"]]
