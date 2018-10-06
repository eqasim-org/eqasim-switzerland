import pandas as pd
import numpy as np
from tqdm import tqdm
import data.spatial.zone_shapes
from sklearn.neighbors import KDTree
import data.constants as c

def configure(context, require):
    require.stage("data.statent.statent")
    require.stage("population.commute")
    require.stage("population.sociodemographics")
    require.stage("population.matching")

def execute(context):
    df_commute = pd.DataFrame(context.stage("population.commute"), copy = True)
    df_commute = df_commute[df_commute["commute_purpose"] == "education"]
    df_commute["mz_person_id"] = df_commute["person_id"]
    del df_commute["person_id"]

    df_persons = context.stage("population.sociodemographics")
    df_underage = df_persons[df_persons["age"] < c.MZ_AGE_THRESHOLD]

    df_persons = pd.merge(
        df_persons[df_persons["age"] >= c.MZ_AGE_THRESHOLD],
        df_commute, on = "mz_person_id")

    df_persons = pd.concat([
        df_persons[["person_id", "age", "home_x", "home_y"]],
        df_underage[["person_id", "age", "home_x", "home_y"]]
    ])

    df_statent = context.stage("data.statent.statent")
    df_statent = df_statent[~df_statent["education_type"].isna()]

    age_bounds = [(-np.inf, 6), (6, 12), (12, 16), (16, np.inf)]
    education_types = ["kindergarten", "primary", "secondary", "tertiary"]
    query_sizes = (1, 1, 5, 10)

    for (lower_bound, upper_bound), type, query_size in zip(age_bounds, education_types, query_sizes):
        f_persons = (df_persons["age"] > lower_bound) & (df_persons["age"] <= upper_bound)
        f_statent = df_statent["education_type"] == type

        education_coordinates = list(zip(df_statent.loc[f_statent, "x"], df_statent.loc[f_statent, "y"]))
        home_coordinates = list(zip(df_persons.loc[f_persons, "home_x"], df_persons.loc[f_persons, "home_y"]))

        tree = KDTree(education_coordinates)
        indices = tree.query(home_coordinates, query_size, return_distance = False)
        selector = np.random.randint(query_size, size = (indices.shape[0],))
        indices = np.choose(selector, indices.T)

        df_persons.loc[f_persons, "education_x"] = df_statent.iloc[indices]["x"].values
        df_persons.loc[f_persons, "education_y"] = df_statent.iloc[indices]["y"].values
        df_persons.loc[f_persons, "education_enterprise_id"] = df_statent.iloc[indices]["enterprise_id"].values

        print("  %s (%d persons, %d locations)" % (type, np.count_nonzero(f_persons), np.count_nonzero(f_statent)))

    return df_persons
