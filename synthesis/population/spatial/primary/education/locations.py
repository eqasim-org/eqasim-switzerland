import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import data.spatial.utils as spatial_utils
import data.utils
import logging

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.statent.statent")
    context.stage("synthesis.population.enriched")
    context.stage("data.constants")
    context.config("random_seed")
    context.stage("data.microcensus.commute")


# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In data.microcensus.commute
# we already produce information on education commute.

STANDARD_DEVIATION_DISTANCE = 200  # meters (radius of 600 m for 99.7% of distribution)
MIN_PROB = 1e-4
def get_distance_weight(dx):
    # Gaussian around 0 with adaptive std for numerical stability
    std = max(STANDARD_DEVIATION_DISTANCE, float(np.min(dx)) / 2.0 if dx.size else STANDARD_DEVIATION_DISTANCE)
    coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
    return np.maximum(coef * np.exp(-0.5 * (dx / std) ** 2), MIN_PROB)

def execute(context):
    df_persons = context.stage("synthesis.population.enriched")
    commute    = context.stage("data.microcensus.commute")["education"]
    df_statent = context.stage("data.statent.statent")
    c          = context.stage("data.constants")
    
    # Merge commute information into the persons
    df_commute = commute[["person_id", "commute_mode", "commute_home_distance"]]
    df_commute = df_commute.rename({"person_id": "mz_person_id"}, axis=1)
    df_persons = pd.merge(df_persons, df_commute, on="mz_person_id")

    # Filter out locations without education type
    df_statent = df_statent[~df_statent["education_type"].isna()]

    # Prepare filters and education types
    filters_persons, filters_locations, education_types, query_sizes = data.utils.prepare_education_locations(df_persons, df_statent, c)
    
    # Set up RNG
    rng = np.random.RandomState(context.config("random_seed"))

    for person_selector, location_selector, type, query_size in zip(filters_persons, filters_locations, education_types, query_sizes):
        f_persons = person_selector#(df_persons["age"] > lower_bound) & (df_persons["age"] <= upper_bound)
        df_candidates = df_statent[location_selector]#[df_statent["education_type"] == type]

        education_coordinates = np.vstack([df_candidates["x"], df_candidates["y"]]).T
        home_coordinates = np.vstack([df_persons.loc[f_persons, "home_x"], df_persons.loc[f_persons, "home_y"]]).T

        tree = KDTree(education_coordinates)
        distances, indices = tree.query(home_coordinates, query_size, return_distance=True)
        
        # Build probabilities
        mz_distances = df_persons.loc[f_persons, "commute_home_distance"].values
        weight = df_candidates["number_employees"].values # the higher is the number of employees, the more likely the education location has higher capacity
        
        q_low, q_high = np.quantile(weight, [0.25, 0.75]) # clip it here to avoid having too large differences in weights, thus this would bias the assignment too much
        weight = np.clip(weight, q_low, q_high) if q_low < q_high else weight
        weight = np.maximum(weight / weight.sum(), MIN_PROB)

        distance_weight  = get_distance_weight(distances - mz_distances[:, np.newaxis])
        candidate_weight = weight[indices]
        probabilities    = (distance_weight * candidate_weight)
        probabilities   /= np.sum(probabilities, axis=1)[:, np.newaxis]

        # selection
        picks = np.array([rng.multinomial(1, row).argmax() for row in probabilities])
        indices = indices[np.arange(indices.shape[0]), picks]
        
        # Assign education location
        df_persons.loc[f_persons, "education_x"] = df_candidates.iloc[indices]["x"].values
        df_persons.loc[f_persons, "education_y"] = df_candidates.iloc[indices]["y"].values
        df_persons.loc[f_persons, "education_location_id"] = df_candidates.iloc[indices]["enterprise_id"].values

        logger.info("  %s (%d persons, %d locations)", type, np.count_nonzero(f_persons), len(df_candidates))

    df_persons = df_persons[["person_id",
                             "education_x", "education_y",
                             "education_location_id"]].rename({"education_location_id": "destination_id",
                                                               "education_x": "x",
                                                               "education_y": "y"},
                                                              axis=1)

    df_persons = spatial_utils.to_gpd(context, df_persons, coord_type="education")

    return df_persons[["person_id", "destination_id", "geometry"]]
