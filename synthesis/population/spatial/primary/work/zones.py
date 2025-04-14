import functools

import numpy as np
import pandas as pd


def configure(context):
    context.stage("data.microcensus.commute")
    context.stage("data.od.matrix")
    context.stage("data.od.distances")
    context.stage("data.spatial.zones")
    context.stage("synthesis.population.enriched")
    
    context.config("random_seed")


# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In data.microcensus.commute
# we already produce information on education commute.

def execute(context):
    df_zones = context.stage("data.spatial.zones")

    # Load commute information for work
    commute = context.stage("data.microcensus.commute")
    df_commute = commute["work"][["person_id", "commute_mode", "commute_home_distance"]]
    df_commute = df_commute.rename({"person_id": "mz_person_id"}, axis=1)

    # Load person information
    df_persons = context.stage("synthesis.population.enriched")[["person_id", "household_id",
                                                                 "mz_person_id", "home_zone_id"]]

    # Merge commute information into the persons
    df = pd.merge(df_persons, df_commute, on="mz_person_id")
    
    # Set up RNG
    rng = np.random.RandomState(context.config("random_seed"))

    df_demand = df.groupby(["commute_mode", "home_zone_id"]).size().reset_index(name="count")
    pdf_matrices, cdf_matrices = context.stage("data.od.matrix")
    commute_counts = {}

    print("Computing commute counts ...")
    for mode in ["car", "pt", "bike", "walk", "car_passenger"]:
        source_mode = "car" if mode == "car_passenger" else mode

        origin_counts = np.array([
            np.sum(df_demand.loc[
                       (df_demand["commute_mode"] == mode) & (df_demand["home_zone_id"] == origin_zone), "count"
                   ]) for origin_zone in context.progress(df_zones["zone_id"], label=mode)
        ])[:, np.newaxis]

        counts = np.zeros(pdf_matrices[source_mode].shape, dtype=np.int)

        for i in range(len(df_zones)):
            if origin_counts[i] > 0:
                assert (~np.any(np.isnan(pdf_matrices[source_mode][i])))
                counts[i, :] = rng.multinomial(n=origin_counts[i], pvals=pdf_matrices[source_mode][i, :])

        commute_counts[mode] = counts
        assert (len(counts) == len(df_zones))

    distances = context.stage("data.od.distances")
    work_zones = np.zeros((len(df),), dtype=np.int)
    zone_ids = list(df_zones["zone_id"])

    with context.progress(label="Assigning work zones", total=5 * len(df_zones)) as progress:
        commute_mode_values = df["commute_mode"].values
        home_zone_values = df["home_zone_id"].values
        commute_distances = df["commute_home_distance"].values

        for mode in ["car", "pt", "bike", "walk", "car_passenger"]:
            mode_f = (commute_mode_values == mode)
            source_mode = "car" if mode == "car_passenger" else mode

            for origin_index, origin_zone in enumerate(zone_ids):
                destination_counts = commute_counts[mode][origin_index, :]
                destination_order = np.argsort(distances[origin_index, :])

                # Sort destination_counts and corresponding zone_ids
                sorted_counts = destination_counts[destination_order]
                sorted_zone_ids = [zone_ids[i] for i in destination_order]

                # Efficient flat list of destinations
                destinations = np.repeat(sorted_zone_ids, sorted_counts)

                if destinations.size > 0:
                    f = mode_f & (home_zone_values == origin_zone)
                    person_indices = np.where(f)[0]

                    if person_indices.size > 0:
                        commute_home_dists = commute_distances[person_indices]
                        person_order = np.argsort(commute_home_dists)

                        ordered_indices = person_indices[person_order]
                        work_zones[ordered_indices] = destinations

                progress.update()


    df.loc[:, "work_zone_id"] = work_zones
    df = df[["person_id", "work_zone_id", "commute_mode"]]
    assert (len(df) == len(df.dropna()))
    return df
