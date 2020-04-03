import pandas as pd
import numpy as np
import data.constants as c
from tqdm import tqdm
import functools

def configure(context):
    context.stage("data.microcensus.commute")
    context.stage("data.od.matrix")
    context.stage("data.od.distances")
    context.stage("population.sociodemographics")
    context.stage("data.spatial.zones")

# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In data.microcensus.commute
# we already produce information on education commute.

# However, for now we will recover the simple scheme from Kirill!

def execute(context):
    df = context.stage("population.sociodemographics")
    df_zones = context.stage("data.spatial.zones")

    # Load commute information for work
    df_commute = pd.DataFrame(context.stage("data.microcensus.commute")[[
        "person_id", "commute_mode", "commute_home_distance", "commute_purpose"
    ]], copy = True)
    df_commute = df_commute[df_commute["commute_purpose"] == "work"]
    df_commute["mz_person_id"] = df_commute["person_id"]
    del df_commute["person_id"]

    # Load person information
    df_persons = context.stage("population.sociodemographics")[[
        "person_id", "household_id", "mz_person_id", "home_zone_id"
    ]]

    # Merge commute information into the persons
    df = pd.merge(
        df_persons, df_commute, on = "mz_person_id"
    )

    df_demand = df.groupby(["commute_mode", "home_zone_id"]).size().reset_index(name = "count")
    pdf_matrices, cdf_matrices = context.stage("data.od.matrix")
    commute_counts = {}

    print("Computing commute counts ...")
    for mode in ["car", "pt", "bike", "walk", "car_passenger"]:
        source_mode = "car" if mode == "car_passenger" else mode

        origin_counts = np.array([
            np.sum(df_demand.loc[
                (df_demand["commute_mode"] == mode) & (df_demand["home_zone_id"] == origin_zone), "count"
            ]) for origin_zone in tqdm(df_zones["zone_id"], desc = mode)
        ])[:, np.newaxis]

        counts = np.zeros(pdf_matrices[source_mode].shape, dtype = np.int)

        for i in range(len(df_zones)):
            if origin_counts[i] > 0:
                assert(~np.any(np.isnan(pdf_matrices[source_mode][i])))
                counts[i,:] = np.random.multinomial(origin_counts[i], pdf_matrices[source_mode][i,:])

        commute_counts[mode] = counts
        assert(len(counts) == len(df_zones))

    distances = context.stage("data.od.distances")
    work_zones = np.zeros((len(df),), dtype = np.int)
    zone_ids = list(df_zones["zone_id"])

    with tqdm(desc = "Assigning work zones", total = 5 * len(df_zones)) as progress:
        for mode in ["car", "pt", "bike", "walk", "car_passenger"]:
            mode_f = df["commute_mode"] == mode

            for origin_index, origin_zone in enumerate(zone_ids):
                destination_counts = commute_counts[mode][origin_index,:]
                destination_order = np.argsort(distances[origin_index,:])
                destinations = [[zone_ids[i]] * destination_counts[i] for i in destination_order]
                destinations = functools.reduce(lambda x, y: x + y, destinations)

                if len(destinations) > 0:
                    f = mode_f & (df["home_zone_id"] == origin_zone)
                    person_indices = np.where(f)[0]
                    person_order = np.argsort(df[f]["commute_home_distance"])
                    work_zones[person_indices[person_order]] = destinations

                progress.update()

    df.loc[:, "work_zone_id"] = work_zones
    df = df[["person_id", "work_zone_id", "commute_mode"]]
    assert(len(df) == len(df.dropna()))

    return df
