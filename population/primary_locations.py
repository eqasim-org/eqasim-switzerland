import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.stage("population.commute")
    require.stage("data.od.matrix")
    require.stage("population.sociodemographics")

# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In population.commute
# we already produce information on education commute.

# However, for now we will recover the simple scheme from Kirill!

def execute(context):
    exit()
    #df_commute = context.stage("population.commute")
    #pdf_matrices, cdf_matrices, unique_zone_ids = context.stage("data.od.matrix")

    # We don't need all the data, so let's do some filtering already
    df_spatial = context.stage("data.statpop.spatial_structure")[[
        "household_id", "zone"
    ]]

    df_persons = context.stage("population.sociodemographics")[[
        "person_id", "household_id", "mz_person_id"
    ]]

    df_commute = context.stage("population.commute")[[
        "person_id", "commute_mode", "commute_home_distance", "commute_purpose"
    ]]
    df_commute = df_commute[df_commute["commute_purpose"] == "work"]
    df_commute["mz_person_id"] = df_commute["person_id"]
    del df_commute["person_id"]

    # Merge commute information into the persons
    df = pd.merge(
        df_persons, df_commute, on = "mz_person_id"
    )

    # Merge home zone information into the persons
    df = pd.merge(
        df, df_spatial, on = "household_id"
    )

    print(df)

    return {}
