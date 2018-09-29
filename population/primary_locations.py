import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.stage("population.commute")
    require.stage("data.od.matrix")
    require.stage("population.sociodemographics")
    require.stage("data.statpop.spatial_structure")

# TODO: We only assign work here through OD matrices. However, we *can* generate
# OD matrices for education as well (the STATPOP information is available). What
# would need to be done is to adjust data.od.matrix to produce two kinds of
# matrices and then we would need to use this information here. In population.commute
# we already produce information on education commute.
# Shouldn't DRIVING LICENSE even be a required attribute (rather than one that is
# only preferred for the matching)? The way it is now activity chains with "car"
# can be matched to people who don't have a license. On the other hand, we need
# to revise how we handle car. In MZ is believe it can also mean that the person
# is only a passenger. Not sure if we have information in there whether the person
# is a driver? Then it would be easy to set up another mode "ride" directly from
# the MZ. Now we may convert "car" to "ride" later on when there is a person with
# a car trip but without a license or a car in the household. However, this had
# not been done in Kirills version.

# However, for now we will recover the simple scheme from Kirill!

def execute(context):
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
