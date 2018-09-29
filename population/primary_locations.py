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
    df_commute = context.stage("population.commute")
    pdf_matrices, cdf_matrices, unique_zone_ids = context.stage("data.od.matrix")
    df_persons = context.stage("population.sociodemographics")

    # WIP

    return {}
