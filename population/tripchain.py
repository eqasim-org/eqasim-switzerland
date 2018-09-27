import pandas as pd
import numpy as np
import data.constants as c

def configure(context, require):
    require.stage("population.sociodemographics")
    require.stage("data.microcensus.microcensus")

def execute(context):
    df_mz, df_mz_trips = context.stage("data.microcensus.microcensus")
    df_persons = context.stage("population.sociodemographics")

    

    return {}
