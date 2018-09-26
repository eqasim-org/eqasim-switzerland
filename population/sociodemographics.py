import pandas as pd
import numpy as np
import data.constants as c
import population.hot_deck_matching

def configure(context, require):
    require.stage("population.sociodemographics")

def execute(context):
    df_matching = context.stage("population.matching")

    print(df_matching)

    return {}
