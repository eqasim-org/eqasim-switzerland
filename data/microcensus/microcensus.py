import pandas as pd
import numpy as np
import data.utils
import data.constants as c
import data.microcensus.income

def configure(context, require):
    require.stage("data.microcensus.households")
    require.stage("data.microcensus.persons")
    require.stage("data.microcensus.trips")
    # require.cache = False

def execute(context):
    df_mz_persons = context.stage("data.microcensus.persons")
    df_mz_households = context.stage("data.microcensus.households")
    df_mz_trips = context.stage("data.microcensus.trips")

    df_mz = pd.merge(df_mz_persons, df_mz_households)
    df_mz = data.microcensus.income.impute(df_mz)

    # Filter persons for which we do not have sufficient information
    unknown_ids = set(df_mz_trips[
        (df_mz_trips["mode"] == "unknown") | (df_mz_trips["purpose"] == "unknown")
    ]["person_id"])

    df_mz_trips = df_mz_trips[~df_mz_trips["person_id"].isin(unknown_ids)]
    df_mz = df_mz[~df_mz["person_id"].isin(unknown_ids)]

    return df_mz, df_mz_trips
