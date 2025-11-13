import pandas as pd
import numpy as np


MS_REGIONS = {'canton_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], 
              'cluster': [2, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 0]}
MS_REGIONS = pd.DataFrame(MS_REGIONS)

def configure(context):
    context.stage("synthesis.population.enriched")
    context.stage("data.constants")

def execute(context):
    df_persons = context.stage("synthesis.population.enriched")

    # get public transport subscriptions
    df_persons["hasGeneralSubscription"] = df_persons.subscriptions_ga
    df_persons["hasHalbtaxSubscription"] = df_persons.subscriptions_halbtax
    df_persons["hasRegionalSubscription"] = df_persons.subscriptions_verbund | df_persons.subscriptions_strecke
    df_persons["hasJuniorSubscription"] = df_persons.subscriptions_junior
    df_persons["hasGleis7Subscription"] = df_persons.subscriptions_gleis7

    # add the region
    df_persons = df_persons.merge(MS_REGIONS, on="canton_id", how="left")
    df_persons = df_persons.rename(columns={"cluster":"region"})
    assert not df_persons["region"].isna().any(), "Some persons have no region assigned!"

    # add income (income per capita)
    INCOME_CLASS_MAP = {0: 2000, 1: 3000, 2: 4500, 3: 7000, 4: 9000, 5: 11000,  6: 13000, 7: 15000, 8: 17000} # same as cmdp/dmc/data/prepare_survey_data.py
    df_persons["income"] = df_persons.income_class.map(INCOME_CLASS_MAP)
    df_persons["income"] = df_persons["income"] / df_persons["household_size"].fillna(1).clip(lower=1, upper=7)

    # add availabilities
    c = context.stage("data.constants")
    df_persons['car_availability'] = (df_persons['car_availability']!=c.CAR_AVAILABILITY_NEVER).astype(bool)
    df_persons['bike_availability'] = (df_persons['number_of_bikes_class']!=c.BIKE_AVAILABILITY_FOR_NONE).astype(bool)
    df_persons['is_car_passenger'] = df_persons['is_car_passenger'].fillna(False).astype(bool)
    df_persons['driving_license'] = df_persons['driving_license'].fillna(False).astype(bool)

    return df_persons[[
        # basic attributes
        "person_id", "sex", "age", "region", "driving_license", "income",

        # pt subscriptions
        "hasGeneralSubscription", "hasHalbtaxSubscription",
        "hasRegionalSubscription", "hasJuniorSubscription", "hasGleis7Subscription",
        
        # availabilities
        "car_availability", "bike_availability", "is_car_passenger"
    ]]