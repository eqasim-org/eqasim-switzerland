import pandas as pd
import numpy as np
from mode_choice.dmc_defaults import Defaults


MS_REGIONS = Defaults.MS_REGIONS.reset_index().copy()
INCOME_CLASS_MAP = Defaults.INCOME_CLASS_MAP

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
    df_persons["income"] = df_persons.income_class.map(INCOME_CLASS_MAP)
    
    # Calculate income per capita using the OECD equivalence scale: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Glossary:Equivalised_income
    df_persons['is_child'] = df_persons['age'] < 14
    num_children = df_persons.groupby('household_id')['is_child'].transform('sum')
    num_adults = df_persons['household_size'] - num_children
    assert (num_adults >= 1).all(), "All households should have at least one adult."
    equvalent_size =  1 + 0.5 * (num_adults - 1) + 0.3 * num_children
    df_persons["income"] = df_persons["income"] / equvalent_size

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