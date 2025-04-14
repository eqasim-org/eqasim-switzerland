import numpy as np
import pandas as pd

import data.constants as c

"""
This stage fuses sampled STATPOP data with microcensus data.
"""


def configure(context):
    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.sampled")
    context.stage("data.microcensus.persons")


def execute(context):
    df_matched, unmatched_ids = context.stage("synthesis.population.matched")
    df_sampled = context.stage("synthesis.population.sampled")
    df_mz = context.stage("data.microcensus.persons")

    assert (len(df_matched) == len(df_sampled) - len(unmatched_ids))

    # Attach matching information
    df_persons = pd.merge(df_sampled, df_matched, on=["person_id", "household_id"])

    # Attach household attributes through head of household
    df_mz["mz_head_id"] = df_mz[["person_id"]]
    df_persons = pd.merge(df_persons,
                          df_mz[["mz_head_id", "income_class", "number_of_cars_class", "number_of_bikes_class"]],
                          on="mz_head_id")

    # Attach person attributes
    df_mz["mz_person_id"] = df_mz[["person_id"]]
    df_persons = pd.merge(df_persons,
                          df_mz[["mz_person_id", "driving_license", "car_availability", "employed",
                                 "subscriptions_ga",
                                 "subscriptions_halbtax",
                                 "subscriptions_verbund",
                                 "subscriptions_strecke",
                                 "subscriptions_gleis7",
                                 "subscriptions_junior",
                                 "subscriptions_other",
                                 "subscriptions_ga_class",
                                 "subscriptions_verbund_class",
                                 "subscriptions_strecke_class",
                                 "is_car_passenger"]],
                          on="mz_person_id", how="left"
                          )

    # Reset children
    children_selector = df_persons["age"] < c.MZ_AGE_THRESHOLD
    df_persons.loc[children_selector, "driving_license"] = False
    df_persons.loc[children_selector, "employed"] = False
    df_persons.loc[children_selector, "marital_status"] = c.MARITAL_STATUS_SINGLE
    df_persons.loc[children_selector, "car_availability"] = c.CAR_AVAILABILITY_NEVER

    # Make sure we have now NaNs included (commented out, because home_quater_id MAY be NaN deliberately)
    # assert(len(df_persons.drop(["mz_person_id", "mz_head_id"], axis = 1).dropna()) == len(df_matching))

    # Make sure all mz_id == NaN are agents under threshold age
    assert (np.sum(df_persons[df_persons["mz_person_id"].isna()]["age"] >= c.MZ_AGE_THRESHOLD) == 0)

    # Set mz_person_id == NaN to -1 and format ids to int
    df_persons["mz_person_id"] = df_persons["mz_person_id"].fillna(-1).astype(int)
    df_persons["mz_head_id"] = df_persons["mz_head_id"].fillna(-1).astype(int)

    return df_persons
