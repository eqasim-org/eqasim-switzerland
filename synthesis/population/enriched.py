import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")
"""
This stage fuses sampled census data with microcensus data.
"""


def configure(context):
    context.stage("synthesis.population.matched")
    context.stage("synthesis.population.sampled")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")

def execute(context):
    df_matched, unmatched_ids = context.stage("synthesis.population.matched")
    df_sampled                = context.stage("synthesis.population.sampled")
    df_mz                     = context.stage("data.microcensus.persons")
    c                         = context.stage("data.constants")

    assert (len(df_matched) == len(df_sampled) - len(unmatched_ids))

    if c.census == "statpop":

        # Attach matching information
        df_persons = pd.merge(df_sampled, df_matched, on=["person_id", "household_id"])
        # Attach household attributes through head of household
        # df_mz["mz_head_id"] = df_mz[["person_id"]]
        # df_persons = pd.merge(df_persons,
        #                     df_mz[["mz_head_id", "number_of_cars_class", "number_of_bikes_class"]],
        #                     on="mz_head_id")

        # Attach person attributes
        df_mz["mz_person_id"] = df_mz[["person_id"]]
        df_persons = pd.merge(df_persons,
                            df_mz[["mz_person_id",
                                    "bike_availability", "is_car_passenger"]],
                            on="mz_person_id", how="left"
                            )
        # recode bike availability to two values:
        var_raw = pd.to_numeric(df_persons["bike_availability"], errors="coerce")
        df_persons["bike_availability"] = np.where(var_raw == c.BIKE_AVAILABILITY_NEVER, 0, 1).astype("int64")
        # Reset children
        children_selector = df_persons["age"] < c.MZ_AGE_THRESHOLD
        df_persons.loc[children_selector, "driving_license"]  = False
        #df_persons.loc[children_selector, "employed"]         = False
        df_persons.loc[children_selector, "marital_status"]   = c.MARITAL_STATUS_SINGLE
        df_persons.loc[children_selector, "car_availability"] = 0
        df_persons.loc[children_selector, "bike_availability"] = 0

        # # Make sure we have now NaNs included (commented out, because home_quater_id MAY be NaN deliberately)
        # # assert(len(df_persons.drop(["mz_person_id", "mz_head_id"], axis = 1).dropna()) == len(df_matching))

        # # Make sure all mz_id == NaN are agents under threshold age
        # assert (np.sum(df_persons[df_persons["mz_person_id"].isna()]["age"] >= c.MZ_AGE_THRESHOLD) == 0)

        # # Set mz_person_id == NaN to -1 and format ids to int
        # df_persons["mz_person_id"] = df_persons["mz_person_id"].fillna(-1).astype(int)
        # df_persons["mz_head_id"] = df_persons["mz_head_id"].fillna(-1).astype(int)

        # # Clean driving license attribute
        # N_underage_driving = len(df_persons[(df_persons["age"]<18) & (df_persons["driving_license"])])
        # print(f"Identified {N_underage_driving} agents under 18 years but having a driving license.")
        # print("This is due to statistical matching - those agents were not matched using the age variable.")
        # print("Fixing this to ensure consistency of the results.")
        # df_persons.loc[df_persons["age"]<18, "driving_license"] = False

    elif c.census == "are_synpop":

        # Attach matching information
        df_persons = pd.merge(df_sampled, df_matched, on=["person_id"])

        # Attach person attributes
        df_mz["mz_person_id"] = df_mz[["person_id"]]
        df_persons = pd.merge(df_persons,
                            df_mz[["mz_person_id",
                                   "income_class", "age", "number_of_bikes_class",
                                    "car_availability", 
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
        children_selector = df_persons["age_class"] == 0
        df_persons.loc[children_selector, "driving_license"] = False
        df_persons.loc[children_selector, "employment_status"] = "student"
        df_persons.loc[children_selector, "car_availability"] = 0

        # Filling those with NA income class with 0
        df_persons.loc[df_persons["income_class"].isna(), "income_class"] = 0
        df_persons.loc[df_persons["age"].isna(), "age"]                   = 5 # TODO check why/how some agents can have NaN age

        # Filling missing IDs
        df_persons["statpop_person_id"]    = df_persons["synpop_person_id"]
        df_persons["statpop_household_id"] = df_persons["synpop_person_id"]
        df_persons["mz_head_id"]           = df_persons["mz_person_id"].fillna(0).astype(int)
        df_persons["age"]                  = df_persons["age"].fillna(5).astype(int)

        # Make sure we have now NaNs included (commented out, because home_quater_id MAY be NaN deliberately)
        # assert(len(df_persons.drop(["mz_person_id", "mz_head_id"], axis = 1).dropna()) == len(df_matching))

        # Make sure all mz_id == NaN are agents under threshold age
        assert (np.sum(df_persons[df_persons["mz_person_id"].isna()]["age_class"] >= 1) == 0)

        # Set mz_person_id == NaN to -1 and format ids to int
        df_persons["mz_person_id"] = df_persons["mz_person_id"].fillna(-1).astype(int)

        # Clean driving license attribute
        N_underage_driving = len(df_persons[(df_persons["age_class"]<=1) & (df_persons["driving_license"])])
        logger.info("Identified %d agents under 18 years but having a driving license.", N_underage_driving)
        logger.info("This is due to statistical matching - those agents were not matched using the age variable.")
        logger.info("Fixing this to ensure consistency of the results.")
        df_persons.loc[df_persons["age_class"]<=1, "driving_license"] = False

    #print(df_persons["collective_housing_resident"].value_counts())
    return df_persons
