import pandas as pd
import numpy as np
import data.constants as c
import population.algo.hot_deck_matching

def configure(context, require):
    require.config("weekend_scenario", False)
    require.config("hot_deck_matching_runners", -1)
    require.stage("data.microcensus.microcensus")
    require.stage("data.statpop.statpop")

def execute(context):
    df_mz, df_mz_trips = context.stage("data.microcensus.microcensus")
    is_weekend_scenario = context.config["weekend_scenario"]
    hdm_runners = context.config["hot_deck_matching_runners"]

    # Source are the MZ observations, for each STATPOP person, a sample is drawn from there
    df_source = pd.DataFrame(df_mz[
        (is_weekend_scenario & df_mz["weekend"]) # use only weekend samples for a weekend scenario
             |
        (~is_weekend_scenario & ~df_mz["weekend"]) # and only weekday samples for a weekday
    ])

    df_statpop = context.stage("data.statpop.statpop")

    # Match houesholds
    age_selector = df_statpop["age"] >= c.MZ_AGE_THRESHOLD
    head_selector = age_selector & df_statpop["is_head"]

    df_target = pd.DataFrame(df_statpop[head_selector])

    population.algo.hot_deck_matching.run(
        df_target, "person_id",
        df_source, "person_id",
        "household_weight",
        ["age_class", "sex", "marital_status"],
        ["household_size_class"],
        runners = hdm_runners
         #, "home_structure"] TODO: Add this again!!!
    )

    # Remove households with unmatchable heads of household

    unmatchable = head_selector & (df_target["hdm_source_id"] == -1)
    unmatchable_houeshold_ids = np.unique(df_target[unmatchable]["household_id"].values)
    remove = df_statpop["household_id"].isin(unmatchable_houeshold_ids)
    df_statpop = df_statpop.loc[~remove, :]

    print("Umatchable heads of houeshold:", sum(unmatchable))
    print("Removed houesholds:", len(unmatchable_houeshold_ids))
    print("Removed persons:", sum(remove))

    df_target["hdm_source_id"] = df_target["hdm_source_id"].astype(np.int)
    df_source["person_id"] = df_source["person_id"].astype(np.int)

    # Get the attributes from the MZ for the head of houeshold (and thus for the household)
    df_household_attributes = pd.merge(df_target[[
        "household_id", "hdm_source_id"
    ]], df_source[[
        "person_id", "income_class", "number_of_cars_class", "number_of_bikes_class"
    ]], left_on = "hdm_source_id", right_on = "person_id")

    df_household_attributes["mz_head_id"] = df_household_attributes["hdm_source_id"]
    del df_household_attributes["hdm_source_id"]
    del df_household_attributes["person_id"]

    df_statpop = pd.merge(df_statpop, df_household_attributes)

    # Match persons
    age_selector = df_statpop["age"] >= c.MZ_AGE_THRESHOLD
    df_target = pd.DataFrame(df_statpop[age_selector])

    population.algo.hot_deck_matching.run(
        df_target, "person_id",
        df_source, "person_id",
        "person_weight",
        ["age_class", "sex", "marital_status"],
        ["household_size_class", "income_class", "number_of_cars_class", "number_of_bikes_class"],
        runners = hdm_runners
        #, "home_structure"] TODO: Add this again!!!
    )

    # Remove unmatchable persons
    unmatchable = df_target["hdm_source_id"] == -1
    unmatchable_houeshold_ids = np.unique(df_target[unmatchable]["household_id"].values)
    remove = df_statpop["household_id"].isin(unmatchable_houeshold_ids)
    df_statpop = df_statpop.loc[~remove, :]

    print("Umatchable persons:", sum(unmatchable))
    print("Removed persons:", len(unmatchable_houeshold_ids))
    print("Removed persons:", sum(remove))

    df_matching = pd.merge(df_statpop[[
        "person_id", "household_id", "mz_head_id"
    ]], df_target[[
        "person_id", "hdm_source_id"
    ]])

    df_matching["mz_person_id"] = df_matching["hdm_source_id"]
    del df_matching["hdm_source_id"]

    return df_matching
