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

    # Remove and track unmatchable houesholds (i.e. head of household)

    initial_statpop_length = len(df_statpop)
    initial_target_length = len(df_target)

    unmatchable_household_selector = df_target["hdm_source_id"] == -1
    umatchable_household_ids = set(df_target.loc[unmatchable_household_selector, "household_id"].values)
    unmatchable_person_selector = df_statpop["household_id"].isin(umatchable_household_ids)

    removed_ids = set(df_target.loc[unmatchable_person_selector, "person_id"].values)

    df_target = df_target.loc[~unmatchable_household_selector, :]
    df_statpop = df_statpop.loc[~unmatchable_person_selector, :]

    removed_houesholds_count = sum(unmatchable_household_selector)
    removed_persons_count = sum(unmatchable_person_selector)

    print("Unmatchable heads of household: ", removed_houesholds_count)
    print("  Removed households: ", removed_houesholds_count)
    print("  Removed persons: ", removed_persons_count)

    assert(len(df_target) == initial_target_length - removed_houesholds_count)
    assert(len(df_statpop) == initial_statpop_length - removed_persons_count)

    # Convert IDs
    df_target["hdm_source_id"] = df_target["hdm_source_id"].astype(np.int)
    df_source["person_id"] = df_source["person_id"].astype(np.int)

    # Get the attributes from the MZ for the head of houeshold (and thus for the household)
    df_attributes = pd.merge(
        df_target[[
            "household_id", "hdm_source_id"
        ]],
        df_source[[
            "person_id", "income_class", "number_of_cars_class", "number_of_bikes_class"
        ]],
        left_on = "hdm_source_id", right_on = "person_id"
    )

    df_attributes["mz_head_id"] = df_attributes["hdm_source_id"]
    del df_attributes["hdm_source_id"]
    del df_attributes["person_id"]

    assert(len(df_attributes) == len(df_target))

    # Attach attrbiutes to STATPOP for the second matching

    initial_statpop_size = len(df_statpop)

    df_statpop = pd.merge(
        df_statpop, df_attributes, on = "household_id"
    )

    assert(len(df_statpop) == initial_statpop_size)

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

    # Remove and track unmatchable persons

    initial_statpop_length = len(df_statpop)
    initial_target_length = len(df_target)

    unmatchable_person_selector = df_target["hdm_source_id"] == -1
    umatchable_household_ids = set(df_target.loc[unmatchable_person_selector, "household_id"].values)
    unmatchable_member_selector = df_statpop["household_id"].isin(umatchable_household_ids)

    removed_ids |= set(df_statpop.loc[unmatchable_member_selector, "person_id"].values)

    df_target = df_target.loc[~unmatchable_person_selector, :]
    df_statpop = df_statpop.loc[~unmatchable_member_selector, :]

    removed_persons_count = sum(unmatchable_person_selector)
    removed_houesholds_count = len(umatchable_household_ids)
    removed_members_count = sum(unmatchable_member_selector)

    print("Unmatchable persons: ", removed_persons_count)
    print("  Removed households: ", removed_houesholds_count)
    print("  Removed household members: ", removed_members_count)

    assert(len(df_target) == initial_target_length - removed_persons_count)
    assert(len(df_statpop) == initial_statpop_length - removed_members_count)

    # Extract only the matching information

    df_matching = pd.merge(
        df_statpop[[ "person_id", "household_id", "mz_head_id" ]],
        df_target[[ "person_id", "hdm_source_id" ]],
        on = "person_id", how = "left")

    df_matching["mz_person_id"] = df_matching["hdm_source_id"]
    del df_matching["hdm_source_id"]

    assert(len(df_matching) == len(df_statpop))

    # Check that all person who don't have a MZ id now are under age
    assert(np.all(df_statpop[
        df_statpop["person_id"].isin(
            df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
        )
    ]["age"] < c.MZ_AGE_THRESHOLD))

    assert(not np.any(df_matching["mz_head_id"] == -1))

    # Return
    return df_matching, removed_ids
