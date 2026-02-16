import itertools

import numba
import numpy as np
import pandas as pd

"""
This stage attaches observations from the microcensus to the synthetic population sample.
This is done by statistical matching. Here, a recursive version of statistical matching is implemented.
It progressively decreases the minimum number of observations to ensure the most important attributes are always matched.
"""

def configure(context):
    context.config("hot_deck_matching_runners")
    context.config("random_seed")
    context.config("matching_minimum_observations", 20)
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.activity_chains")
    context.stage("synthesis.population.sampled")
    context.stage("data.constants")


@numba.jit(nopython=True, parallel=True)
def sample_indices(uniform, cdf, selected_indices):
    indices = np.arange(len(uniform))

    for i, u in enumerate(uniform):
        indices[i] = np.count_nonzero(cdf < u)

    return selected_indices[indices]


def decrease_minimum_observation(N):
    # Any decreasing function can be implemented here.
    return N-1


def is_left_slice(list1, list2):
    return list1[:len(list2)] == list2


def recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
                         rng=None, minimum_observations=0):
    
    # Reduce data frames
    df_source = df_source[[source_identifier, weight] + columns].copy()
    df_target = df_target[[target_identifier] + columns].copy()

    # Sort data frames
    df_source = df_source.sort_values(by=columns)
    df_target = df_target.sort_values(by=columns)

    # Find unique values for all columns
    unique_values = {}

    for column in columns:
        unique_values[column] = list(sorted(set(df_source[column].unique()) | set(df_target[column].unique())))

    # Generate filters for all columns and values
    source_filters, target_filters = {}, {}

    for column, column_unique_values in unique_values.items():
        source_filters[column] = [df_source[column].values == value for value in column_unique_values]
        target_filters[column] = [df_target[column].values == value for value in column_unique_values]

    # Define search order
    source_filters = [source_filters[column] for column in columns]
    target_filters = [target_filters[column] for column in columns]

    # Perform matching
    weights = df_source[weight].values
    assigned_indices = np.ones((len(df_target),), dtype=np.int) * -1
    unassigned_mask = np.ones((len(df_target),), dtype=np.bool)
    assigned_levels = np.ones((len(df_target),), dtype=np.int) * -1
    uniform = rng.random_sample(size=(len(df_target),))

    column_indices = [np.arange(len(unique_values[column])) for column in columns]

    if mandatory_columns:
        minimum_level = len(mandatory_columns)
    else:
        minimum_level = 1

    for level in range(minimum_level, len(column_indices) + 1)[::-1]:
        level_column_indices = column_indices[:level]
        
        if np.count_nonzero(unassigned_mask) > 0:
            for column_index in itertools.product(*level_column_indices):
                f_source = np.logical_and.reduce([source_filters[i][k] for i, k in enumerate(column_index)])
                f_target = np.logical_and.reduce(
                    [target_filters[i][k] for i, k in enumerate(column_index)] + [unassigned_mask])

                selected_indices = np.nonzero(f_source)[0]
                requested_samples = np.count_nonzero(f_target)

                if requested_samples == 0:
                    continue

                if len(selected_indices) < minimum_observations:
                    continue

                selected_weights = weights[f_source]
                cdf = np.cumsum(selected_weights)
                cdf /= cdf[-1]

                assigned_indices[f_target] = sample_indices(uniform[f_target], cdf, selected_indices)
                assigned_levels[f_target] = level
                unassigned_mask[f_target] = False

    # Randomly assign unmatched observations
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    assigned_indices[unassigned_mask] = sample_indices(uniform[unassigned_mask], cdf, np.arange(len(weights)))
    assigned_levels[unassigned_mask] = 0

    # Write back indices
    df_target[source_identifier] = df_source[source_identifier].values[assigned_indices]

    return df_target, assigned_levels


def statistical_matching(progress, df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
                         random_seed=0, minimum_observations=0, percentage_matched = 0, initial_nb_of_agents = 0):
    
    # Columns check: mandatory columns should be a "left-slice" of columns.
    if mandatory_columns:
        if not is_left_slice(columns, mandatory_columns):
            raise RuntimeError("Mandatory columns must match the beginning of columns!")

    if initial_nb_of_agents == 0 and len(df_target) > 0:
        initial_nb_of_agents = len(df_target)

    assert(initial_nb_of_agents > 0)

    # Set up RNG
    rng = np.random.RandomState(random_seed)

    # Termination step
    if minimum_observations == 7:
        # At this point, the goal is to match everyone. So we do not consider the mandatory columns any longer.
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, None,
                         rng, minimum_observations)

        share_of_matched_agents = round(len(df_matching) / initial_nb_of_agents * 100,2) + percentage_matched
        
        print(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
        return df_matching, assigned_levels

    else:
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns,
                         rng, minimum_observations)
        
        df_not_matching_on_mandatory = df_matching[assigned_levels < len(mandatory_columns)]
        df_matching_on_mandatory     = df_matching[assigned_levels >= len(mandatory_columns)]

        matched_levels               = assigned_levels[assigned_levels >= len(mandatory_columns)]

        next_minimum_observations    =  decrease_minimum_observation(minimum_observations)

        share_of_matched_agents = len(df_matching_on_mandatory) / initial_nb_of_agents * 100 + percentage_matched

        print(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
        matching_the_missing, levels = statistical_matching(progress, df_source, source_identifier, weight, df_not_matching_on_mandatory, target_identifier, columns, mandatory_columns, random_seed, next_minimum_observations, share_of_matched_agents, initial_nb_of_agents)
        
        return pd.concat([df_matching_on_mandatory, matching_the_missing]), np.concatenate((matched_levels, levels))
        

def nonparallel_statistical_matching(context, df_source, source_identifier, weight, df_target, target_identifier, columns,
                                  mandatory_columns, minimum_observations=0):
    
    random_seed = context.config("random_seed")
    
    return statistical_matching(context.progress, df_source, source_identifier, weight, df_target, target_identifier,
                                columns, mandatory_columns, random_seed, minimum_observations)


def run_statistical_matching_extended(context, df_source, source_identifier, weight,
                                      df_population, target_identifier,
                                      columns, mandatory_columns,
                                      minimum_observations=0, population_selector=None,
                                      option = "person"):
    
    df_target = df_population.copy()
    
    if population_selector is not None:
        df_target = pd.DataFrame(df_target[population_selector]).copy()

    df_assignment, levels = nonparallel_statistical_matching(
        context,
        df_source, source_identifier, weight,
        df_target, target_identifier,
        columns,
        mandatory_columns,
        minimum_observations=minimum_observations)
    
    df_target = pd.merge(df_target, df_assignment, on=target_identifier)

    assert len(df_target) == len(df_assignment)

    context.set_info("matched_counts", {
        count: np.count_nonzero(levels >= count) for count in range(len(columns) + 1)
    })

    for count in range(len(columns) + 1):
        print("%d matched levels:" % count, np.count_nonzero(levels >= count),
              "%.2f%%" % (100 * np.count_nonzero(levels >= count) / len(df_target),))
        
    # Remove and track unmatchable households (i.e. head of household)

    initial_population_length = len(df_population)
    initial_target_length     = len(df_target)
        
    if option == "household":

        unmatchable_household_selector = levels < 1
        umatchable_household_ids       = set(df_target.loc[unmatchable_household_selector, "household_id"].values)

        unmatchable_person_selector    = df_population["household_id"].isin(umatchable_household_ids)
        removed_person_ids             = set(df_population.loc[unmatchable_person_selector, "person_id"].values)

        removed_household_ids = set() | umatchable_household_ids

        df_target     = df_target.loc[~unmatchable_household_selector, :]
        df_population = df_population.loc[~unmatchable_person_selector, :]

        removed_households_count = sum(unmatchable_household_selector)
        removed_persons_count    = sum(unmatchable_person_selector)

        print("Unmatchable heads of household: ", removed_households_count)
        print("  Removed households: ", removed_households_count)
        print("  Removed persons: ", removed_persons_count)
        print("")

        assert (len(df_target)     == initial_target_length     - removed_households_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [removed_person_ids, removed_household_ids]
    
    elif option == "person":

        unmatchable_person_selector        = levels < 1
        unmatchable_person_ids             = set(df_target.loc[unmatchable_person_selector, "person_id"].values)
        unmatchable_person_selector        = df_population["person_id"].isin(unmatchable_person_ids) 
        unmatchable_person_selector_target = df_target["person_id"].isin(unmatchable_person_ids)  

        df_target     = df_target.loc[~unmatchable_person_selector_target, :]
        df_population = df_population.loc[~unmatchable_person_selector, :]  

        removed_persons_count = sum(unmatchable_person_selector)

        print("  Removed persons: ", removed_persons_count)
        print("")

        assert (len(df_target)     == initial_target_length     - removed_persons_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [unmatchable_person_ids, None]

    

def execute(context):
    df_mz        = context.stage("data.microcensus.persons")
    c            = context.stage("data.constants")
    scenario_day = context.config("specific_day_scenario")

    # Source are the MZ observations, for each STATPOP person, a sample is drawn from there
    df_source = df_mz.copy()
    if scenario_day == "workday":
        df_source = df_mz[df_mz["workday"]]
    elif scenario_day == "weekend":
        df_source = df_mz[df_mz["weekend"]]
    elif scenario_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        df_source = df_mz[df_mz["day"] == scenario_day]
    else:
        raise ValueError(f"Unimplemented day for scenario: {scenario_day}")

    df_source     = df_source.rename(columns={"person_id": "mz_id"})
    df_source["canton_id"] = df_source["canton_id"].astype("int64")

    df_population = context.stage("synthesis.population.sampled")
    
    df_population.loc[:, "employment_status"]                                                                     = 0
    df_population.loc[df_population["employed"] == 1, "employment_status"]                                        = 1
    df_population.loc[(df_population["employed"] == 3) & (df_population["is_student"] == 1), "employment_status"] = 2
    df_population.loc[(df_population["employed"] == 2) & (df_population["is_student"] == 1), "employment_status"] = 2
    df_population.loc[(df_population["employed"] == 1) & (df_population["is_student"] == 1), "employment_status"] = 3
    df_population["sex"] = df_population["sex"].astype(np.int64)

    df_source["household_size_class"]     = df_source["household_size_class"].clip(upper=2)
    df_population["household_size_class"] = df_population["household_size_class"].clip(upper=2)
    df_source["N_children_under_12"]      = df_source["N_children_under_12"].ne(0) 
    df_source["sex"]                      = df_source["sex"].astype(np.int64)
    
    if c.census == "statpop":

        ## We first want to match by household to be able
        ## to add extra attributes to the persons

        number_of_population_persons    = len(np.unique(df_population["person_id"]))
        number_of_population_households = len(np.unique(df_population["household_id"]))

        population_selector  = df_population["age"] >= c.MZ_AGE_THRESHOLD
        population_selector &= df_population["is_head"]
        print(df_source["household_size_class"].unique())
        ##TODO: add here employment job type
        #columns_household_matching           = ["municipality_type", "ovgk", "household_size_class", "sp_region", "canton_id"]#, "age_class"]
        columns_household_matching           = ["municipality_type", "income_class", "household_size_class", "sp_region", "N_children_under_12"]#, "age_class"]
        mandatory_columns_household_matching = columns_household_matching[:3]

        df_target, df_population, removed_ids_list = run_statistical_matching_extended(context, 
                                                             df_source, "mz_id", "household_weight",
                                                             df_population.copy(), "person_id",
                                                             columns_household_matching, mandatory_columns_household_matching,
                                                             minimum_observations = context.config("matching_minimum_observations"), 
                                                             population_selector = population_selector,
                                                             option = "household")
        
        print("First statistical matching done")

        # Convert IDs
        df_target["mz_id"] = df_target["mz_id"].astype(np.int)
        df_source["mz_id"] = df_source["mz_id"].astype(np.int)

        # Get the attributes from the MZ for the head of household (and thus for the household)
        df_attributes = pd.merge(
            df_target[[
                "household_id", "mz_id"
            ]],
            df_source[[
                "mz_id", "number_of_cars_class", "number_of_bikes_class"
            ]],
            on = "mz_id"
        )

        df_attributes["mz_head_id"] = df_attributes["mz_id"]
        del df_attributes["mz_id"]

        assert (len(df_attributes) == len(df_target))

        # Attach attrbiutes to STATPOP for the second matching
        print("Attach attributes to STATPOP for the second matching")
        initial_population_size = len(df_population)

        df_population = pd.merge(
            df_population, df_attributes, on="household_id"
        )

        assert (len(df_population) == initial_population_size)
        del df_attributes

        ## Now that we have added attributes
        ## SECOND MATCHING - NORMAL PEOPLE
        df_population["number_of_cars_class"] = df_population["number_of_cars_class"].clip(upper=1)
        df_source["number_of_cars_class"] = df_source["number_of_cars_class"].clip(upper=1)

        population_selector_normal = (df_population["age"] >= c.MZ_AGE_THRESHOLD) & ~(df_population["collective_housing_resident"])
        # HT and activity-chains are better with canton_id instead of muncipality_type
        #columns_individual_matching           = ["age_class", "sex", "employment_status", "marital_status", "number_of_cars_class", "canton_id", "sp_region"]
        columns_individual_matching           = ["age_class", "sex", "employment_status", "marital_status", "number_of_cars_class", "canton_id", "ovgk"]
        mandatory_columns_individual_matching = columns_individual_matching[:5] # Trying with 5 and not 4 to make canton id mandatory and see how it affects PT subscriptions

        print("Second statistical matching starting")

        df_target_normal, df_population_normal, removed_ids_list_normal  = run_statistical_matching_extended(context, 
                                                              df_source, "mz_id", "person_weight",
                                                              df_population.copy(), "person_id",
                                                              columns_individual_matching, mandatory_columns_individual_matching,
                                                              minimum_observations = context.config("matching_minimum_observations"), 
                                                              population_selector = population_selector_normal,
                                                              option = "person")
        
        # Extract only the matching information

        df_matching_normal = pd.merge(
            df_population_normal[["person_id", "household_id", "mz_head_id"]],
            df_target_normal[["person_id", "mz_id"]],
            on="person_id", how="left")
        
        df_matching_normal = df_matching_normal.rename(columns = {"mz_id": "mz_id_normal"})
        
        ## SECOND MATCHING - STATPOP PEOPLE WITH RESIDENCE AT MUNICPALITY CENTER

        population_selector = (df_population["age"] >= c.MZ_AGE_THRESHOLD) & (df_population["collective_housing_resident"])
        # with low population samples it can happen that we do not have these individuals
        if (population_selector.any()):
            df_source_center = df_source.merge(context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]], how = "left", right_on = "person_id", left_on = "mz_id")
            df_source_center = df_source_center[df_source_center["activity_chain"]=="home"]

            print("Second statistical matching starting - people with weird residence")

            df_target_center, df_population_center, removed_ids_list_center  = run_statistical_matching_extended(context, 
                                                                df_source_center, "mz_id", "household_weight",
                                                                df_population.copy(), "person_id",
                                                                columns_individual_matching, mandatory_columns_individual_matching,
                                                                minimum_observations = context.config("matching_minimum_observations"), 
                                                                population_selector = population_selector,
                                                                option = "household")
            
            df_matching_center = pd.merge(
                df_population_center[["person_id", "household_id", "mz_head_id"]],
                df_target_center[["person_id", "mz_id"]],
                on="person_id", how="left")     

            df_matching_center = df_matching_center.rename(columns = {"mz_id": "mz_id_center"})  

            removed_ids_list = removed_ids_list + removed_ids_list_center + removed_ids_list_normal
            df_matching      = pd.merge(df_matching_normal, df_matching_center, on = ["person_id", "household_id", "mz_head_id"])
            df_matching["mz_id"] = df_matching["mz_id_normal"].combine_first(df_matching["mz_id_center"])
            del df_matching["mz_id_center"]
            
        else:
            df_matching = df_matching_normal.copy()
            df_matching["mz_id"] = df_matching["mz_id_normal"]

        del df_matching["mz_id_normal"]
       
    
    elif c.census == "are_synpop":
        number_of_population_persons    = len(np.unique(df_population["person_id"]))

        population_selector = df_population["age_class"] > 0

        #columns_individual_matching           = [ "ovgk", "age_class", "sex", "employment_status", "number_of_cars_class", "N_children_under_18"]
        columns_individual_matching           = ["sp_region", "sex", "age_class", "driving_license", "canton_id"]
        mandatory_columns_individual_matching = columns_individual_matching[:5]

        df_population["employment_status"] = df_population["employed"]

        df_target, df_population, removed_ids_list  = run_statistical_matching_extended(context, 
                                                              df_source, "mz_id", "person_weight",
                                                              df_population.copy(), "person_id",
                                                              columns_individual_matching, mandatory_columns_individual_matching,
                                                              minimum_observations = context.config("matching_minimum_observations"), 
                                                              population_selector=population_selector, 
                                                              option = "person")
        
        df_matching = pd.merge(
            df_population[["person_id"]],
            df_target[["person_id", "mz_id"]],
            on="person_id", how="left")

    # Wrap up    

    df_matching["mz_person_id"] = df_matching["mz_id"]
    del df_matching["mz_id"]

    assert (len(df_matching) == len(df_population))

    # Check that all person who don't have a MZ id now are under age
    if c.census == "statpop":
        assert (np.all(df_population[
                    df_population["person_id"].isin(
                        df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
                    )
                ]["age"] < c.MZ_AGE_THRESHOLD))

    
        assert (not np.any(df_matching["mz_head_id"] == -1))

    elif c.census == "are_synpop":
        assert (np.all(df_population[
                    df_population["person_id"].isin(
                        df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
                    )
                ]["age_class"] == 0))

    print("Matching is done. In total, the following observations were removed from the census: ")

    if c.census == "statpop":
        removed_household_ids = removed_ids_list[1]
        print("  Households: %d (%.2f%%)" % ( len(removed_household_ids), 100.0 * len(removed_household_ids) / number_of_population_households))

    removed_person_ids = removed_ids_list[0]
    print("  Persons: %d (%.2f%%)" % (len(removed_person_ids), 100.0 * len(removed_person_ids) / number_of_population_persons))

    # Return
    return df_matching, removed_person_ids


