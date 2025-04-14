import itertools

import numba
import numpy as np
import pandas as pd

import data.constants as c

"""
This stage attaches observations from the microcensus to the synthetic population sample.
This is done by statistical matching. Here, a recursive version of statistical matching is implemented.
It progressively decreases the minimum number of observations to ensure the most important attributes are always matched.
"""


def configure(context):
    context.config("threads")
    context.config("random_seed")
    context.config("matching_minimum_observations", 20)
    context.config("weekend_scenario", False)

    context.stage("data.microcensus.persons")
    context.stage("synthesis.population.sampled")


@numba.jit(nopython=True, parallel=True)
def sample_indices(uniform, cdf, selected_indices):
    indices = np.arange(len(uniform))

    for i, u in enumerate(uniform):
        indices[i] = np.count_nonzero(cdf < u)

    return selected_indices[indices]


def decrease_minimum_observation(N):
    # Possible decreasing functions can be implemented here.
    return N-1


def is_left_slice(list1, list2):
    return list1[:len(list2)] == list2


def statistical_matching(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
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
    assigned_indices = np.ones((len(df_target),), dtype=int) * -1
    unassigned_mask = np.ones((len(df_target),), dtype=bool)
    assigned_levels = np.ones((len(df_target),), dtype=int) * -1
    uniform = rng.random_sample(size=(len(df_target),))

    column_indices = [np.arange(len(unique_values[column])) for column in columns]

    initial_number =  np.count_nonzero(unassigned_mask)

    if mandatory_columns:
        minimum_level = len(mandatory_columns)
    else:
        minimum_level = 1

    for level in range(minimum_level, len(column_indices) + 1)[::-1]:
        level_column_indices = column_indices[:level]
        
        if np.count_nonzero(unassigned_mask) > 0:
            nb_agents_to_match_initial = np.count_nonzero(unassigned_mask)
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
                
            nb_agents_to_match_final = np.count_nonzero(unassigned_mask)
            nb_agents_matched        = nb_agents_to_match_initial - nb_agents_to_match_final
            remaining_share          = round(nb_agents_to_match_final / initial_number * 100, 2)
            matched_share            = round(nb_agents_matched / initial_number * 100, 2)

    # Randomly assign unmatched observations
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    assigned_indices[unassigned_mask] = sample_indices(uniform[unassigned_mask], cdf, np.arange(len(weights)))
    assigned_levels[unassigned_mask] = 0

    # Write back indices
    df_target[source_identifier] = df_source[source_identifier].values[assigned_indices]

    return df_target, assigned_levels   


def recursive_statistical_matching(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns=None,
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
    if minimum_observations == 1:
        # At this point, the goal is really to match everyone. So we do not consider the mandatory columns any longer.
        df_matching, assigned_levels = statistical_matching(df_source, source_identifier, weight, df_target, target_identifier, columns, None,
                         rng, minimum_observations)

        share_of_matched_agents = round(len(df_matching) / initial_nb_of_agents * 100,2) + percentage_matched
        
        print(f"{minimum_observations} obs required - matching {share_of_matched_agents:.2f}% of the population.")
        print("")
        
        return df_matching, assigned_levels

    else:
        df_matching, assigned_levels = statistical_matching(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns,
                         rng, minimum_observations)
        
        df_not_matching_on_mandatory = df_matching[assigned_levels < len(mandatory_columns)]
        df_matching_on_mandatory     = df_matching[assigned_levels >= len(mandatory_columns)]

        unmatched_levels             = assigned_levels[assigned_levels < len(mandatory_columns)]
        matched_levels               = assigned_levels[assigned_levels >= len(mandatory_columns)]

        next_minimum_observations    =  decrease_minimum_observation(minimum_observations)

        share_of_matched_agents = len(df_matching_on_mandatory) / initial_nb_of_agents * 100 + percentage_matched

        print(f"{minimum_observations} obs required - matching {share_of_matched_agents:.2f}% of the population.")
        
        matching_the_missing, levels = recursive_statistical_matching(df_source, source_identifier, weight, df_not_matching_on_mandatory, target_identifier, columns, mandatory_columns, random_seed, next_minimum_observations, share_of_matched_agents, initial_nb_of_agents)
        
        return pd.concat([df_matching_on_mandatory, matching_the_missing]), np.concatenate((matched_levels, levels))


def execute(context):
    df_mz = context.stage("data.microcensus.persons")
    is_weekend_scenario = context.config("weekend_scenario")

    # Source are the MZ observations, for each STATPOP person, a sample is drawn from there
    df_source = pd.DataFrame(df_mz[
                                 (is_weekend_scenario & df_mz[
                                     "weekend"])  # use only weekend samples for a weekend scenario
                                 |
                                 (~is_weekend_scenario & ~df_mz["weekend"])  # and only weekday samples for a weekday
                                 ])

    df_population = context.stage("synthesis.population.sampled")
    number_of_statpop_persons = len(np.unique(df_population["person_id"]))
    number_of_statpop_households = len(np.unique(df_population["household_id"]))

    ## We first want to match by household to be able
    ## to add extra attributes to the persons

    # Match households
    age_selector = df_population["age"] >= c.MZ_AGE_THRESHOLD
    head_selector = age_selector & df_population["is_head"]

    df_target = pd.DataFrame(df_population[head_selector])
    columns = [ "ovgk", "N_children_12", "age_class", "canton_id", "sex"]

    # Perform statistical matching
    df_source = df_source.rename(columns={"person_id": "mz_id"})

    df_assignment, levels = recursive_statistical_matching(
        df_source, "mz_id", "household_weight",
        df_target, "person_id",
        columns,
        columns[:3],
        minimum_observations=context.config("matching_minimum_observations"),
        percentage_matched=0)

    df_target = pd.merge(df_target, df_assignment, on="person_id")
    assert len(df_target) == len(df_assignment)

    context.set_info("matched_counts", {
        count: np.count_nonzero(levels >= count) for count in range(len(columns) + 1)
    })

    for count in range(len(columns) + 1):
        print("%d matched levels:" % count, np.count_nonzero(levels >= count),
              "%.2f%%" % (100 * np.count_nonzero(levels >= count) / len(df_target),))

    # Remove and track unmatchable households (i.e. head of household)

    initial_statpop_length = len(df_population)
    initial_target_length = len(df_target)

    unmatchable_household_selector = levels < 1
    umatchable_household_ids = set(df_target.loc[unmatchable_household_selector, "household_id"].values)
    unmatchable_person_selector = df_population["household_id"].isin(umatchable_household_ids)

    removed_person_ids = set(df_population.loc[unmatchable_person_selector, "person_id"].values)
    removed_household_ids = set() | umatchable_household_ids

    df_target = df_target.loc[~unmatchable_household_selector, :]
    df_population = df_population.loc[~unmatchable_person_selector, :]

    removed_households_count = sum(unmatchable_household_selector)
    removed_persons_count = sum(unmatchable_person_selector)

    print("Unmatchable heads of household: ", removed_households_count)
    print("  Removed households: ", removed_households_count)
    print("  Removed persons: ", removed_persons_count)
    print("")

    assert (len(df_target) == initial_target_length - removed_households_count)
    assert (len(df_population) == initial_statpop_length - removed_persons_count)

    # Convert IDs
    df_target["mz_id"] = df_target["mz_id"].astype(np.int)
    df_source["mz_id"] = df_source["mz_id"].astype(np.int)

    # Get the attributes from the MZ for the head of household (and thus for the household)
    df_attributes = pd.merge(
        df_target[[
            "household_id", "mz_id"
        ]],
        df_source[[
            "mz_id", "income_class", "number_of_cars_class", "number_of_bikes_class"
        ]],
        on = "mz_id"
    )

    df_attributes["mz_head_id"] = df_attributes["mz_id"]
    del df_attributes["mz_id"]

    assert (len(df_attributes) == len(df_target))

    # Attach attrbiutes to STATPOP for the second matching
    print("Attach attributes to STATPOP for the second matching")
    initial_statpop_size = len(df_population)

    df_population = pd.merge(
        df_population, df_attributes, on="household_id"
    )

    assert (len(df_population) == initial_statpop_size)
    del df_attributes

    ## Now that we have added attributes
    ## Match persons
    age_selector = df_population["age"] >= c.MZ_AGE_THRESHOLD
    df_target = pd.DataFrame(df_population[age_selector])

    columns = [ "number_of_cars_class", "N_children_12", "ovgk", "age_class", "sex",  "marital_status"]

    df_assignment, levels = recursive_statistical_matching(
        df_source, "mz_id", "person_weight",
        df_target, "person_id",
        columns,
        columns[:4],
        minimum_observations=context.config("matching_minimum_observations"))

    df_target = pd.merge(df_target, df_assignment, on="person_id")
    assert len(df_target) == len(df_assignment)

    context.set_info("matched_counts", {
        count: np.count_nonzero(levels >= count) for count in range(len(columns) + 1)
    })

    for count in range(len(columns) + 1):
        print("%d matched levels:" % count, np.count_nonzero(levels >= count),
              "%.2f%%" % (100 * np.count_nonzero(levels >= count) / len(df_target),))

    # Remove and track unmatchable persons
    initial_statpop_length = len(df_population)
    initial_target_length = len(df_target)

    unmatchable_person_selector = levels < 1
    umatchable_household_ids = set(df_target.loc[unmatchable_person_selector, "household_id"].values)
    unmatchable_member_selector = df_population["household_id"].isin(umatchable_household_ids)

    removed_person_ids |= set(df_population.loc[unmatchable_member_selector, "person_id"].values)
    removed_household_ids |= umatchable_household_ids

    df_target = df_target.loc[~unmatchable_person_selector, :]
    df_population = df_population.loc[~unmatchable_member_selector, :]

    removed_persons_count = sum(unmatchable_person_selector)
    removed_households_count = len(umatchable_household_ids)
    removed_members_count = sum(unmatchable_member_selector)

    print("Unmatchable persons: ", removed_persons_count)
    print("  Removed households: ", removed_households_count)
    print("  Removed household members: ", removed_members_count)
    print("")

    assert (len(df_target) == initial_target_length - removed_persons_count)
    assert (len(df_population) == initial_statpop_length - removed_members_count)

    # Extract only the matching information

    df_matching = pd.merge(
        df_population[["person_id", "household_id", "mz_head_id"]],
        df_target[["person_id", "mz_id"]],
        on="person_id", how="left")

    df_matching["mz_person_id"] = df_matching["mz_id"]
    del df_matching["mz_id"]

    assert (len(df_matching) == len(df_population))

    # Check that all person who don't have a MZ id now are under age
    assert (np.all(df_population[
                       df_population["person_id"].isin(
                           df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
                       )
                   ]["age"] < c.MZ_AGE_THRESHOLD))

    assert (not np.any(df_matching["mz_head_id"] == -1))

    print("Matching is done. In total, the following observations were removed from STATPOP: ")
    print("  Households: %d (%.2f%%)" % (
    len(removed_household_ids), 100.0 * len(removed_household_ids) / number_of_statpop_households))
    print("  Persons: %d (%.2f%%)" % (
    len(removed_person_ids), 100.0 * len(removed_person_ids) / number_of_statpop_persons))

    # Return
    return df_matching, removed_person_ids
