import itertools

import numba
import numpy as np
import pandas as pd

import data.constants as c

"""
This stage attaches observations from the microcensus to the synthetic population sample.
This is done by statistical matching.
"""

@numba.jit(nopython=True, parallel=True)
def sample_indices(uniform, cdf, selected_indices):
    indices = np.arange(len(uniform))

    for i, u in enumerate(uniform):
        indices[i] = np.count_nonzero(cdf < u)

    return selected_indices[indices]

def sample_indices_non_parallel(uniform, cdf, selected_indices):
    indices = np.arange(len(uniform))

    for i, u in enumerate(uniform):
        indices[i] = np.count_nonzero(cdf < u)

    return selected_indices[indices]


def statistical_matching(progress, df_source, source_identifier, weight, df_target, target_identifier, columns,
                         random_seed=0, minimum_observations=0):
    random = np.random.RandomState(random_seed)

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
    uniform = random.random_sample(size=(len(df_target),))

    column_indices = [np.arange(len(unique_values[column])) for column in columns]

    for level in range(1, len(column_indices) + 1)[::-1]:
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

                progress.update(np.count_nonzero(f_target))

    # Randomly assign unmatched observations
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    assigned_indices[unassigned_mask] = sample_indices(uniform[unassigned_mask], cdf, np.arange(len(weights)))
    assigned_levels[unassigned_mask] = 0

    progress.update(np.count_nonzero(unassigned_mask))

    assert np.count_nonzero(unassigned_mask) == 0
    assert np.count_nonzero(assigned_indices == -1) == 0

    # Write back indices
    df_target[source_identifier] = df_source[source_identifier].values[assigned_indices]
    df_target = df_target[[target_identifier, source_identifier]]

    return df_target, assigned_levels


def _run_parallel_statistical_matching(context, args):
    # Pass arguments
    df_target, random_seed = args

    # Pass data
    df_source = context.data("df_source")
    source_identifier = context.data("source_identifier")
    weight = context.data("weight")
    target_identifier = context.data("target_identifier")
    columns = context.data("columns")
    minimum_observations = context.data("minimum_observations")

    return statistical_matching(context.progress, df_source, source_identifier, weight, df_target, target_identifier,
                                columns, random_seed, minimum_observations)


def parallel_statistical_matching(context, df_source, source_identifier, weight, df_target, target_identifier, columns,
                                  minimum_observations=0):
    random_seed = context.config("random_seed")
    processes = context.config("threads")

    random = np.random.RandomState(random_seed)
    chunks = np.array_split(df_target, processes)

    with context.progress(label="Statistical matching ...", total=len(df_target)):
        with context.parallel({
            "df_source": df_source, "source_identifier": source_identifier, "weight": weight,
            "target_identifier": target_identifier, "columns": columns,
            "minimum_observations": minimum_observations
        }) as parallel:
            random_seeds = random.randint(10000, size=len(chunks))
            results = parallel.map(_run_parallel_statistical_matching, zip(chunks, random_seeds))

            levels = np.hstack([r[1] for r in results])
            df_target = pd.concat([r[0] for r in results])

            return df_target, levels

def configure(context):
    context.config("threads")
    context.config("random_seed", 0)
    context.config("matching_minimum_observations", 20)
    context.config("weekend_scenario", False)
    context.config("specific_weekend_scenario", "all") # options are "all", "saturday", "sunday"
    context.config("include_children")

    context.stage("data.microcensus.persons")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.matched")
    
    if context.config("include_children"):
        context.stage("data.kids_probability_table")
        context.stage("data.uk_data_for_kids")
        context.stage("data.microcensus.activity_chains")
        
        
def execute(context):
    #df_population                   = context.stage("synthesis.population.sampled").copy()
    df_matching, removed_person_ids, df_population = context.stage("synthesis.population.matched")        
    include_kids                    = context.config("include_children")
    is_weekend_scenario             = context.config("weekend_scenario")
    specific_weekend_scenario       = context.config("specific_weekend_scenario")
    
    
    if include_kids:
        age_selector   = df_population["age"] < c.MZ_AGE_THRESHOLD
        df_target_init = pd.DataFrame(df_population[age_selector]).copy()
        
        df_source_uk, trips_uk = context.stage("data.uk_data_for_kids")
        df_source_mz = context.stage("data.microcensus.activity_chains").copy()
        
        df_matched = df_matching.copy()
        
        df_population.to_csv("/nas/asallard/export/df_population.csv", index = False)
        df_source_uk.to_csv("/nas/asallard/export/df_source_uk.csv", index = False)
        df_source_mz.to_csv("/nas/asallard/export/df_source_mz.csv", index = False)
        df_matched.to_csv("/nas/asallard/export/df_matched.csv", index = False)
        
        # Source are the MZ observations, for each STATPOP person, a sample is drawn from there
        df_source_mz = pd.DataFrame(df_source_mz[
                                      (is_weekend_scenario & df_source_mz[
                                          "weekend"])  # use only weekend samples for a weekend scenario
                                      |
                                      (~is_weekend_scenario & ~df_source_mz["weekend"])  # and only weekday samples for a weekday
                                      ])

        #If specific weekend context is needed for saturday or sunday
        if (is_weekend_scenario & (specific_weekend_scenario != "all")):
            df_source_mz = pd.DataFrame(df_source_mz[((specific_weekend_scenario == "saturday") & df_source_mz["saturday"]) |
                                      ((specific_weekend_scenario == "sunday") & df_source_mz["sunday"])])
        
        # Select education activity chains from the microcensus, the other ones from UK data
        df_source_mz = df_source_mz[df_source_mz["category_activities"] != 2]
        df_source_uk = df_source_uk[df_source_uk["category_activities"] == 2]
            
        #df_source = pd.concat([df_source_uk, df_source_mz])
        
        df_source_mz["source_id"] = df_source_mz["person_id"]
        df_source_uk["source_id"] = df_source_uk["person_id"]
        
        proba_table = context.stage("data.kids_probability_table")
        proba_table.to_csv("/nas/asallard/export/proba_table.csv", index = False)
        print(proba_table)
        
        ages_kids = list(set(df_target_init["age"]))
        df_target_init.loc[:, "category_activities"] = 0
        
        for age in ages_kids:
            filter_age = df_target_init["age"] == age
            df_age = df_target_init[filter_age]
            unirandom  =  np.random.uniform(0,1,len(df_age))
            
            row = proba_table.iloc[[age]].values[0][1:]
            
            cdf = np.cumsum(row)
            cdf /= cdf[-1]
            
            ind = sample_indices_non_parallel(unirandom, cdf, np.arange(len(row)))
            
            df_target_init.loc[filter_age, "category_activities"] = ind
            
        print(df_target_init.groupby(["age", "category_activities"]).count()["person_id"])
                    
        # statistical matching for education, home, and non-education

        # 1. education -> use df_source_mz
        df_target_1 = df_target_init[df_target_init["category_activities"] == 1].copy()
        columns = ["sex", "household_size_class",
                   "municipality_type", "number_of_cars_class", "number_of_bikes_class"
                   ]
                   
        source1 = df_source_mz[df_source_mz["category_activities"] == 1]
        
        df_assignment1, levels1 = parallel_statistical_matching(
            context,
            source1, "source_id", "person_weight",
            df_target_1, "person_id",
            columns,
            minimum_observations=context.config("matching_minimum_observations"))
            
        #2. Home
        df_target_2 = df_target_init[df_target_init["category_activities"] == 0].copy()
        columns = ["sex", "household_size_class",
                   "municipality_type", "number_of_cars_class", "number_of_bikes_class"
                   ]
                   
        source2 = df_source_mz[df_source_mz["category_activities"] == 0]
        
        df_assignment2, levels2 = parallel_statistical_matching(
            context,
            source2, "source_id", "person_weight",
            df_target_2, "person_id",
            columns,
            minimum_observations=context.config("matching_minimum_observations"))
            
        # 3. Non education
        df_target_3 = df_target_init[df_target_init["category_activities"] == 2].copy()       
        columns = ["age", "sex", "household_size_class",
                   "municipality_type", "number_of_cars_class", "number_of_bikes_class"
                  ]
        df_source_uk = df_source_uk[df_source_uk["category_activities"] == 2]
        df_assignment3, levels3 = parallel_statistical_matching(
            context,
            df_source_uk, "source_id", "person_weight",
            df_target_3, "person_id",
            columns,
            minimum_observations=context.config("matching_minimum_observations"))
            
           
        df_assignment = pd.concat([df_assignment1, df_assignment2, df_assignment3])
        
        df_target = pd.merge(df_target_init, df_assignment, on="person_id")
        assert len(df_target) == len(df_assignment)
        
        df_target["mz_person_id"] = df_target["source_id"]
        del df_target["source_id"]
        
        levels = np.array([*levels1, *levels2, *levels3])
        
        for count in range(len(columns) + 1):
            print("%d matched levels:" % count, np.count_nonzero(levels >= count),
                  "%.2f%%" % (100 * np.count_nonzero(levels >= count) / len(df_target),))
    
        # Remove and track unmatchable persons
        initial_statpop_length = len(df_population)
        initial_target_length  = len(df_target)
        
        
    
        unmatchable_person_selector = levels < 1
        umatchable_household_ids    = set(df_target.loc[unmatchable_person_selector, "household_id"].values)
        unmatchable_member_selector = df_population["household_id"].isin(umatchable_household_ids)
    
        removed_person_ids    = set(df_population.loc[unmatchable_member_selector, "person_id"].values)
        removed_household_ids = umatchable_household_ids
    
        df_target     = df_target.loc[~unmatchable_person_selector, :]
        df_population = df_population.loc[~unmatchable_member_selector, :]
    
        removed_persons_count    = sum(unmatchable_person_selector)
        removed_households_count = len(umatchable_household_ids)
        removed_members_count    = sum(unmatchable_member_selector)
    
        print("Unmatchable persons: ", removed_persons_count)
        print("  Removed households: ", removed_households_count)
        print("  Removed household members: ", removed_members_count)
        print("")
        
        
        assert (len(df_target) == initial_target_length - removed_persons_count)
        assert (len(df_population) == initial_statpop_length - removed_members_count)
        
        # Merge with outputs of previous stage
        df_matched = df_matched[~ df_matched["person_id"].isin(df_target["person_id"])]
        
        df_matching_all = pd.concat([df_matched, df_target])
        df_matching = df_matching_all[["person_id", "household_id",  "mz_head_id", "mz_person_id"]]
        
        print(len(df_matching))
        df_matching.to_csv("/nas/asallard/export/df_matching.csv", index = False)
        
    return df_matching, removed_person_ids, removed_household_ids
