import os
#os.environ["NUMBA_DISABLE_JIT"] = "1"
import itertools

import numba
import numpy as np
import pandas as pd
import logging
logger = logging.getLogger("synpp")
"""
This stage attaches observations from the microcensus to the synthetic population sample.
This is done by statistical matching. Here, a recursive version of statistical matching is implemented.
It progressively decreases the minimum number of observations to ensure the most important attributes are always matched.
"""

def configure(context):
    context.config("hot_deck_matching_runners")
    context.config("random_seed")
    context.config("matching_minimum_observations", 10)
    context.config("specific_day_scenario", default = "workday")

    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.activity_chains")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
    context.stage("data.constants")

def compare_feature_distribution(df_population_sub, df_source_sub, feature, weight_col="person_weight"):
    """
    Compare unweighted population shares vs weighted source shares for one feature.
    Expects inputs to already be filtered to the segment of interest.
    """
    pop = df_population_sub
    src = df_source_sub

    # Population: unweighted shares
    pop_dist = (
        pop[feature]
        .value_counts(dropna=False, normalize=True)
        .rename("share_population")
        .reset_index()
        .rename(columns={"index": feature})
    )

    # Source: weighted shares
    src_w = (
        src.groupby(feature, dropna=False)[weight_col]
        .sum()
        .rename("weight_sum")
        .reset_index()
    )
    total_w = src_w["weight_sum"].sum()
    src_w["share_source_weighted"] = src_w["weight_sum"] / total_w if total_w != 0 else 0.0

    out = pop_dist.merge(src_w[[feature, "share_source_weighted"]], on=feature, how="outer").fillna(0.0)
    out["diff_pop_minus_source"] = out["share_population"] - out["share_source_weighted"]
    out["abs_diff"] = out["diff_pop_minus_source"].abs()

    # Sort by worst mismatch (absolute)
    out = out.sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return out


def print_matching_diagnostics(df_population_sub, df_source_sub, features, label="", weight_col="person_weight", top_n=8):
    """
    Prints per-feature mismatch summaries (TVD + top categories by abs diff).
    """
    pop_n = len(df_population_sub)
    src_n = len(df_source_sub)
    logger.info(f"\n--- Matching diagnostics: {label} ---")
    logger.info(f"Population rows: {pop_n} | Source rows: {src_n}\n")

    if pop_n == 0:
        logger.info("No population rows in this segment.\n")
        return
    if src_n == 0:
        logger.info("No source rows in this segment.\n")
        return

    for feat in features:
        if feat not in df_population_sub.columns:
            logger.info(f"Feature '{feat}' not in population subset -> skipping diagnostics for this feature.\n")
            continue
        if feat not in df_source_sub.columns:
            logger.info(f"Feature '{feat}' not in source subset -> skipping diagnostics for this feature.\n")
            continue

        out = compare_feature_distribution(df_population_sub, df_source_sub, feat, weight_col=weight_col)
        tvd = 0.5 * out["abs_diff"].sum()  # Total Variation Distance
        max_abs = out["abs_diff"].max() if len(out) else 0.0

        logger.info(f"Feature '{feat}': TVD={tvd:.4f} | max_abs_diff={max_abs:.4f}")
        show = out[[feat, "share_population", "share_source_weighted", "diff_pop_minus_source"]].head(top_n)
        logger.info("\n" + show.to_string(index=False))
        logger.info("")


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
    assigned_indices = np.ones((len(df_target),), dtype=np.int64) * -1
    unassigned_mask = np.ones((len(df_target),), dtype=np.bool_)
    assigned_levels = np.ones((len(df_target),), dtype=np.int64) * -1
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
    if minimum_observations == 1:
        # At this point, the goal is to match everyone. So we do not consider the mandatory columns any longer.
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, None,
                         rng, minimum_observations)

        share_of_matched_agents = round(len(df_matching) / initial_nb_of_agents * 100,2) + percentage_matched
        
        logger.info(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
        return df_matching, assigned_levels

    else:
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns,
                         rng, minimum_observations)
        
        df_not_matching_on_mandatory = df_matching[assigned_levels < len(mandatory_columns)]
        df_matching_on_mandatory     = df_matching[assigned_levels >= len(mandatory_columns)]

        matched_levels               = assigned_levels[assigned_levels >= len(mandatory_columns)]

        next_minimum_observations    =  decrease_minimum_observation(minimum_observations)

        share_of_matched_agents = len(df_matching_on_mandatory) / initial_nb_of_agents * 100 + percentage_matched

        logger.info(f"{minimum_observations} obs required - {share_of_matched_agents:.2f}% of the population matched.")
        
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
        matched_count = np.count_nonzero(levels >= count)
        matched_percent = 100 * matched_count / len(df_target) if len(df_target) > 0 else 0.0
        logger.info(f"{count} matched levels: {matched_count} ({matched_percent:.2f}%)")
        
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

        logger.info("Unmatchable heads of household: %d", removed_households_count)
        logger.info("  Removed households: %d", removed_households_count)
        logger.info("  Removed persons: %d", removed_persons_count)
        logger.info("")

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

        logger.info("  Removed persons: %d", removed_persons_count)
        logger.info("")

        assert (len(df_target)     == initial_target_length     - removed_persons_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [unmatchable_person_ids, None]

    
def get_mz_persons(context):
    df_persons = context.stage("data.microcensus.persons")
    df_trips = context.stage("data.microcensus.trips")[0]
    
    # remove persons who are not employed, but have a work as one of the purposes in their activity chain
    employed_persons = set(df_persons[df_persons["employed"] == True]["person_id"])
    persons_with_work_purpose = set(df_trips[(df_trips["origin_purpose"] == "work") | (df_trips["purpose"] == "work")]["person_id"])
    persons_to_remove = persons_with_work_purpose - employed_persons
    logger.info(f"Removing {len(persons_to_remove)} persons who are not employed but have 'work' as a purpose in their activity chain.")

    df_persons = df_persons[~df_persons["person_id"].isin(persons_to_remove)]
    return df_persons

def execute(context):
    df_mz = get_mz_persons(context)
    const        = context.stage("data.constants")
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
    
    # add commute distance and work location type
    df_work = context.stage("work_locations")[["person_id", "work_location_type", "commute_distance"]]
    df_population = pd.merge(df_population, df_work, on="person_id", how="left")
    assert df_population.loc[df_population.employed==1,"work_location_type"].notna().all(), "Some employed gents are missing commute distance"
    df_population[ "commute_distance"] = df_population["commute_distance"].fillna(-1)
    df_population[ "work_location_type"] = df_population["work_location_type"].fillna("none")
    
    # trasform work_location_type to int
    DICT_WORK_LOCATION_TYPE = {"none": 0, "fixed": 1, "remote": 2, "moving": 3}
    df_population["work_location_type"] = df_population["work_location_type"].map(DICT_WORK_LOCATION_TYPE)
    df_source["work_location_type"] = df_source["work_location_type"].map(DICT_WORK_LOCATION_TYPE)

    # add commute distance classes
    COMMUTE_DISTANCE_BOUNDS = np.array([-10, 0, 1, 3, 6, 9, 12, 15, 20, 50, 1000]) * 1e3 # convert km -> m
    df_population["commute_distance_class"] = np.digitize(df_population["commute_distance"], COMMUTE_DISTANCE_BOUNDS)
    df_source["commute_distance_class"] = np.digitize(df_source["commute_distance"], COMMUTE_DISTANCE_BOUNDS)

    # add age classes 
    AGE_CLASS_UPPER_BOUNDS = [6, 15, 18, 24, 40, 51, 65, 80]
    df_population["age_class"] = np.digitize(df_population["age"], AGE_CLASS_UPPER_BOUNDS)
    df_source["age_class"] = np.digitize(df_source["age"], AGE_CLASS_UPPER_BOUNDS)

    # this is not necessary, but just to make sure employment is working correctly
    df_source["employed"] = df_source["employed"].astype(int)
    df_population["employed"] = (df_population["employed"]==1).astype(int)

    # further cleaning
    df_source["household_size_class"] = df_source["household_size_class"].clip(upper=2)
    df_population["household_size"] = df_population["household_size"].clip(upper=2)

    df_source["N_children_under_12"] = df_source["N_children_under_12"].ne(0)  # presence of children under 12

    df_source["sex"] = df_source["sex"].astype(np.int64)
    var_raw = pd.to_numeric(df_source["car_availability"], errors="coerce")
    df_source["car_availability"] = np.where(var_raw == const.CAR_AVAILABILITY_NEVER, 0, 1).astype("int64")


    # checkig if any duplicates
    assert df_population['person_id'].duplicated().sum()==0, "Duplicate person_id found in population dataframe. Please ensure person_id is unique for each individual in the population."
    assert df_source['mz_id'].duplicated().sum()==0, "Duplicate mz_id found in source dataframe. Please ensure mz_id is unique for each individual in the source."

    if const.census == "statpop":

        number_of_population_persons    = len(np.unique(df_population["person_id"]))
        number_of_population_households = len(np.unique(df_population["household_id"]))

        population_selector  = df_population["age"] >= const.MZ_AGE_THRESHOLD
        df_population["number_of_cars_class"] = df_population["number_of_cars_class"].clip(upper=3)
        df_source["number_of_cars_class"] = df_source["number_of_cars_class"].clip(upper=3)

        # HT and activity-chains are better with canton_id instead of muncipality_type
        columns_individual_matching = [
            "age_class", "sex", "car_availability", "employment_status", "commute_distance_class",
            "ovgk",  "N_children_under_12", "sp_region", "work_location_type", "canton_id"
        ]

        df_population["marital_status"] = df_population["marital_status"].astype("int64")
        df_population["car_availability"] = df_population["car_availability"].astype("int64")
        df_population["municipality_type"] = df_population["municipality_type"].astype("object")
        df_source["municipality_type"] = df_source["municipality_type"].astype("object")
        df_population["sp_region"] = df_population["sp_region"].astype("int64")
        df_source["sp_region"] = df_source["sp_region"].astype("int64")
        df_source["canton_id"] = df_source["canton_id"].astype("int64")
        df_population["canton_id"] = df_population["canton_id"].astype("int64")
        df_population["ovgk"] = (df_population["ovgk"] != "None").astype("int64")
        df_source["ovgk"] = (df_source["ovgk"] != "None").astype("int64")

        mandatory_columns_individual_matching = columns_individual_matching[:8]

        logger.info("Statistical matching starting (normal people split by age band with band-filtered source)")

        # --- NORMAL PEOPLE: split into age bands + filter source by same band ---
        df_population_work = df_population.copy()
        targets_by_band = {}
        removed_person_ids_normal = set()

        # Define bands (inclusive bounds)
        age_bands = {
            "u15":   (None, 14),
            "15_23": (15, 23),
            "gt24":  (24, 150),
        }

        for band_name, (age_min, age_max) in age_bands.items():
            # Selector for this band, respecting MZ_AGE_THRESHOLD and excluding collective housing
            sel = (~df_population_work["collective_housing_resident"]) & (df_population_work["age"] >= const.MZ_AGE_THRESHOLD)
            if age_min is not None:
                sel &= (df_population_work["age"] >= age_min)
            if age_max is not None:
                sel &= (df_population_work["age"] <= age_max)

            if not sel.any():
                continue

            # Filter source to the same age band
            src_band = df_source.copy()
            if age_min is not None:
                src_band = src_band[src_band["age"] >= age_min]
            if age_max is not None:
                src_band = src_band[src_band["age"] <= age_max]

            # Safety fallback: if band-filtered source is empty, fall back to full source
            if len(src_band) == 0:
                logger.warning(f"Source is empty for band '{band_name}' after age filter; falling back to full df_source.")
                src_band = df_source
                
            # Diagnostics BEFORE matching this band (systematic feature checks)
            pop_diag = df_population_work.loc[sel, columns_individual_matching + ["person_id", "household_id"]].copy()
            src_diag = src_band.loc[:, columns_individual_matching + ["mz_id", "person_weight"]].copy()
            print_matching_diagnostics(
                pop_diag,
                src_diag,
                features=columns_individual_matching,
                label=f"normal band '{band_name}' (pre-match)",
                weight_col="person_weight",
                top_n=8
            )

            logger.info(f"  - Matching normal people band: {band_name}")
            if band_name == "u15":
                youth = [
                "age_class", "sex",
                "ovgk", "sp_region", "canton_id",
                ]
                youth_mandatory = [
                "age_class", "sex",
                "ovgk", "sp_region"
                ]
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    youth, youth_mandatory,
                    minimum_observations=context.config("matching_minimum_observations"),
                    population_selector=sel,
                    option="person"
                )
            elif band_name == "15_23":
                youth = [
                "age_class", "sex",
                "ovgk", "employment_status", "car_availability", "sp_region", "commute_distance_class", "canton_id", "work_location_type"
                ]
                youth_mandatory = [
                "age_class", "sex",
                "ovgk", "employment_status", "car_availability", "sp_region"
                ]
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    youth, youth_mandatory,
                    minimum_observations=context.config("matching_minimum_observations"),
                    population_selector=sel,
                    option="person"
                )
            else:
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    columns_individual_matching, mandatory_columns_individual_matching,
                    minimum_observations=context.config("matching_minimum_observations"),
                    population_selector=sel,
                    option="person"
                )

            targets_by_band[band_name] = df_target_band
            removed_person_ids_normal |= set(removed_ids_list_band[0])

        df_population_normal = df_population_work

        # Build one mapping table and fill mz_id from the corresponding band
        df_matching_normal = df_population_normal[["person_id", "household_id"]].copy()

        band_cols = []
        for band_name, df_target_band in targets_by_band.items():
            col = f"mz_id_normal_{band_name}"
            band_cols.append(col)

            df_matching_normal = pd.merge(
                df_matching_normal,
                df_target_band[["person_id", "mz_id"]].rename(columns={"mz_id": col}),
                on="person_id", how="left"
            )

        # Ensure the expected column exists
        df_matching_normal["mz_id_normal"] = np.nan

        # Combine band assignments (first non-null wins)
        for col in band_cols:
            df_matching_normal["mz_id_normal"] = df_matching_normal["mz_id_normal"].combine_first(df_matching_normal[col])
            del df_matching_normal[col]

        removed_ids_list_normal = [removed_person_ids_normal, None]

        # --- SECOND MATCHING - STATPOP PEOPLE WITH RESIDENCE AT MUNICIPALITY CENTER ---
        population_selector = (df_population["age"] >= const.MZ_AGE_THRESHOLD) & (df_population["collective_housing_resident"])

        # with low population samples it can happen that we do not have these individuals
        if (population_selector.any()):
            df_source_center = df_source.merge(
                context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]],
                how="left", right_on="person_id", left_on="mz_id"
            )
            df_source_center = df_source_center[df_source_center["activity_chain"] == "home"]

            logger.info("Second statistical matching starting - people with strange residence")

            df_target_center, df_population_center, removed_ids_list_center  = run_statistical_matching_extended(
                context,
                df_source_center, "mz_id", "household_weight",
                df_population.copy(), "person_id",
                columns_individual_matching, mandatory_columns_individual_matching,
                minimum_observations=context.config("matching_minimum_observations"),
                population_selector=population_selector,
                option="household"
            )

            df_matching_center = pd.merge(
                df_population_center[["person_id", "household_id"]],
                df_target_center[["person_id", "mz_id"]],
                on="person_id", how="left"
            )

            df_matching_center = df_matching_center.rename(columns={"mz_id": "mz_id_center"})

            removed_ids_list = removed_ids_list_center + removed_ids_list_normal
            df_matching      = pd.merge(df_matching_normal, df_matching_center, on=["person_id", "household_id"])
            df_matching["mz_id"] = df_matching["mz_id_normal"].combine_first(df_matching["mz_id_center"])
            del df_matching["mz_id_center"]

        else:
            df_matching = df_matching_normal.copy()
            df_matching["mz_id"] = df_matching["mz_id_normal"]
            removed_ids_list = removed_ids_list_normal

        del df_matching["mz_id_normal"]

    elif const.census == "are_synpop":
        number_of_population_persons    = len(np.unique(df_population["person_id"]))

        population_selector = df_population["age_class"] > 0

        columns_individual_matching           = [ "ovgk", "age_class", "sex", "employment_status", "number_of_cars_class", "N_children_under_18"]
        mandatory_columns_individual_matching = columns_individual_matching[:4]

        df_target, df_population, removed_ids_list  = run_statistical_matching_extended(
            context,
            df_source, "mz_id", "household_weight",
            df_population.copy(), "person_id",
            columns_individual_matching, mandatory_columns_individual_matching,
            minimum_observations=context.config("matching_minimum_observations"),
            population_selector=population_selector,
            option="person"
        )

        df_matching = pd.merge(
            df_population[["person_id"]],
            df_target[["person_id", "mz_id"]],
            on="person_id", how="left"
        )

    # Wrap up
    # Ensure missing mz_id becomes -1 (so your downstream assertions using == -1 work)
    df_matching["mz_id"] = pd.to_numeric(df_matching["mz_id"], errors="coerce").fillna(-1).astype(np.int64)

    df_matching["mz_person_id"] = df_matching["mz_id"]
    del df_matching["mz_id"]

    assert (len(df_matching) == len(df_population))

    # Check that all person who don't have a MZ id now are under age
    if const.census == "statpop":
        assert (np.all(df_population[
            df_population["person_id"].isin(
                df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
            )
        ]["age"] < const.MZ_AGE_THRESHOLD))

    elif const.census == "are_synpop":
        assert (np.all(df_population[
            df_population["person_id"].isin(
                df_matching.loc[df_matching["mz_person_id"] == -1]["person_id"]
            )
        ]["age_class"] == 0))

    logger.info("Matching is done. In total, the following observations were removed from the census: ")

    removed_person_ids = removed_ids_list[0]
    pct_removed = 100.0 * len(removed_person_ids) / number_of_population_persons if number_of_population_persons > 0 else 0.0
    logger.info("  Persons: %d (%.2f%%)", len(removed_person_ids), pct_removed)

    # Return
    return df_matching, removed_person_ids
