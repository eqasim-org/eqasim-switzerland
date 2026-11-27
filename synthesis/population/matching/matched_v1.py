import numba
import numpy as np
import pandas as pd
import logging
from synthesis.population.matching import opening_hours, spatial_feature, workflow
logger = logging.getLogger("synpp")

"""
This stage attaches observations from the microcensus to the synthetic population sample.
This is done by statistical matching. Here, a recursive version of statistical matching is implemented.
It progressively decreases the minimum number of observations to ensure the most important attributes are always matched.
"""


def configure(context):
    opening_hours.configure(context)
    spatial_feature.configure(context)
    context.config("hot_deck_matching_runners", 1)
    context.config("random_seed")
    context.config("matching_minimum_observations", 10)
    context.config("matching_enable_diagnostics", False)
    context.config("specific_day_scenario", default = "workday")
    context.config("matching_density_bounds", [400., 950., 1800.])

    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.activity_chains")
    context.stage("data.microcensus.commute")
    context.stage("synthesis.population.sampled")
    context.stage("synthesis.population.spatial.primary.work.work_locations", alias="work_locations")
    context.stage("data.constants")
    
    if context.config("census", "statpop") == "statpop":
        context.stage("synthesis.population.spatial.primary.education.locations", alias="education_locations")


def execute(context):

    # Load the common recipient cohort and prepare V1's categorical attributes.

    prepared = workflow.preprocess(context, "v1")

    # Run the unchanged recursive hot-deck donor search.

    if not prepared.eligible.any():
        return workflow.postprocess(prepared, workflow.initial_mapping(prepared), set())
    mapping, removed = _match_prepared(context, prepared)

    # Restore bypasses, align output IDs and account for all removed residents.

    return workflow.postprocess(prepared, mapping, removed)


def validate(context):
    # Synpp hashes stage source but not imported commute-predictor helpers.
    import hashlib
    from pathlib import Path
    return hashlib.sha256(Path(__file__).with_name("commute.py").read_bytes() +
                          Path(opening_hours.__file__).read_bytes() +
                          Path(workflow.__file__).read_bytes() +
                          Path(spatial_feature.__file__).read_bytes() +
                          Path(__file__).parent.parent.joinpath("models", "non_private.py").read_bytes()).hexdigest()


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


@numba.njit(cache=True)
def sample_indices(uniform, cdf, selected_indices):
    n = len(uniform)
    out = np.empty(n, dtype=np.int64)

    # Binary search over CDF for each random draw.
    for i in range(n):
        u = uniform[i]
        lo = 0
        hi = len(cdf)

        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid

        out[i] = selected_indices[lo]

    return out


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

    # Perform matching
    weights = df_source[weight].values
    assigned_indices = np.ones((len(df_target),), dtype=int) * -1
    unassigned_mask = np.ones((len(df_target),), dtype=bool)
    assigned_levels = np.ones((len(df_target),), dtype=int) * -1
    uniform = rng.random_sample(size=(len(df_target),))

    if mandatory_columns:
        minimum_level = len(mandatory_columns)
    else:
        minimum_level = 1

    for level in range(len(columns), minimum_level - 1, -1):
        if not np.any(unassigned_mask):
            break

        level_columns = columns[:level]

        # Group indices once per level instead of evaluating all value combinations
        # against full-length boolean masks.
        source_groups = df_source.groupby(level_columns, sort=False).indices
        target_groups = df_target.groupby(level_columns, sort=False).indices

        for key, target_indices_all in target_groups.items():
            target_indices = target_indices_all[unassigned_mask[target_indices_all]]

            if len(target_indices) == 0:
                continue

            selected_indices = source_groups.get(key, None)
            if selected_indices is None:
                continue

            if len(selected_indices) < minimum_observations:
                continue

            selected_weights = weights[selected_indices]
            cdf = np.cumsum(selected_weights)
            cdf /= cdf[-1]

            assigned_indices[target_indices] = sample_indices(uniform[target_indices], cdf, selected_indices)
            assigned_levels[target_indices] = level
            unassigned_mask[target_indices] = False

    # Randomly assign unmatched observations
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    assigned_indices[unassigned_mask] = sample_indices(uniform[unassigned_mask], cdf, np.arange(len(weights), dtype=np.int64))
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

        retained = int(np.count_nonzero(assigned_levels >= 1))
        rejected = len(df_matching) - retained
        share_of_matched_agents = 100. * retained / initial_nb_of_agents + percentage_matched
        logger.info("V1 final preference relaxation: %d/%d remaining recipients matched at least the first "
                    "ordered attribute (%s); %d zero-attribute fallbacks will be rejected. "
                    "Accepted %.2f%% of this cohort so far; opening-hours region masks are never relaxed.",
                    retained, len(df_matching), columns[0], rejected, share_of_matched_agents)
        
        return df_matching, assigned_levels

    else:
        df_matching, assigned_levels = recursive_iteration_statmatch(df_source, source_identifier, weight, df_target, target_identifier, columns, mandatory_columns,
                         rng, minimum_observations)
        
        df_not_matching_on_mandatory = df_matching[assigned_levels < len(mandatory_columns)]
        df_matching_on_mandatory     = df_matching[assigned_levels >= len(mandatory_columns)]

        matched_levels               = assigned_levels[assigned_levels >= len(mandatory_columns)]

        next_minimum_observations    =  decrease_minimum_observation(minimum_observations)

        share_of_matched_agents = len(df_matching_on_mandatory) / initial_nb_of_agents * 100 + percentage_matched

        log = logger.info if len(df_matching_on_mandatory) else logger.debug
        log("V1 preference pass: minimum donor observations=%d; accepted %d/%d remaining recipients "
            "with all mandatory preferences, retrying %d; cumulative %.2f%% of this cohort",
            minimum_observations, len(df_matching_on_mandatory), len(df_target),
            len(df_not_matching_on_mandatory), share_of_matched_agents)
        
        matching_the_missing, levels = statistical_matching(progress, df_source, source_identifier, weight, df_not_matching_on_mandatory, target_identifier, columns, mandatory_columns, random_seed, next_minimum_observations, share_of_matched_agents, initial_nb_of_agents)
        
        return pd.concat([df_matching_on_mandatory, matching_the_missing]), np.concatenate((matched_levels, levels))
        

def nonparallel_statistical_matching(context, df_source, source_identifier, weight, df_target, target_identifier, columns,
                                  mandatory_columns, minimum_observations=0):
    
    random_seed = context.config("random_seed")

    # Partition outside the relaxation recursion so even its final fallback
    # remains inside the recipient's exact opening-hours region.
    region = opening_hours.REGION
    if region in df_target:
        if region not in df_source or df_source[region].isna().any() or df_target[region].isna().any():
            raise ValueError("Both matching frames require complete opening-hours regions")
        assignments, all_levels = [], []
        for label, target in df_target.groupby(region, sort=True):
            source = df_source.loc[df_source[region].eq(label)]
            logger.info("V1 opening-hours region %s: %d recipients, %d eligible donors; region equality is hard",
                        label, len(target), len(source))
            target = target.drop(columns=region)
            if source.empty:
                logger.warning("V1 region %s: rejecting %d recipients because this cohort has no donor in the same region",
                               label, len(target))
                assigned = target[[target_identifier] + columns].copy()
                assigned[source_identifier] = -1
                levels = np.full(len(target), -1, dtype=int)
            else:
                assigned, levels = statistical_matching(context.progress, source.drop(columns=region),
                    source_identifier, weight, target, target_identifier, columns, mandatory_columns,
                    random_seed, minimum_observations)
            assignments.append(assigned)
            all_levels.append(pd.Series(levels, index=assigned[target_identifier]))
        if not assignments:
            empty = df_target[[target_identifier] + columns].copy()
            empty[source_identifier] = pd.Series(dtype="int64")
            return empty, np.empty(0, dtype=int)
        combined = pd.concat(assignments).set_index(target_identifier).loc[df_target[target_identifier]].reset_index()
        levels = pd.concat(all_levels).reindex(df_target[target_identifier]).to_numpy()
        return combined, levels
    
    return statistical_matching(context.progress, df_source, source_identifier, weight, df_target, target_identifier,
                                columns, mandatory_columns, random_seed, minimum_observations)


def run_statistical_matching_extended(context, df_source, source_identifier, weight,
                                      df_population, target_identifier,
                                      columns, mandatory_columns,
                                      minimum_observations=0, population_selector=None,
                                      option = "person"):
    
    columns = spatial_feature.columns(columns, df_source, df_population)
    mandatory_columns = spatial_feature.columns(mandatory_columns, df_source, df_population)
    df_target = df_population.copy()
    
    if population_selector is not None:
        df_target = df_target.loc[population_selector].copy()
    logger.info("V1 %s cohort: %d recipients, %d donors before region masks; ordered attributes=[%s]; "
                "mandatory preferences=[%s] (may relax at the final pass)",
                "collective home-only" if option == "household" else "ordinary", len(df_target), len(df_source),
                ", ".join(columns), ", ".join(mandatory_columns or []))

    df_assignment, levels = nonparallel_statistical_matching(
        context,
        df_source, source_identifier, weight,
        df_target, target_identifier,
        columns,
        mandatory_columns,
        minimum_observations=minimum_observations)
    
    df_target = pd.merge(
        df_target,
        df_assignment,
        on=target_identifier,
        validate="one_to_one",
    )

    assert len(df_target) == len(df_assignment)

    #context.set_info("matched_counts", {
    #    count: np.count_nonzero(levels >= count) for count in range(len(columns) + 1)
    #})

    mandatory_count = len(mandatory_columns or [])
    logger.info("V1 cohort search result: all attributes=%d; mandatory preferences with some optional relaxed=%d; "
                "some mandatory preferences relaxed=%d; rejected no-attribute fallback=%d; rejected empty region pool=%d",
                int(np.count_nonzero(levels == len(columns))),
                int(np.count_nonzero((levels >= max(1, mandatory_count)) & (levels < len(columns)))),
                int(np.count_nonzero((levels >= 1) & (levels < mandatory_count))),
                int(np.count_nonzero(levels == 0)), int(np.count_nonzero(levels < 0)))
        
    # Remove and track unmatchable households (i.e. head of household)

    initial_population_length = len(df_population)
    initial_target_length     = len(df_target)
        
    if option == "household":

        unmatchable_household_selector = levels < 1
        umatchable_household_ids       = set(df_target.loc[unmatchable_household_selector, "household_id"].values)

        # Return only the failed recipient IDs. Whole-household expansion is
        # shared with V2/V3 in workflow.postprocess, after all searches finish.
        removed_person_ids = set(df_target.loc[unmatchable_household_selector, "person_id"])
        removed_household_ids = umatchable_household_ids
        df_target = df_target.loc[~unmatchable_household_selector, :]
        df_population = df_population.loc[~df_population.person_id.isin(removed_person_ids), :]
        removed_persons_count = len(removed_person_ids)
        logger.info("V1 collective cohort: %d unmatched recipients; household removal "
                    "is deferred to shared postprocessing", removed_persons_count)
        assert len(df_target) == initial_target_length - removed_persons_count
        assert len(df_population) == initial_population_length - removed_persons_count

        return df_target, df_population, [removed_person_ids, removed_household_ids]
    
    elif option == "person":

        unmatchable_person_selector        = levels < 1
        unmatchable_person_ids             = set(df_target.loc[unmatchable_person_selector, "person_id"].values)
        unmatchable_person_selector        = df_population["person_id"].isin(unmatchable_person_ids) 
        unmatchable_person_selector_target = df_target["person_id"].isin(unmatchable_person_ids)  

        df_target     = df_target.loc[~unmatchable_person_selector_target, :]
        df_population = df_population.loc[~unmatchable_person_selector, :]  

        removed_persons_count = sum(unmatchable_person_selector)

        logger.info("V1 ordinary cohort: retained %d/%d recipients; removed %d without a retained attribute match "
                    "or compatible region donor", len(df_target), initial_target_length, removed_persons_count)

        assert (len(df_target)     == initial_target_length     - removed_persons_count)
        assert (len(df_population) == initial_population_length - removed_persons_count)

        return df_target, df_population, [unmatchable_person_ids, None]

    
def get_mz_persons(context):
    """Compatibility entry point for shared survey cleaning and commute preparation."""
    return workflow.load_survey(context)


def _match_prepared(context, prepared):
    """Run the original recursive donor search on centrally prepared inputs."""
    df_source, df_population = prepared.source, prepared.population
    const = prepared.const
    diagnostics_enabled = context.config("matching_enable_diagnostics")
    if const.census == "statpop":
        number_of_population_persons    = len(np.unique(df_population["person_id"]))

        population_selector = workflow.eligible_mask(df_population, const)
        

        # HT and activity-chains are better with canton_id instead of municipality_type
        # TODO: 3 levels income class here

        columns_individual_matching = [
            "age_class", "sex", "car_availability", "employed", "employment_status", "ovgk", "population_density_class",  
            "N_children_under_12", "sp_region", "broad_commute_distance_class", "commute_distance_class", 
            "income_class", "canton_id", "work_location_type", "urban"
        ]
        mandatory_columns_individual_matching = [
            "age_class", "sex", "car_availability", "employed", "employment_status", "ovgk", "population_density_class",
            "N_children_under_12", "sp_region", "broad_commute_distance_class"
        ]

        logger.info("V1 matching ordinary residents in separate age-band donor pools")

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

        # Accommodation evidence does not select a different donor pool.
        # Recovered residents use the same age bands and matching variables.
        for band_name, (age_min, age_max) in age_bands.items():
            # Selector for this band, respecting MZ_AGE_THRESHOLD and excluding collective housing
            sel = (~df_population_work["collective_housing_resident"]) & workflow.eligible_mask(df_population_work, const)
            if age_min is not None:
                sel &= (df_population_work["age"] >= age_min)
            if age_max is not None:
                sel &= (df_population_work["age"] <= age_max)
            if not sel.any():
                continue

            # Filter source to the same age band
            src_band = df_source
            if age_min is not None:
                src_band = src_band[src_band["age"] >= age_min]
            if age_max is not None:
                src_band = src_band[src_band["age"] <= age_max]
            # Safety fallback: if band-filtered source is empty, fall back to full source
            if len(src_band) == 0:
                logger.warning(f"Source is empty for band '{band_name}' after age filter; falling back to full df_source.")
                src_band = df_source
                
            # Diagnostics BEFORE matching this band (systematic feature checks)
            if diagnostics_enabled:
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

            logger.info("V1 age band %s: %d ordinary recipients, %d donors after age filtering",
                        band_name, int(sel.sum()), len(src_band))
            
            if band_name == "u15":
                youth           = ["age_class", "sex", "ovgk", "population_density_class", "sp_region", "urban", "canton_id"]
                youth_mandatory = ["age_class", "sex","ovgk", "population_density_class", "sp_region"]
                
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    youth, youth_mandatory,
                    minimum_observations = context.config("matching_minimum_observations"),
                    population_selector  = sel,
                    option = "person"
                )
            
            elif band_name == "15_23":
                youth           = ["age_class", "sex", "employed", "car_availability", "ovgk", "population_density_class", 
                                   "employment_status", "sp_region", "broad_commute_distance_class", "urban", "work_location_type",
                                   "commute_distance_class", "canton_id"]
                youth_mandatory = ["age_class", "sex", "employed", "car_availability", "ovgk", "population_density_class", 
                                   "employment_status","sp_region"]
                
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    youth, youth_mandatory,
                    minimum_observations = context.config("matching_minimum_observations"),
                    population_selector = sel,
                    option = "person"
                )
            
            else:
                df_target_band, df_population_work, removed_ids_list_band = run_statistical_matching_extended(
                    context,
                    src_band, "mz_id", "person_weight",
                    df_population_work, "person_id",
                    columns_individual_matching, mandatory_columns_individual_matching,
                    minimum_observations = context.config("matching_minimum_observations"),
                    population_selector = sel,
                    option = "person"
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
                on="person_id", how="left", validate="one_to_one"
            )

        # Ensure the expected column exists
        df_matching_normal["mz_id_normal"] = np.nan

        # Combine band assignments (first non-null wins)
        for col in band_cols:
            df_matching_normal["mz_id_normal"] = df_matching_normal["mz_id_normal"].combine_first(df_matching_normal[col])
            del df_matching_normal[col]

        removed_ids_list_normal = [removed_person_ids_normal, None]

        # --- SECOND MATCHING - STATPOP PEOPLE WITH RESIDENCE AT MUNICIPALITY CENTER ---
        population_selector = workflow.eligible_mask(df_population, const) & df_population["collective_housing_resident"]

        # with low population samples it can happen that we do not have these individuals
        if (population_selector.any()):
            df_source_center = df_source.merge(
                context.stage("data.microcensus.activity_chains")[["person_id", "activity_chain"]],
                how="left", right_on="person_id", left_on="mz_id",
                validate="one_to_one",
            )
            df_source_center = df_source_center[df_source_center["activity_chain"] == "home"]

            logger.info("V1 collective residents: %d recipients, %d home-only donor diaries; broad donor age-band filter bypassed",
                        int(population_selector.sum()), len(df_source_center))

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
                on="person_id", how="left", validate="one_to_one"
            )

            df_matching_center = df_matching_center.rename(columns={"mz_id": "mz_id_center"})

            removed_ids_list = removed_ids_list_center + removed_ids_list_normal
            df_matching = pd.merge(
                df_matching_normal,
                df_matching_center,
                on=["person_id", "household_id"],
                validate="one_to_one",
            )
            df_matching["mz_id"] = df_matching["mz_id_normal"].combine_first(df_matching["mz_id_center"])
            del df_matching["mz_id_center"]

        else:
            df_matching = df_matching_normal.copy()
            df_matching["mz_id"] = df_matching["mz_id_normal"]
            removed_ids_list = removed_ids_list_normal

        del df_matching["mz_id_normal"]

    elif const.census == "are_synpop":
        number_of_population_persons    = len(np.unique(df_population["person_id"]))

        population_selector = workflow.eligible_mask(df_population, const)

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
            on="person_id", how="left", validate="one_to_one"
        )

    # Wrap up
    # Ensure missing mz_id becomes -1 (so your downstream assertions using == -1 work)
    df_matching["mz_id"] = pd.to_numeric(df_matching["mz_id"], errors="coerce").fillna(-1).astype(int)

    df_matching["mz_person_id"] = df_matching["mz_id"]
    del df_matching["mz_id"]

    # Shared postprocessing accounts for rows omitted by either search/merge.

    # The common postprocessor validates bypasses, IDs and retained recipients.

    removed_person_ids = removed_ids_list[0]

    return df_matching, removed_person_ids
