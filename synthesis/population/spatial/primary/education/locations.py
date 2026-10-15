import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import data.spatial.utils as spatial_utils
import data.utils
import logging
from data.statent.education_institutions import pooled_compatibility
from data.statent.education_campus_weights import DEFAULT_ELIGIBILITY_RULES

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("data.statent.statent")
    context.stage("data.statent.education_institutions")
    context.stage("data.constants")
    context.config("random_seed")
    context.stage("data.microcensus.commute")
    context.stage("data.microcensus.persons")
    context.config("specific_day_scenario", "workday")
    context.config("education_site_eligibility", str(DEFAULT_ELIGIBILITY_RULES))
    if context.config("census", "statpop") == "statpop":
        context.stage("synthesis.population.sampled")
        context.stage("synthesis.population.spatial.primary.education.secondary_distances")
        context.stage("data.spatial.zones")
        context.stage("data.od.tertiary_education_matrix")
        context.stage("data.od.education_matrix")
        context.stage("synthesis.population.spatial.primary.education.balancing")
    else:
        context.stage("synthesis.population.enriched")


def validate(context):
    # Imported choice helpers and reviewed exclusions must invalidate anchors too.
    import hashlib
    from pathlib import Path
    files = [Path(__file__).with_name(name) for name in ["tertiary.py", "secondary.py"]]
    files += [Path(data.utils.__file__), Path(context.config("education_site_eligibility"))]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


# All students receive anchors before diary matching; pooled domestic tertiary
# anchors use SS destination zones and provisional annual BFS campus shares.

STANDARD_DEVIATION_DISTANCE = 200  # meters (radius of 600 m for 99.7% of distribution)
MIN_PROB = 1e-4
def get_distance_weight(dx):
    # Gaussian around 0 with adaptive std for numerical stability
    std = max(STANDARD_DEVIATION_DISTANCE, float(np.min(dx)) / 2.0 if dx.size else STANDARD_DEVIATION_DISTANCE)
    coef = 1.0 / (std * np.sqrt(2.0 * np.pi))
    return np.maximum(coef * np.exp(-0.5 * (dx / std) ** 2), MIN_PROB)

def assign_locations(df_persons, df_statent, c, seed, df_compatibility=None, secondary_distance_model=None):
    """Provide study-compatible locations before replacing domestic tertiary OD.

    The age-based branches preserve existing handling for children and unresolved
    classifications. Abroad/unknown cases remain explicitly labelled proxies.
    """
    df_persons = df_persons.copy()
    df_persons["education_x"] = np.nan
    df_persons["education_y"] = np.nan
    # The updated base branch uses namespaced string facility IDs. An object
    # column can hold these IDs and the initial missing values without coercion.
    df_persons["education_location_id"] = pd.Series(pd.NA, index=df_persons.index, dtype="object")
    df_persons["education_assignment_group"] = "unassigned"
    df_persons["education_assignment_status"] = "legacy_no_classification"
    required = {"education_destination_status", "current_education", "is_student"}
    if c.census == "statpop" and required.issubset(df_persons.columns) and df_compatibility is not None:
        df_persons["education_assignment_status"] = "legacy_unresolved_level"
        df_persons.loc[df_persons["age"].lt(15), "education_assignment_status"] = "age_below_survey_coverage"
        df_persons.loc[df_persons["is_student"].eq(0), "education_assignment_status"] = "legacy_nonstudent_education_plan"
        df_persons.loc[df_persons["education_destination_status"].eq("abroad"), "education_assignment_status"] = "legacy_abroad_proxy"
    # Prepare filters and education types
    filters_persons, filters_locations, education_types, query_sizes = data.utils.prepare_education_locations(df_persons, df_statent, c, df_compatibility)
    
    # Set up RNG
    rng = np.random.RandomState(seed)

    for person_selector, location_selector, type, query_size in zip(filters_persons, filters_locations, education_types, query_sizes):
        f_persons = person_selector
        if not f_persons.any():
            continue
        df_candidates = df_statent[location_selector]
        if df_candidates.empty:
            raise ValueError("No compatible education establishments for %s (%d people)" % (type, f_persons.sum()))

        education_coordinates = np.vstack([df_candidates["x"], df_candidates["y"]]).T
        home_coordinates = np.vstack([df_persons.loc[f_persons, "home_x"], df_persons.loc[f_persons, "home_y"]]).T

        tree = KDTree(education_coordinates)
        distances, indices = tree.query(home_coordinates, min(query_size, len(df_candidates)), return_distance=True)
        
        # Build probabilities
        mz_distances = df_persons.loc[f_persons, "commute_home_distance"].values
        weight = df_candidates["number_employees"].to_numpy(dtype=float)
        weight = np.where(np.isfinite(weight) & (weight > 0), weight, 0.0)
        
        # Employee counts are a location preference, not student capacity. Limit
        # extremes so a large establishment cannot dominate the nearby choices.
        q_low, q_high = np.quantile(weight, [0.25, 0.75])
        weight = np.clip(weight, q_low, q_high) if q_low < q_high else weight
        weight = np.maximum(weight / weight.sum(), MIN_PROB) if weight.sum() > 0 else np.ones(len(weight))

        distance_weight  = get_distance_weight(distances - mz_distances[:, np.newaxis])
        candidate_weight = weight[indices]
        probabilities    = (distance_weight * candidate_weight)
        probabilities   /= np.sum(probabilities, axis=1)[:, np.newaxis]

        # selection
        picks = np.array([rng.multinomial(1, row).argmax() for row in probabilities])
        indices = indices[np.arange(indices.shape[0]), picks]
        
        # Assign education location
        df_persons.loc[f_persons, "education_x"] = df_candidates.iloc[indices]["x"].values
        df_persons.loc[f_persons, "education_y"] = df_candidates.iloc[indices]["y"].values
        df_persons.loc[f_persons, "education_location_id"] = df_candidates.iloc[indices]["enterprise_id"].values
        df_persons.loc[f_persons, "education_assignment_group"] = type
        if not type.startswith("age_") and c.census == "statpop" and df_compatibility is not None:
            df_persons.loc[f_persons, "education_assignment_status"] = "classified_domestic"
            unknown = f_persons & df_persons["education_destination_status"].eq("unknown")
            df_persons.loc[unknown, "education_assignment_status"] = "classified_unknown_destination_proxy"

        logger.info("  %s (%d persons, %d locations)", type, np.count_nonzero(f_persons), len(df_candidates))

    if df_persons["education_location_id"].isna().any():
        raise ValueError("Education plans remain without locations; inspect missing age or eligibility")
    locations = df_persons[["person_id",
                             "education_x", "education_y",
                             "education_location_id", "education_assignment_group", "education_assignment_status"]].rename({"education_location_id": "destination_id",
                                                               "education_x": "x",
                                                               "education_y": "y"},
                                                              axis=1)
    if c.census == "statpop":
        from synthesis.population.spatial.primary.education.secondary import assign_secondary_locations
        # Replace only classified domestic secondary cases after the legacy draw.
        # An independent stream preserves all other base assignments.
        locations = assign_secondary_locations(df_persons, locations, df_statent, seed + 2, secondary_distance_model)
    return locations


def draw_reference_distances(persons, mz_persons, mz_education, seed, scenario="workday"):
    """Independent MZ age/school pools; never requires a matched diary donor.

    Keep the legacy school-age boundaries. Under-six MZ coverage is absent, so
    their nearest-kindergarten fallback is explicit rather than a fake observation.
    Above SS age, these draws seed only complete proxy locations; SS OD replaces
    supported domestic secondary and tertiary destinations afterwards.
    """
    if scenario in ["workday", "weekend"]:
        day = mz_persons[scenario]
    elif scenario in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        day = mz_persons.day.eq(scenario)
    else:
        raise ValueError("Unknown education distance day scenario: %s" % scenario)
    source = mz_persons.loc[day
                            & mz_persons.employment_status.isin([2, 3])].merge(
        mz_education[["person_id", "commute_home_distance"]], on="person_id", validate="one_to_one")
    distances = pd.to_numeric(source.commute_home_distance, errors="coerce")
    weights = pd.to_numeric(source.person_weight, errors="coerce")
    source = source.loc[np.isfinite(distances) & distances.ge(0) & np.isfinite(weights) & weights.gt(0)].copy()
    result = persons.copy()
    result["commute_home_distance"] = 0.
    result["education_distance_reference"] = "nearest_school_no_mz_age_coverage"
    rng = np.random.RandomState(seed)
    for lower, upper in [(-np.inf, 6), (6, 12), (12, 16), (16, np.inf)]:
        selected = result.age.gt(lower) & result.age.le(upper)
        observed = source.loc[source.age.gt(lower) & source.age.le(upper)]
        if not selected.any() or observed.empty:
            continue
        result.loc[selected, "commute_home_distance"] = rng.choice(
            observed.commute_home_distance.to_numpy(), selected.sum(),
            p=observed.person_weight.to_numpy()/observed.person_weight.sum())
        result.loc[selected, "education_distance_reference"] = "independent_mz_age_school_pool"
    return result


def execute(context):
    c = context.stage("data.constants")
    statpop = c.census == "statpop"
    persons = context.stage("synthesis.population.sampled" if statpop else "synthesis.population.enriched").copy()
    if statpop:
        # Every classified student gets a usual anchor BEFORE diary matching.
        # Whether today's diary visits it is measured only in downstream analysis.
        persons = persons.loc[persons.is_student.eq(1)].copy()
        persons = draw_reference_distances(persons, context.stage("data.microcensus.persons"),
            context.stage("data.microcensus.commute")["education"], context.config("random_seed"),
            context.config("specific_day_scenario"))
    else:
        commute = context.stage("data.microcensus.commute")["education"]
        persons = persons.merge(commute[["person_id", "commute_home_distance"]].rename(
            columns={"person_id":"mz_person_id"}), on="mz_person_id", validate="many_to_one")
    registry = context.stage("data.statent.education_institutions")
    rules = pd.read_csv(context.config("education_site_eligibility"), dtype={"statent_local_id":str})
    compatibility = pooled_compatibility(registry, rules)
    distance_model = context.stage("synthesis.population.spatial.primary.education.secondary_distances") if statpop else None
    locations = assign_locations(persons, context.stage("data.statent.statent"), c,
        context.config("random_seed"), compatibility, distance_model)
    if statpop:
        # Preserve independent reference provenance in the cached anchor output;
        # child coverage fallbacks must remain inspectable after donor matching.
        locations["education_distance_reference"] = locations.person_id.map(
            persons.set_index("person_id").education_distance_reference)
        from synthesis.population.spatial.primary.education.secondary import assign_secondary_od_locations
        from synthesis.population.spatial.primary.education.tertiary import assign_od_locations
        # The secondary distance fit supplies within-zone building preferences,
        # independently of the diary. The SS destination zone always wins.
        secondary_target = locations.set_index("person_id").education_secondary_target_distance_m
        persons["commute_home_distance"] = persons.person_id.map(secondary_target).fillna(persons.commute_home_distance)
        locations = assign_secondary_od_locations(persons, locations, context.stage("data.statent.statent"),
            context.stage("data.spatial.zones"), context.stage("data.od.education_matrix")[0],
            context.config("random_seed") + 3)
        locations = assign_od_locations(persons, locations, context.stage("data.statent.statent"), compatibility,
            context.stage("data.spatial.zones"), context.stage("data.od.tertiary_education_matrix")[0],
            persons.person_id, context.config("random_seed") + 1,
            context.stage("synthesis.population.spatial.primary.education.balancing"))
        diagnostics = context.stage("synthesis.population.spatial.primary.education.balancing")["diagnostics"]
        if not diagnostics.empty and not diagnostics.converged.all():
            logger.warning("Pooled annual campus shares have unresolved residuals; inspect balancing_diagnostics.csv")
    homes = persons.set_index("person_id").loc[locations.person_id]
    locations["commute_distance"] = np.hypot(locations.x.to_numpy()-homes.home_x.to_numpy(),
                                             locations.y.to_numpy()-homes.home_y.to_numpy())
    if (locations.person_id.duplicated().any() or len(locations) != len(persons)
            or not np.isfinite(locations[["x", "y", "commute_distance"]]).all().all()):
        raise ValueError("Every student must retain exactly one finite pre-matching education anchor")
    result = spatial_utils.to_gpd(context, locations, coord_type="education")
    return result
