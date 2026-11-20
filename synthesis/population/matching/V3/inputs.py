"""V1-compatible categories and residence/day eligibility for V3."""

from ..workflow import preprocess, scenario_mask

ADULT = ["age_class", "sex", "car_availability", "employed", "employment_status", "ovgk",
         "population_density_class", "N_children_under_12", "sp_region", "broad_commute_distance_class",
         "commute_distance_class", "income_class", "canton_id", "work_location_type", "urban"]
ADULT_MANDATORY = ADULT[:10]
CHILD = ["age_class", "sex", "ovgk", "population_density_class", "sp_region", "urban", "canton_id"]
CHILD_MANDATORY = CHILD[:5]
YOUTH = ["age_class", "sex", "employed", "car_availability", "ovgk", "population_density_class",
         "employment_status", "sp_region", "broad_commute_distance_class", "urban", "work_location_type",
         "commute_distance_class", "canton_id"]
YOUTH_MANDATORY = YOUTH[:8]
HARD = {"child": ["age_class"], "youth": ["age_class", "employed", "car_availability"],
        "adult": ["car_availability", "employed", "employment_status"]}


def prepare_inputs(context, const):
    prepared = preprocess(context, "v3", const, include_regions=False)
    return prepared.source, prepared.population, prepared.roster
