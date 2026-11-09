"""Compare saved V1/V3 diary assignments on the same synthetic input population."""

import hashlib
from pathlib import Path

from . import population_comparison, distance_maps


def configure(context):
    context.config("analysis_path")
    context.config("specific_day_scenario", "workday")
    context.config("matching_comparison_grid_size", 5000.)
    context.config("matching_comparison_canton_grid_size", 1000.)
    context.config("matching_comparison_map_min_people", 5)
    context.stage("synthesis.population.matching.matched_v1", alias="v1")
    context.stage("synthesis.population.matching.matched_v3", alias="v3")
    context.stage("synthesis.population.sampled")
    context.stage("data.microcensus.persons")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.activity_chains")
    context.stage("data.constants")
    context.stage("data.spatial.cantons")


def execute(context):

    # Read completed assignments and the original survey; no downstream simulation is needed.

    mappings = {name.upper(): context.stage(name)[0] for name in ("v1", "v3")}
    output = Path(context.config("analysis_path")) / "matching" / "v1_v3"

    # Aggregate diaries by donor reuse rather than expanding millions of activity rows.

    return population_comparison.generate(output, mappings,
        context.stage("synthesis.population.sampled"), context.stage("data.microcensus.persons"),
        context.stage("data.microcensus.trips")[0], context.stage("data.microcensus.activity_chains"),
        context.stage("data.constants"), context.config("specific_day_scenario"),
        cantons=context.stage("data.spatial.cantons"), distance_map_options={
            "grid_size": context.config("matching_comparison_grid_size"),
            "canton_grid_size": context.config("matching_comparison_canton_grid_size"),
            "min_people": context.config("matching_comparison_map_min_people")})


def validate(context):
    return hashlib.sha256(Path(population_comparison.__file__).read_bytes() +
                          Path(distance_maps.__file__).read_bytes()).hexdigest()
