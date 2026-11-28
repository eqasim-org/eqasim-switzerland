"""Cache final prepared people, activities and commercial trips once for SUMO."""

import hashlib
from pathlib import Path

from scripts.monitoring import monitored
from sumo.functions.scenario import modes, prepared
from sumo.functions.scenario.population.vehicle_tools import (
    CACHE_FILENAME,
    write_trip_cache,
)
from sumo.functions.tables import write_table


def configure(context):
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("synthesis.prepared.population")
    context.config("exlude_external_population_in_sumo", True)
    context.config("sumo_population_crs", "EPSG:2056")


@monitored
def execute(context):

    # Reuse the final plans, including airport replacements and imported populations.

    bundle = context.stage("synthesis.prepared.population")
    population = prepared.people(
        bundle, context.config("exlude_external_population_in_sumo")
    )
    selected = modes.selected(context.config("sumo_modes"))
    cache = Path(context.path()).resolve()

    # Cache only IPC paths; later stages never reload the full prepared population.

    return {
        "persons": write_table(population["persons"], cache / "persons.arrow"),
        "activities": write_table(population["activities"], cache / "activities.arrow"),
        "vehicle_trips": write_trip_cache(
            population,
            cache / CACHE_FILENAME,
            context.config("sumo_population_crs"),
            selected,
        ),
        "commercial_trips": write_table(
            prepared.vehicles(bundle, selected), cache / "commercial.arrow"
        ),
    }


def validate(context):
    from sumo.functions import tables
    from sumo.functions.scenario.population import tools, vehicle_tools

    return [
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (prepared, tools, vehicle_tools, tables)
    ]
