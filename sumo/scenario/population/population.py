"""Prepare person attributes and activity chains for the SUMO demand writers."""

from pathlib import Path
import hashlib

import polars as pl
from sumo import tables
from sumo.tables import write_table
from sumo.scenario import modes
from sumo.scenario.population import tools as population_tools, vehicle_tools
from sumo.scenario.population.vehicle_tools import CACHE_FILENAME, write_trip_cache
from sumo.scenario.population.tools import (
    as_frame, coalesce_external_activities, combine_populations, prepare_activities,
)


def configure(context):
    modes.configure(context)
    context.config("sumo_population_crs", "EPSG:2056")
    context.stage("synthesis.population.models.subscriptions")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.config("enable_mode_choice", False)
    context.stage(
        "mode_choice.run" if context.config("enable_mode_choice")
        else "synthesis.population.trips_with_departures", alias="population_trips",
    )
    context.config("include_external_population", True)
    context.config("exlude_external_population_in_sumo", True)
    if context.config("include_external_population") and not context.config("exlude_external_population_in_sumo"):
        context.stage("data.external_population.read_outputs")
    context.config("include_cross_border", False)
    if context.config("include_cross_border"):
        context.stage("data.cross_border.generate_cross_border_traffic")


def execute(context):

    # Prepare resident activities using the final modes and corrected departures.

    persons = as_frame(context.stage("synthesis.population.models.subscriptions"))
    activities = prepare_activities(
        context.stage("synthesis.population.activities"),
        context.stage("synthesis.population.spatial.locations"),
        context.stage("population_trips"),
    )
    populations = [(persons, activities, "resident")]

    # Keep cross-border demand and include external people only when explicitly requested.

    if context.config("include_external_population") and not context.config("exlude_external_population_in_sumo"):
        external = context.stage("data.external_population.read_outputs")
        external_activities = coalesce_external_activities(as_frame(external[1]))
        populations.append((external[0], external_activities, "external"))
    if context.config("include_cross_border"):
        crossing = context.stage("data.cross_border.generate_cross_border_traffic")
        populations.append((crossing[0], crossing[1], "crossborder"))

    # Cache compact vehicle demand while the prepared population is already in memory.

    population = combine_populations(populations)
    population["vehicle_trips"] = write_trip_cache(population, Path(context.path()) / CACHE_FILENAME,
        context.config("sumo_population_crs"), modes.selected(context.config("sumo_modes")))

    # Return file paths so Synpp never pickles the full population tables.

    return {"persons": write_table(population["persons"], Path(context.path()) / "persons.arrow"),
            "activities": write_table(population["activities"], Path(context.path()) / "activities.arrow"),
            "vehicle_trips": population["vehicle_trips"]}


def validate(context):
    fingerprint = hashlib.sha256()
    for module in (tables, population_tools, vehicle_tools):
        fingerprint.update(Path(module.__file__).read_bytes())
    available = all((Path(context.path()) / name).is_file()
                    for name in ("persons.arrow", "activities.arrow", CACHE_FILENAME))
    return [fingerprint.hexdigest(), pl.__version__, available]
