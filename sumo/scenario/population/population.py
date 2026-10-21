"""Prepare person attributes and activity chains for the SUMO demand writers."""

from sumo.scenario.population.tools import (
    as_frame, combine_populations, prepare_activities,
)


def configure(context):
    context.stage("synthesis.population.models.subscriptions")
    context.stage("synthesis.population.activities")
    context.stage("synthesis.population.spatial.locations")
    context.config("enable_mode_choice", False)
    context.stage(
        "mode_choice.run" if context.config("enable_mode_choice")
        else "synthesis.population.trips_with_departures", alias="population_trips",
    )
    context.config("include_external_population", True)
    if context.config("include_external_population"):
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

    # Include the same optional person populations as the MATSim scenario.

    if context.config("include_external_population"):
        external = context.stage("data.external_population.read_outputs")
        populations.append((external[0], external[1], "external"))
    if context.config("include_cross_border"):
        crossing = context.stage("data.cross_border.generate_cross_border_traffic")
        populations.append((crossing[0], crossing[1], "crossborder"))

    # Return independent, validated tables for reuse by SUMO export stages.

    return combine_populations(populations)
