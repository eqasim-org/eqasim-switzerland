"""Compatibility stage: scenario demand is prepared under sumo.scenario."""

from sumo.scenario.unrouted_demand import _population_trips


def configure(context):
    context.stage("sumo.scenario.demand")


def execute(context):

    # Reuse the canonical scenario demand and its shared mode filtering.

    return context.stage("sumo.scenario.demand")
