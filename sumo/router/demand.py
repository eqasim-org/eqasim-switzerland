"""Compatibility stage: scenario demand is prepared under sumo.macro_scenario."""

from scripts.monitoring import monitored


def configure(context):
    context.stage("sumo.macro_scenario.demand")


@monitored
def execute(context):

    # Reuse the canonical scenario demand and its shared mode filtering.

    return context.stage("sumo.macro_scenario.demand")
