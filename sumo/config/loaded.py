"""Prepare assignment followed by per-trip routing on final loaded edge weights."""

from sumo.functions.config.loaded import write
from scripts.monitoring import monitored
from sumo.functions.source import source_digest


def configure(context):
    context.stage("sumo.config.macroscopic")
    context.stage("sumo.macro_scenario.mapped")
    context.config("random_seed")


@monitored
def execute(context):

    # Copy the assignment recipe so its outputs belong to this workflow's directory.

    return write(
        context,
        context.stage("sumo.config.macroscopic"),
        context.stage("sumo.macro_scenario.mapped"),
    )


def validate(context):
    return source_digest("sumo.functions.config.loaded")
