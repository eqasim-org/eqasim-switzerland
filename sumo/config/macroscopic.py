"""Generate a native marouter configuration for flow-dependent road assignment."""

from sumo.functions.config import tools
from sumo.functions.config.macroscopic import configure_profile, write
from sumo.functions.config.macroscopic import time_bounds as time_bounds
from scripts.monitoring import monitored
from sumo.functions.source import source_digest


def configure(context):
    context.stage("sumo.macro_scenario.mapped")
    configure_profile(context)


@monitored
def execute(context):

    # Write the shared assignment recipe for mapped scenario demand.

    return write(context, context.stage("sumo.macro_scenario.mapped"))


def validate(context):
    return {
        "inputs": (
            tools.external_token(context, ("sumo_assignment_previous_weights",))
        ),
        "helpers": source_digest("sumo.functions.config.macroscopic"),
    }
