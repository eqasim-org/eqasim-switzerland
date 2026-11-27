"""Split the full network at region boundaries before loading for timed crossings."""

from sumo.functions.config import tools as config_tools
from sumo.functions.cutter.tools import build_network, read_region
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.scenario import modes


def configure(context):
    modes.configure(context)
    config_tools.configure_network(context)
    context.stage("sumo.micro_scenario.region")
    context.stage("sumo.runtime.sumo")


@monitored
def execute(context):

    # Insert boundary nodes in the full network while keeping both sides of every crossing.

    region = read_region(context.stage("sumo.micro_scenario.region"))
    return build_network(
        context, config_tools.network_path(context), region, keep_outside=True
    )
