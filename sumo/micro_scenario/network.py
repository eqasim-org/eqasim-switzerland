"""Build a region-only network from the loaded, boundary-split full scenario."""

from sumo.functions.config.tools import positive
from sumo.functions.cutter.tools import build_network, read_region
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.scenario import modes


def configure(context):
    modes.configure(context)
    context.stage("sumo.macro_scenario")
    context.stage("sumo.micro_scenario.region")
    context.stage("sumo.runtime.sumo")
    context.config("sumo_cutter_components", "largest")
    context.config("sumo_cutter_min_edge_length", 0.01)


@monitored
def execute(context):

    # Remove outside road pieces and retain a valid connected component by default.

    loaded = context.stage("sumo.macro_scenario")
    region = read_region(context.stage("sumo.micro_scenario.region"))
    return build_network(
        context,
        loaded["network"],
        region,
        keep_outside=False,
        components=context.config("sumo_cutter_components"),
        minimum_length=positive(
            context.config("sumo_cutter_min_edge_length"), "sumo_cutter_min_edge_length"
        ),
    )
