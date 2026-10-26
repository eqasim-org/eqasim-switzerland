"""Build a region-only network from the loaded, boundary-split full scenario."""

from sumo.config.tools import positive
from sumo.cutter.tools import build_network, read_region
from sumo.scenario import modes


def configure(context):
    modes.configure(context)
    context.stage("sumo.simulation.load_network")
    context.stage("sumo.cutter.geometry")
    context.stage("sumo.runtime.sumo")
    context.config("sumo_cutter_components", "largest")
    context.config("sumo_cutter_min_edge_length", 0.01)


def execute(context):

    # Remove outside road pieces and retain a valid connected component by default.

    loaded = context.stage("sumo.simulation.load_network")
    region = read_region(context.stage("sumo.cutter.geometry"))
    return build_network(context, loaded["network"], region, keep_outside=False,
                         components=context.config("sumo_cutter_components"),
                         minimum_length=positive(context.config("sumo_cutter_min_edge_length"), "sumo_cutter_min_edge_length"))
