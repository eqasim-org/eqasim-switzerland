"""Split the full network at region boundaries before loading for timed crossings."""

from sumo.config import tools as config_tools
from sumo.cutter.tools import build_network, read_region
from sumo.scenario import modes


def configure(context):
    modes.configure(context)
    config_tools.configure_network(context)
    context.stage("sumo.cutter.geometry")
    context.stage("sumo.runtime.sumo")


def execute(context):

    # Insert boundary nodes in the full network while keeping both sides of every crossing.

    region = read_region(context.stage("sumo.cutter.geometry"))
    return build_network(context, config_tools.network_path(context), region, keep_outside=True)
