"""Cache mode-specific endpoint reachability on the full assignment network."""

import hashlib
from pathlib import Path

from sumo.config import tools
from sumo.scenario import modes
from sumo.scenario.network import mapping


def configure(context):
    tools.configure_network(context)
    modes.configure(context)
    context.config("sumo_region_geometry", None)
    if context.config("sumo_region_geometry") is not None:
        context.stage("sumo.cutter.split_network")


def execute(context):

    # Index the exact network that native assignment will use, including splits.

    network = (context.stage("sumo.cutter.split_network")["network"]
               if context.config("sumo_region_geometry") is not None else tools.network_path(context))
    classes = modes.vehicle_classes(modes.selected(context.config("sumo_modes")))

    # Cache compact columnar endpoint masks instead of a pickled network graph.

    return mapping.build_index(network, context.path(), classes)


def validate(context):
    return hashlib.sha256(Path(mapping.__file__).read_bytes()).hexdigest()
