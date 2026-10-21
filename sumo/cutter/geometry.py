"""Read polygon boundaries and transform them into the full network coordinate frame."""

from pathlib import Path

from shapely import to_wkb

from sumo.config import tools as config_tools
from sumo.cutter.tools import geometry_token, load_region
from sumo.scenario.network.tools import read_network


def configure(context):
    config_tools.configure_network(context)
    context.config("sumo_region_geometry", None)
    context.config("sumo_region_layer", None)
    context.config("sumo_region_crs", None)
    context.config("sumo_network_crs", None)
    if context.config("sumo_region_geometry") is None:
        raise ValueError("Set sumo_region_geometry to a polygon file before running the cutter")


def execute(context):

    # Read the source polygon and use the network's projection and local offset.

    source = context.config("sumo_region_geometry")
    if source is None:
        raise ValueError("Set sumo_region_geometry to a polygon file before running the cutter")
    network = config_tools.network_path(context)
    region = load_region(source, context.config("sumo_region_layer"), read_network(network)["location"],
                         context.config("sumo_region_crs"), context.config("sumo_network_crs"))

    # Cache the combined polygon, retaining holes and disconnected polygon parts.

    output = Path(context.path()).resolve() / "region.network.wkb"
    output.write_bytes(to_wkb(region))
    return {"geometry": str(output), "source": str(Path(source).resolve()), "bounds": list(region.bounds)}


def validate(context):
    return geometry_token(context.config("sumo_region_geometry"))
