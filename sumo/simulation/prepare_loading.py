"""Filter and map loading endpoints independently of macroscopic assignment."""

from pathlib import Path

from sumo.config import tools
from sumo.scenario.tools import filter_loading_trips
from sumo.scenario.network.mapping import map_trips


def configure(context):
    from sumo.scenario import modes
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.scenario.demand")
    context.stage("sumo.scenario.network.routing")
    tools.configure_network(context)
    context.config("sumo_region_geometry", None)
    if context.config("sumo_region_geometry") is not None:
        context.stage("sumo.cutter.split_network")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.stage("sumo.scenario.border")
    context.config("sumo_mapmatch_distance", None)


def execute(context):

    # Resolve the original or boundary-split full network and selected scenario demand.

    cache = Path(context.path()).resolve()
    source = context.stage("sumo.scenario.demand")
    network = tools.network_path(context)
    split = None
    if context.config("sumo_region_geometry") is not None:
        split = context.stage("sumo.cutter.split_network")
        network = split["network"]
    loading = cache / "loading.trips.xml.gz"
    count = filter_loading_trips(source["queries"], loading)
    if not count:
        raise ValueError("No selected car/truck/LCV or other road vehicle demand to load")
    if split is not None:
        from sumo.cutter.tools import remap_trip_endpoints
        remap_trip_endpoints(loading, split["mapping"], cache / "split.trips.xml.gz")
        loading = cache / "split.trips.xml.gz"

    # Match every endpoint to roads connected to the mode's main directed core.

    mapped = cache / "reachable.trips.xml.gz"
    border = context.stage("sumo.scenario.border") if context.config("clip_trips_to_border_in_sumo") else None
    endpoint_mapping = map_trips(loading, mapped, context.stage("sumo.scenario.network.routing"),
                                 context.config("sumo_mapmatch_distance"), border)
    loading = mapped

    return {"network": str(network), "trips": str(loading), "source_count": count,
            "endpoint_mapping": endpoint_mapping, "boundary_split": split is not None}


def validate(context):
    import hashlib
    from sumo.cutter import tools as cutter_tools
    from sumo.scenario import tools as scenario_tools
    from sumo.scenario.network import mapping
    return [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (mapping,cutter_tools,scenario_tools)]
