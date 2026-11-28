"""Filter and map loading endpoints independently of macroscopic assignment."""

from pathlib import Path

from sumo.functions.config import tools
from scripts.monitoring import monitored
from sumo.functions.scenario.network.mapping import map_trips
from sumo.functions.scenario.tools import filter_loading_trips


def configure(context):
    from sumo.functions.scenario import modes

    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.macro_scenario.demand")
    context.stage("sumo.macro_scenario.connected_network")
    tools.configure_network(context)
    context.config("sumo_region_geometry", None)
    if context.config("sumo_region_geometry") is not None:
        context.stage("sumo.macro_scenario.split_network")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.stage("sumo.macro_scenario.border")
    context.config("sumo_mapmatch_distance", None)


@monitored
def execute(context):

    # Resolve the original or boundary-split full network and selected scenario demand.

    cache = Path(context.path()).resolve()
    source = context.stage("sumo.macro_scenario.demand")
    network = tools.network_path(context)
    split = None
    if context.config("sumo_region_geometry") is not None:
        split = context.stage("sumo.macro_scenario.split_network")
        network = split["network"]
    loading = cache / "loading.trips.xml.gz"
    count = filter_loading_trips(source["queries"], loading)
    if not count:
        raise ValueError(
            "No selected car/truck/LCV or other road vehicle demand to load"
        )
    if split is not None:
        from sumo.functions.cutter.tools import remap_trip_endpoints

        remap_trip_endpoints(loading, split["mapping"], cache / "split.trips.xml.gz")
        loading = cache / "split.trips.xml.gz"

    # Match every endpoint to roads connected to the mode's main directed core.

    mapped = cache / "reachable.trips.xml.gz"
    border = (
        context.stage("sumo.macro_scenario.border")
        if context.config("clip_trips_to_border_in_sumo")
        else None
    )
    endpoint_mapping = map_trips(
        loading,
        mapped,
        context.stage("sumo.macro_scenario.connected_network"),
        context.config("sumo_mapmatch_distance"),
        border,
    )
    loading = mapped

    # Expose the same mapped-demand contract to standalone config generators.

    return {
        "network": str(network),
        "trips": str(loading),
        "source_count": count,
        "queries": str(loading),
        "loading": str(loading),
        "count": count,
        "loading_count": count,
        "first_departure": source.get("first_departure"),
        "last_departure": source.get("last_departure"),
        "endpoint_mapping": endpoint_mapping,
        "boundary_split": split is not None,
    }


def validate(context):
    import hashlib

    from sumo.functions.cutter import tools as cutter_tools
    from sumo.functions.scenario import tools as scenario_tools
    from sumo.functions.scenario.network import mapping

    return [
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (mapping, cutter_tools, scenario_tools)
    ]
