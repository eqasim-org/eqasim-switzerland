"""Resolve geographic trip endpoints on the SUMO network before assignment."""

from pathlib import Path

from sumo.config import tools
from sumo.router.tools import filter_loading_trips
from sumo.runtime import sumo
from sumo.scenario import modes
from sumo.scenario.network.mapping import build_index, map_trips


def configure(context):
    context.stage("sumo.runtime.sumo")
    context.stage("sumo.router.demand")
    tools.configure_network(context)
    context.config("random_seed")
    context.config("sumo_mapmatch_distance", None)
    modes.configure(context)
    if context.config("clip_trips_to_border_in_sumo", True):
        context.stage("sumo.scenario.border")


def execute(context):

    # Resolve input files without modifying their producing stages.

    demand = context.stage("sumo.router.demand")
    if not demand["count"] or not demand["loading_count"]:
        raise ValueError("Macroscopic assignment requires at least one trip contributing road demand")
    network = tools.network_path(context)
    queries = Path(context.path()).resolve() / "queries.mapped.trips.xml.gz"

    # Resolve coordinates with the shared expanding, mode-aware road matcher.

    index = build_index(network, context.path(), modes.vehicle_classes(modes.selected(context.config("sumo_modes"))))
    reachable = Path(context.path()).resolve() / "queries.reachable.trips.xml.gz"
    border = context.stage("sumo.scenario.border") if context.config("clip_trips_to_border_in_sumo") else None
    map_trips(demand["queries"], reachable, index, context.config("sumo_mapmatch_distance"), border)

    # Map endpoints and validate route connectivity for every requested trip.

    sumo.run(context, ["--net-file", network, "--route-files", reachable, "--output-file", queries,
                       "--write-trips", "true", "--ignore-errors", "false",
                       "--seed", context.config("random_seed"), "--log", "mapmatching.log"], binary="duarouter")

    # Separate road-loading trips from the full set of car-network queries.

    loading = Path(context.path()).resolve() / "loading.trips.xml.gz"
    count = filter_loading_trips(queries, loading)
    if count != demand["loading_count"]:
        raise ValueError(f"Mapmatching changed the loading trip count: {count} != {demand['loading_count']}")
    return {**demand, "network": network, "queries": str(queries), "loading": str(loading)}


