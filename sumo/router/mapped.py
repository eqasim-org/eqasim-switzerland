"""Resolve geographic trip endpoints on the SUMO network before assignment."""

from pathlib import Path

from sumo.config import tools
from sumo.router.tools import filter_loading_trips
from sumo.runtime import sumo


def configure(context):
    context.stage("sumo.runtime.sumo")
    context.stage("sumo.router.demand")
    tools.configure_network(context)
    context.config("random_seed")
    context.config("sumo_mapmatch_distance", 2000)


def execute(context):

    # Resolve input files without modifying their producing stages.

    demand = context.stage("sumo.router.demand")
    if not demand["count"] or not demand["loading_count"]:
        raise ValueError("Macroscopic assignment requires at least one trip contributing road demand")
    network = tools.network_path(context)
    queries = Path(context.path()).resolve() / "queries.mapped.trips.xml.gz"

    # Map endpoints and validate route connectivity for every requested trip.

    sumo.run(context, ["--net-file", network, "--route-files", demand["queries"], "--output-file", queries,
                       "--write-trips", "true", "--ignore-errors", "false",
                       "--mapmatch.distance", tools.positive(context.config("sumo_mapmatch_distance"), "sumo_mapmatch_distance"),
                       "--seed", context.config("random_seed"), "--log", "mapmatching.log"], binary="duarouter")

    # Separate road-loading trips from the full set of car-network queries.

    loading = Path(context.path()).resolve() / "loading.trips.xml.gz"
    count = filter_loading_trips(queries, loading)
    if count != demand["loading_count"]:
        raise ValueError(f"Mapmatching changed the loading trip count: {count} != {demand['loading_count']}")
    return {**demand, "network": network, "queries": str(queries), "loading": str(loading)}


