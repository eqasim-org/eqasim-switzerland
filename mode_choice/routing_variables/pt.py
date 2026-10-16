"""Public transport variables; retain detailed components required by DMC."""
from pathlib import Path
import polars as pl
import matsim.runtime.eqasim as eqasim
from mode_choice.routing.java import configure_mode, route_mode, read_results
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.data.frames import join_variables
from mode_choice.defaults import PT_ROUTING_COLUMNS


def configure(context):
    configure_mode(context, "pt")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.config("threads", volatile=True)


def execute(context):
    try:
        return route_public_transport(context)
    except BaseException:
        RoutingServer.stop_server()
        raise


def route_public_transport(context):
    trips = context.stage("mode_choice.data.prepare")
    totals = route_mode(context, "pt").rename({"pt_travel_time_min": "pt_total_travel_time_min",
                                             "pt_distance_km": "pt_total_distance_km"})
    if trips.is_empty():
        return trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in PT_ROUTING_COLUMNS])
    # Until MultiModeRouterServer exposes PT components, the detailed router
    # remains necessary. Total travel time is not in-vehicle travel time.
    directory = Path(context.path())
    config = context.stage("calibration.pt_pricing.generate_config")
    batch_size = context.config("mode_choice_routing_batch_size")
    pt_routes_path = directory / "pt_routes.csv"
    eqasim.run(context, "org.eqasim.core.tools.routing.RunBatchPublicTransportRouter", [
        "--config-path", config, "--input-path", context.stage("mode_choice.routing.requests"),
        "--output-trips-path", str(pt_routes_path),
        "--threads", str(context.config("threads")),
        "--batch-size", str(batch_size),
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator",
    ])
    pt = read_results(pt_routes_path, "identifier", {
        "in_vehicle_time_total_min": "pt_in_vehicle_time_min",
        "in_vehicle_distance_total_km": "pt_in_vehicle_distance_km",
        "access_travel_time_min": "access_time", "egress_travel_time_min": "egress_time",
        "transfer_waiting_time_min": "transfer_wait", "transfer_travel_time_min": "transfer_walk",
        "transfers": "pt_transfers",
    }).with_columns(
        (pl.col("access_time") + pl.col("egress_time")).alias("pt_access_egress_time_min"),
        (pl.col("transfer_wait") + pl.col("transfer_walk")).alias("pt_transfer_time_min"),
    ).drop("access_time", "egress_time", "transfer_wait", "transfer_walk")

    return join_variables(totals, pt)
