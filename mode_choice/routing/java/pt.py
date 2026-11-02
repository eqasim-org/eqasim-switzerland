"""Single-pass Java PT routing, with components returned by the same request."""
from pathlib import Path
import polars as pl
from mode_choice.routing.java.router import configure_mode, route_mode, read_results
from mode_choice.routing.results import complete_routes
from mode_choice.data.frames import join_variables
from mode_choice.defaults import PT_ROUTING_COLUMNS
from mode_choice.routing.options import pt_price_backend

COMPONENTS = {
    "in_vehicle_time_total_min": "pt_in_vehicle_time_min",
    "in_vehicle_distance_total_km": "pt_in_vehicle_distance_km",
    "access_travel_time_min": "access_time", "egress_travel_time_min": "egress_time",
    "transfer_waiting_time_min": "transfer_wait", "transfer_travel_time_min": "transfer_walk",
    "transfers": "pt_transfers",
}


def configure(context):
    configure_mode(context, "pt")


def execute(context):

    # Request totals, utility components and optional fares in one routing pass.

    trips = context.stage("mode_choice.data.prepare")
    totals = route_mode(context, "pt", compute_cost=pt_price_backend(context) == "java").rename({
        "pt_travel_time_min": "pt_total_travel_time_min", "pt_distance_km": "pt_total_distance_km"})
    if trips.is_empty():
        return trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in PT_ROUTING_COLUMNS])

    # Read numerical components directly; itinerary export and pricing are independent.

    components = read_results(Path(context.path()) / "pt_routes.csv", "identifier", COMPONENTS)
    components = complete_routes(trips, components, ["pt_in_vehicle_time_min", "access_time", "egress_time",
                                                    "transfer_wait", "transfer_walk"]).with_columns(
        (pl.col("access_time") + pl.col("egress_time")).alias("pt_access_egress_time_min"),
        (pl.col("transfer_wait") + pl.col("transfer_walk")).alias("pt_transfer_time_min"),
    ).drop("access_time", "egress_time", "transfer_wait", "transfer_walk")
    return complete_routes(trips, join_variables(totals, components), [
        "pt_total_travel_time_min", "pt_in_vehicle_time_min", "pt_access_egress_time_min", "pt_transfer_time_min"])
