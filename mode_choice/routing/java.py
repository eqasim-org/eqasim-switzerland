"""Shared Java transport helpers; per-mode stages live in routing_variables."""
from pathlib import Path
import polars as pl
from mode_choice.defaults import ROUTING_DEFAULTS, CAR_ROUTING_BACKENDS, ROUTING_COST_FILENAME
from mode_choice.routing.server import ensure_server_started
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.routing.results import complete_routes


def car_backend(context):
    backend = context.config("mode_choice_car_routing_backend")
    if backend not in CAR_ROUTING_BACKENDS:
        raise ValueError(f"mode_choice_car_routing_backend must be one of {CAR_ROUTING_BACKENDS}")
    return backend


def configure_mode(context, mode):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    for name, default in ROUTING_DEFAULTS.items():
        context.config(name, default)
    if car_backend(context) == "java" or mode == "pt":
        context.stage("mode_choice.routing.server")
        context.stage("mode_choice.routing.requests")


def read_results(path, identifier, mapping):
    try:
        frame = pl.read_csv(path, columns=[identifier, *mapping], schema_overrides={identifier: pl.String},
                            null_values=["NaN", "nan", "Infinity"])
    except pl.exceptions.NoDataError:
        frame = pl.DataFrame(schema={identifier: pl.String, **{column: pl.Float64 for column in mapping}})
    frame = frame.select(pl.col(identifier).alias("trip_id"),
                         *[pl.col(source).cast(pl.Float64).alias(target) for source, target in mapping.items()])
    if frame["trip_id"].null_count() or frame["trip_id"].n_unique() != frame.height:
        raise ValueError(f"Null or duplicate routing identifiers in {path}")
    return frame


def route_mode(context, mode):
    trips = context.stage("mode_choice.data.prepare")
    columns = {"travel_time": f"{mode}_travel_time_min", "travel_distance": f"{mode}_distance_km"}
    compute_cost = mode in ("car", "pt")
    if compute_cost:
        columns["cost"] = f"{mode}_cost_CHF"
    if trips.is_empty():
        result = trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in columns.values()])
        return save_costs(context, result, mode) if compute_cost else result
    try:
        ensure_server_started(context)
        output = Path(context.path()) / f"{mode}_routes.csv"
        RoutingServer.route_csv(context.stage("mode_choice.routing.requests"), output, mode, compute_cost=compute_cost)
        result = read_results(output, "identifier", columns).with_columns(
            pl.col(f"{mode}_travel_time_min") / 60, pl.col(f"{mode}_distance_km") / 1000)
        result = complete_routes(trips, result, [f"{mode}_travel_time_min"])
        return save_costs(context, result, mode) if compute_cost else result
    except BaseException:
        RoutingServer.stop_server()
        raise


def save_costs(context, result, mode):
    """Persist monetary output separately; routing stages return only variables."""
    column = f"{mode}_cost_CHF"
    result.select("trip_id", column).write_parquet(Path(context.path()) / ROUTING_COST_FILENAME)
    return result.drop(column)
