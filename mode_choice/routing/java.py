"""Persistent multimodal server, or Pandana cars with Java public transport."""
from pathlib import Path
import polars as pl
import matsim.runtime.eqasim as eqasim
from mode_choice.routing.server import ensure_started
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.model.parameters import load
from mode_choice.routing.congestion import input_path
from mode_choice.defaults import ROUTING_DEFAULTS, CAR_ROUTING_BACKENDS, PT_GA, PT_HALBTAX, PT_VERBUND


def configure(context):
    context.stage("mode_choice.routing.server")
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    context.stage("dmc.params")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.config("threads", volatile=True)
    for name, default in ROUTING_DEFAULTS.items():
        context.config(name, default)
    if car_backend(context) == "pandana":
        context.stage("matsim.scenario.network.mapped")


def car_backend(context):
    backend = context.config("mode_choice_car_routing_backend")
    if backend not in CAR_ROUTING_BACKENDS:
        raise ValueError(f"mode_choice_car_routing_backend must be one of {CAR_ROUTING_BACKENDS}")
    return backend


def validate(context):
    path = input_path(context, car_backend(context))
    if path is None:
        return None
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino


def read_results(path, identifier, mapping):
    frame = pl.read_csv(path, columns=[identifier, *mapping], schema_overrides={identifier: pl.String}, null_values=["NaN", "Infinity"])
    frame = frame.select(pl.col(identifier).alias("identifier"),
                         *[pl.col(source).cast(pl.Float64).alias(target) for source, target in mapping.items()])
    if frame["identifier"].n_unique() != frame.height:
        raise ValueError(f"Duplicate routing identifiers in {path}")
    return frame


def execute(context):
    try:
        return _execute(context)
    except BaseException:
        RoutingServer.stop_server()
        raise


def _execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips
    batch_size = context.config("mode_choice_routing_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("mode_choice_routing_batch_size must be a positive integer")
    backend = car_backend(context)
    p = load(context)
    directory = Path(context.path())
    config = context.stage("calibration.pt_pricing.generate_config")
    car_input = directory / "routing_requests.csv"
    trips.select("identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time").write_csv(car_input)
    ensure_started(context)
    modes = ("pt",) if backend == "pandana" else ("car", "pt", "walk", "bike")
    routed = {}
    for mode in modes:
        output = directory / f"server_{mode}_routes.csv"
        RoutingServer.route_csv(car_input, output, mode)
        frame = read_results(output, "identifier", {
            "travel_time": f"{mode}_travel_time_min",
            "travel_distance": f"{mode}_distance_km",
        }).with_columns(pl.col(f"{mode}_travel_time_min") / 60, pl.col(f"{mode}_distance_km") / 1000)
        if frame.height != trips.height or not frame["identifier"].is_in(trips["identifier"].implode()).all():
            raise ValueError(f"Server {mode} routing did not return exactly the requested trip identifiers")
        for column in (f"{mode}_travel_time_min", f"{mode}_distance_km"):
            if frame[column].null_count() or not frame[column].is_finite().all() or (frame[column] < 0).any():
                raise ValueError(f"Invalid server routing values in {column}")
        routed[mode] = frame
    if backend == "pandana":
        from mode_choice.routing.pandana import route
        car = route(context, trips, batch_size)
    else:
        car = routed["car"]
        for mode in ("walk", "bike"):
            trips = trips.join(routed[mode], on="identifier", how="left", validate="1:1")
    # Keep the server's PT totals for diagnostics. They cannot substitute for
    # DMC's separately weighted in-vehicle/access/transfer components.
    trips = trips.join(routed["pt"].rename({"pt_travel_time_min": "pt_total_travel_time_min",
                                         "pt_distance_km": "pt_total_distance_km"}),
                       on="identifier", how="left", validate="1:1")

    pt_routes_path = directory / "pt_routes.csv"
    eqasim.run(context, "org.eqasim.core.tools.routing.RunBatchPublicTransportRouter", [
        "--config-path", config, "--input-path", str(car_input),
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

    # Fare calculation uses synthetic subscriptions, matching MATSim. Route
    # components above exclude initial waiting and include transfer walking;
    # the fare calculator's waiting_time_min has different semantics.
    pt_input, pt_output = directory / "pt_price_requests.csv", directory / "pt_prices.csv"
    trips.select(
        pl.col("identifier").alias("ID"),
        *[pl.col(source).alias(target) for source, target in {
            "origin_x": "originX", "origin_y": "originY",
            "destination_x": "destinationX", "destination_y": "destinationY",
            "home_x": "homeX", "home_y": "homeY", "departure_time": "departureTime_s",
        }.items()],
        pl.col("age").cast(pl.Int64),
        pl.col("pt_subscription").is_in(PT_GA).alias("hasGA"),
        pl.col("pt_subscription").is_in(PT_HALBTAX).alias("hasHalbtaxSubscription"),
        pl.col("pt_subscription").is_in(PT_VERBUND).alias("hasVerbundAbo"),
        pl.lit(False).alias("hasStreckenAbo"),
        pl.lit(False).alias("hasGleis7Abo"),
        pl.lit(False).alias("hasJuniorAbo"),
    ).write_csv(pt_input)
    eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.utils.pricing.RunComputeTransitPrices", [
        "--config-path", config, "--requests-path", str(pt_input), "--output-path", str(pt_output),
        "--ptRegionalRadius_km", str(p["ptRegionalRadius_km"]),
    ])
    prices = read_results(pt_output, "id", {"price": "pt_cost_CHF"})
    return (trips.join(car, on="identifier", how="left", validate="1:1")
            .join(pt, on="identifier", how="left", validate="1:1")
            .join(prices, on="identifier", how="left", validate="1:1"))
