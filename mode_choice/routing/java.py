"""Java PT routing/fares with selectable Java or Pandana car routing."""
from pathlib import Path
import polars as pl
import matsim.runtime.eqasim as eqasim
from mode_choice.model.parameters import load
from mode_choice.routing.congestion import input_path
from mode_choice.defaults import ROUTING_DEFAULTS, CAR_ROUTING_BACKENDS, PT_GA, PT_HALBTAX, PT_VERBUND


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    context.stage("dmc.params")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.config("threads")
    for name, default in ROUTING_DEFAULTS.items():
        context.config(name, default)
    if car_backend(context) == "pandana":
        context.stage("matsim.scenario.network.mapped")


def car_backend(context):
    backend = context.config("mode_choice_car_routing_backend", ROUTING_DEFAULTS["mode_choice_car_routing_backend"])
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
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips
    batch_size = context.config("mode_choice_routing_batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("mode_choice_routing_batch_size must be a positive integer")
    backend = car_backend(context)
    congestion_path = input_path(context, backend)
    p = load(context)
    directory = Path(context.path())
    config = context.stage("calibration.pt_pricing.generate_config")
    car_input, car_output = directory / "car_requests.csv", directory / "car_routes.csv"
    trips.select("identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time").write_csv(car_input)
    if backend == "pandana":
        from mode_choice.routing.pandana import route
        car = route(context, trips, batch_size)
    else:
        eqasim.run(context, "org.eqasim.core.tools.routing.RunBatchRoadRouter", [
            "--config-path", config, "--input-path", str(car_input), "--output-path", str(car_output),
            "--threads", str(context.config("threads")),
            "--batch-size", str(batch_size),
            "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator",
        ] + (["--events-path", str(congestion_path)] if congestion_path is not None else []))
        car = read_results(car_output, "identifier", {
            "in_vehicle_time_min": "car_travel_time_min",
            "in_vehicle_distance_km": "car_distance_km",
        })

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
