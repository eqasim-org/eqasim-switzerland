"""Subscription-aware PT fares, cached independently from routing variables."""
from pathlib import Path
import polars as pl
import matsim.runtime.eqasim as eqasim
from mode_choice.defaults import PT_GA, PT_HALBTAX, PT_VERBUND
from mode_choice.model.parameters import load
from mode_choice.routing.java import read_results
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.defaults")
    context.stage("dmc.params")
    context.stage("calibration.pt_pricing.generate_config")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")


def execute(context):
    try:
        return _execute(context)
    except BaseException:
        RoutingServer.stop_server()
        raise


def _execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips.select("trip_id").with_columns(pl.lit(None, dtype=pl.Float64).alias("pt_cost_CHF"))
    directory = Path(context.path())
    config = context.stage("calibration.pt_pricing.generate_config")
    p = load(context)
    pt_input, pt_output = directory / "pt_price_requests.csv", directory / "pt_prices.csv"
    # The fare tool accepts integer IDs, whereas public stage keys are strings.
    # Keep a lossless local mapping instead of parsing/truncating person IDs.
    requests = trips.with_row_index("fare_id")
    requests.select(
        pl.col("fare_id").alias("ID"),
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
    join_variables(requests.select(pl.col("fare_id").cast(pl.String).alias("trip_id")), prices)
    prices = (requests.select("trip_id", pl.col("fare_id").cast(pl.String))
              .join(prices.rename({"trip_id": "fare_id"}), on="fare_id", how="left", validate="1:1")
              .drop("fare_id"))
    return join_variables(trips.select("trip_id"), prices)
