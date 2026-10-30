"""Residence-canton expected car demand, independent of population sampling."""
import logging
from pathlib import Path
import polars as pl

from mode_choice.calibration.shares import load_canton_targets
from mode_choice.defaults import MODES

logger = logging.getLogger(__name__)


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.routing.java.freight")
    context.stage("data.microcensus.shares")


def execute(context):

    # Count every eligible trip, including people who cannot drive.

    trips = context.stage("mode_choice.data.prepare")
    _, target_path = context.stage("data.microcensus.shares")
    targets = load_canton_targets(target_path)
    cars, summary = car_weights(trips, targets)

    # Freight and LCV are already vehicle demand, so retain their full weight.

    freight = context.stage("mode_choice.routing.java.freight").select(
        "identifier", "mode", pl.lit(1.0).alias("weight"))
    summary.write_parquet(Path(context.path()) / "canton_weights.parquet")
    logger.info("MARouter canton demand: %s", summary.to_dicts())
    return pl.concat([cars, freight])


def car_weights(trips, targets):
    if trips["canton_name_en"].null_count():
        raise ValueError("MARouter weights require residence cantons for every trip")
    missing = set(trips["canton_name_en"]) - set(targets)
    if missing:
        raise ValueError(f"Missing car-share targets for residence cantons: {sorted(missing)}")
    shares = {name: float(values[MODES.index("car")]) for name, values in targets.items()}
    counts = trips.group_by("canton_name_en").agg(
        pl.len().alias("trips"), pl.col("car_availability").fill_null(False).sum().alias("available_trips"),
    ).with_columns(pl.col("canton_name_en").replace_strict(shares, return_dtype=pl.Float64).alias("car_share"))
    counts = counts.with_columns((pl.col("trips") * pl.col("car_share")).alias("target_car_trips"))
    impossible = counts.filter((pl.col("available_trips") == 0) & (pl.col("target_car_trips") > 0))
    if impossible.height:
        raise ValueError(f"Positive car demand but no available drivers: {impossible.to_dicts()}")
    counts = counts.with_columns(
        pl.when(pl.col("available_trips") > 0).then(pl.col("target_car_trips") / pl.col("available_trips"))
        .otherwise(0.0).alias("driver_weight"))
    if counts.filter(pl.col("driver_weight") > 1).height:
        logger.warning("Canton targets exceed available driver trips; assignment weights exceed one")
    result = trips.join(counts.select("canton_name_en", "driver_weight"), on="canton_name_en", validate="m:1",
                        maintain_order="left").select(
        pl.col("trip_id").alias("identifier"),
        pl.when(pl.col("car_availability").fill_null(False)).then(pl.lit("car"))
        .otherwise(pl.lit("car_passenger")).alias("mode"),
        pl.when(pl.col("car_availability").fill_null(False)).then(pl.col("driver_weight")).otherwise(0.0).alias("weight"))
    return result, counts.sort("canton_name_en")
