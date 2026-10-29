"""Export freight and light-commercial demand as native SUMO vehicle trips."""

from pathlib import Path

import numpy as np
import polars as pl

from sumo.scenario.population.tools import (
    as_frame, finite_coordinates, identifier, lonlat_transformer, require_columns, seconds, to_lonlat,
)
from sumo.scenario.population.writer import PopulationWriter
from sumo.scenario import modes


def configure(context):
    modes.configure(context)
    selected = modes.selected(context.config("sumo_modes"))
    context.config("use_freight", True)
    if context.config("use_freight") and modes.includes("truck", selected):
        context.stage("synthesis.freight.trips")
    context.config("use_lcv", True)
    if context.config("use_lcv") and modes.includes("lcv", selected):
        context.stage("synthesis.lcv.trips")
    context.config("sumo_population_crs", "EPSG:2056")
    context.config("population_compresslevel", 1)


def execute(context):

    # Normalize external tables once, then prepare IDs and coordinates with Polars.

    frames = []
    selected = modes.selected(context.config("sumo_modes"))
    for enabled, stage, key, prefix, vehicle_type in (
        ("use_freight", "synthesis.freight.trips", "agent_id", "freight_", "sumo_truck"),
        ("use_lcv", "synthesis.lcv.trips", "trip_id", "lcv_", "sumo_delivery"),
    ):
        mode = "truck" if enabled == "use_freight" else "lcv"
        if context.config(enabled) and modes.includes(mode, selected):
            frame = as_frame(context.stage(stage))
            require_columns(frame, [key, "departure_time", "origin_x", "origin_y", "destination_x", "destination_y"], stage)
            frame = frame.with_columns(pl.col(key).alias("source_id"),
                (pl.lit(prefix) + pl.col(key).map_elements(identifier, return_dtype=pl.String)).alias("id"),
                pl.lit(vehicle_type).alias("sumo_type"))
            frames.append(frame)
    trips = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema={"id": pl.String, "departure_time": pl.Float64})
    if trips["id"].is_duplicated().any():
        raise ValueError("Duplicate SUMO freight/commercial trip identifiers")
    trips = trips.with_columns(pl.col("departure_time").cast(pl.Float64, strict=False))
    if trips["departure_time"].is_null().any() or (~trips["departure_time"].is_finite() | (trips["departure_time"] < 0)).any():
        raise ValueError("Commercial departures must be finite, nonnegative seconds")
    trips = trips.sort(["departure_time", "id"])

    # Project coordinates in bulk and stream the remaining XML metadata.

    transformer = lonlat_transformer(context.config("sumo_population_crs"))
    output = Path(context.path()).resolve() / "vehicles.trips.xml.gz"
    if trips.height:
        for label, first, second in (("origin", "origin_x", "origin_y"), ("destination", "destination_x", "destination_y")):
            coordinates = trips.select(pl.col(first, second).cast(pl.Float64, strict=False)).to_numpy()
            if not np.isfinite(coordinates).all():
                raise ValueError("Invalid commercial trip coordinates")
            lon, lat = transformer.transform(coordinates[:, 0], coordinates[:, 1], errcheck=True)
            if ((np.abs(lon) > 180) | (np.abs(lat) > 90)).any():
                raise ValueError("Invalid geographic commercial trip coordinates")
            trips = trips.with_columns(pl.Series("_" + label, [f"{a:.10f},{b:.10f}" for a, b in zip(lon, lat)]))
    with PopulationWriter(output, context.config("population_compresslevel")) as writer:
        writer.vehicle_type("sumo_truck", "truck")
        writer.vehicle_type("sumo_delivery", "delivery")
        for trip in trips.iter_rows(named=True):
            writer.trip(trip["id"], trip["departure_time"], trip.pop("_origin"), trip.pop("_destination"),
                        vehicle_type=trip["sumo_type"], attributes=trip)

    # Return the unrouted vehicle demand file for the SUMO router.

    return str(output)
