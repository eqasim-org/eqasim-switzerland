"""Export freight and light-commercial demand as native SUMO vehicle trips."""

from pathlib import Path

import pandas as pd

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

    # Normalize optional freight and commercial trips without creating person plans.

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
            frame["source_id"] = frame[key]
            frame["id"] = prefix + frame[key].map(identifier)
            frame["sumo_type"] = vehicle_type
            frames.append(frame)
    trips = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["id", "departure_time"])
    if trips["id"].duplicated().any():
        raise ValueError("Duplicate SUMO freight/commercial trip identifiers")
    trips["departure_time"] = trips["departure_time"].map(seconds)
    trips = trips.sort_values(["departure_time", "id"], kind="stable")

    # Write vehicle types and departure-sorted trips with source metadata.

    transformer = lonlat_transformer(context.config("sumo_population_crs"))
    output = Path(context.path()).resolve() / "vehicles.trips.xml.gz"
    with PopulationWriter(output, context.config("population_compresslevel")) as writer:
        writer.vehicle_type("sumo_truck", "truck")
        writer.vehicle_type("sumo_delivery", "delivery")
        for row in trips.itertuples(index=False, name=None):
            trip = dict(zip(trips.columns, row))
            origin = to_lonlat(transformer, finite_coordinates(trip["origin_x"], trip["origin_y"]))
            destination = to_lonlat(transformer, finite_coordinates(trip["destination_x"], trip["destination_y"]))
            writer.trip(trip["id"], trip["departure_time"], origin, destination,
                        vehicle_type=trip["sumo_type"], attributes=trip)

    # Return the unrouted vehicle demand file for the SUMO router.

    return str(output)
