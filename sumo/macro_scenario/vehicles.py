"""Write prepared freight and LCV trips in SUMO format."""

from pathlib import Path

import numpy as np

from sumo.functions.monitoring.runtime import monitored
from sumo.functions.scenario.population.tools import lonlat_transformer
from sumo.functions.scenario.population.writer import PopulationWriter
from sumo.functions.tables import read_table


def configure(context):
    context.stage("sumo.macro_scenario.population")
    context.config("sumo_population_crs", "EPSG:2056")
    context.config("population_compresslevel", 1)


@monitored
def execute(context):

    # Read compact, mode-filtered trips from the prepared-population adapter.

    frame = read_table(
        context.stage("sumo.macro_scenario.population")["commercial_trips"]
    )
    if frame["id"].is_duplicated().any():
        raise ValueError("Duplicate prepared commercial SUMO trip IDs")
    values = frame.select(
        "departure_time", "origin_x", "origin_y", "destination_x", "destination_y"
    ).to_numpy()
    if not np.isfinite(values).all() or (values[:, 0] < 0).any():
        raise ValueError(
            "Prepared commercial coordinates/times must be finite; departures must be nonnegative"
        )
    frame = frame.sort("departure_time", "id")

    # Project endpoints in bulk; retain source keys for route reuse and auditing.

    transform = lonlat_transformer(context.config("sumo_population_crs"))
    origin = transform.transform(
        frame["origin_x"].to_numpy(), frame["origin_y"].to_numpy(), errcheck=True
    )
    destination = transform.transform(
        frame["destination_x"].to_numpy(),
        frame["destination_y"].to_numpy(),
        errcheck=True,
    )
    target = Path(context.path()) / "vehicles.trips.xml.gz"
    with PopulationWriter(target, context.config("population_compresslevel")) as writer:
        writer.vehicle_type("sumo_truck", "truck")
        writer.vehicle_type("sumo_delivery", "delivery")
        for index, row in enumerate(frame.iter_rows(named=True)):
            writer.trip(
                row["id"],
                row["departure_time"],
                f"{origin[0][index]:.10f},{origin[1][index]:.10f}",
                f"{destination[0][index]:.10f},{destination[1][index]:.10f}",
                vehicle_type=row["sumo_type"],
                attributes={**row, "source_trip_id": row["source_id"]},
            )

    # Return the vehicle file without exposing external dataframe caches.

    return str(target.resolve())
