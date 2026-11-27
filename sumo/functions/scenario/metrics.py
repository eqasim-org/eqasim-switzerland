"""Write compact route metrics during realization or from existing SUMO XML."""

from contextlib import ExitStack
from pathlib import Path

import polars as pl
import pyarrow as pa

from sumo.functions.scenario.tools import elements

SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in [
            "trip_id",
            "source_trip_id",
            "person_id",
            "trip_index",
            "source_mode",
        ]
    ]
    + [
        (name, pa.float64())
        for name in [
            "departure_time_s",
            "reference_travel_time_s",
            "loaded_travel_time_s",
            "distance_m",
        ]
    ]
    + [("loads_network", pa.bool_())]
)


class MetricsWriter:
    """Keep at most one 10,000-row batch while streaming IPC output."""

    def __init__(self, path):
        self.path = str(path)
        self.rows = []
        self.count = 0
        self.schema = pl.from_arrow(pa.Table.from_batches([], schema=SCHEMA)).schema

    def __enter__(self):
        self.stack = ExitStack()
        try:
            self.stream = self.stack.enter_context(pa.OSFile(self.path, "wb"))
            self.writer = self.stack.enter_context(pa.ipc.new_file(self.stream, SCHEMA))
        except BaseException:
            self.stack.close()
            raise
        return self

    def add(self, attrs, metadata, route):
        reference = metadata.get("source_travel_time_s")
        self.rows.append(
            {
                "trip_id": attrs["id"],
                "source_trip_id": metadata.get("source_trip_id"),
                "person_id": metadata.get("person_id"),
                "trip_index": metadata.get("trip_index"),
                "source_mode": metadata.get("mode"),
                "departure_time_s": float(attrs["depart"]),
                "reference_travel_time_s": None
                if reference is None
                else float(reference),
                "loaded_travel_time_s": float(route["cost"]),
                "distance_m": float(route["routeLength"]),
                "loads_network": metadata.get("_sumo_loads_network") == "true",
            }
        )
        self.count += 1
        if len(self.rows) >= 10000:
            self.flush()

    def flush(self):
        if self.rows:
            self.writer.write_table(
                pl.DataFrame(self.rows, schema=self.schema).to_arrow().cast(SCHEMA)
            )
            self.rows.clear()

    def __exit__(self, *exc):
        try:
            if exc[0] is None:
                self.flush()
        finally:
            self.stack.__exit__(*exc)


def write_trip_metrics(routes, output, expected_count):
    """Read an existing route file when its writer cannot emit metrics directly."""
    with MetricsWriter(output) as writer:
        for vehicle in elements(routes, "vehicle"):
            route = vehicle.find("route")
            if (
                route is None
                or route.get("cost") is None
                or route.get("routeLength") is None
            ):
                raise ValueError(f"Missing route cost/length for {vehicle.get('id')}")
            metadata = {
                param.get("key"): param.get("value")
                for param in vehicle.findall("param")
            }
            writer.add(vehicle.attrib, metadata, route.attrib)
    if writer.count != expected_count:
        Path(output).unlink(missing_ok=True)
        raise ValueError(
            f"Expected {expected_count} trip metrics, received {writer.count}"
        )
    return writer.count
