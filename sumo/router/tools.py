"""Routing metrics and compatibility imports for shared scenario writers."""

from pathlib import Path

import pyarrow as pa
import polars as pl

from sumo.scenario.tools import elements, write_trips, filter_loading_trips


def write_trip_metrics(routes, output, expected_count):
    schema = pa.schema([
        ("trip_id", pa.string()), ("source_trip_id", pa.string()), ("person_id", pa.string()),
        ("trip_index", pa.string()), ("source_mode", pa.string()),
        ("departure_time_s", pa.float64()), ("reference_travel_time_s", pa.float64()),
        ("loaded_travel_time_s", pa.float64()), ("distance_m", pa.float64()),
        ("loads_network", pa.bool_()),
    ])
    polars_schema = pl.from_arrow(pa.Table.from_batches([], schema=schema)).schema
    count, batch = 0, []
    with pa.OSFile(str(output), "wb") as stream, pa.ipc.new_file(stream, schema) as writer:
        for vehicle in elements(routes, "vehicle"):
            route = vehicle.find("route")
            if route is None or route.get("cost") is None or route.get("routeLength") is None:
                raise ValueError(f"Missing route cost/length for {vehicle.get('id')}")
            metadata = {param.get("key"): param.get("value") for param in vehicle.findall("param")}
            reference = metadata.get("source_travel_time_s")
            batch.append({
                "trip_id": vehicle.get("id"), "source_trip_id": metadata.get("source_trip_id"),
                "person_id": metadata.get("person_id"), "trip_index": metadata.get("trip_index"),
                "source_mode": metadata.get("mode"), "departure_time_s": float(vehicle.get("depart")),
                "reference_travel_time_s": None if reference is None else float(reference),
                "loaded_travel_time_s": float(route.get("cost")), "distance_m": float(route.get("routeLength")),
                "loads_network": metadata.get("_sumo_loads_network") == "true",
            })
            count += 1
            if len(batch) >= 10000:
                writer.write_table(pl.DataFrame(batch, schema=polars_schema).to_arrow().cast(schema))
                batch.clear()
        if batch:
            writer.write_table(pl.DataFrame(batch, schema=polars_schema).to_arrow().cast(schema))
    if count != expected_count:
        Path(output).unlink(missing_ok=True)
        raise ValueError(f"Expected {expected_count} trip metrics, received {count}")
    return count
