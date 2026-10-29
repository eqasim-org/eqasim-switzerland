"""Parallel Polars sorting and optional native bulk XML export."""

import logging
from pathlib import Path
import subprocess
import time

from lxml import etree
import polars as pl

from sumo.tables import as_polars, write_table
from sumo.scenario.population.vehicle_tools import SCHEMA, trip_records
from sumo.scenario.tools import trip_element, write_serialized_trips

logger = logging.getLogger("synpp")
DEMAND_SCHEMA = {**SCHEMA, "xml": pl.Binary}


class TripSpool:
    """Hold Arrow-backed columns in RAM and sort them in parallel, without SQLite."""

    def __init__(self, path, batch_size=100000, native_writer=None, clipper=None, endpoint_resolver=None):
        self.path, self.batch_size, self.native_writer = Path(path), batch_size, native_writer
        self.clipper = clipper
        self.endpoint_resolver = endpoint_resolver
        self.pending, self.parts = [], []
        self.count, self.loading_count = 0, 0
        self.started = time.monotonic()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.parts.clear()
        self.pending.clear()

    def add_population(self, frame, all_as_car=False):
        frame = as_polars(frame)
        if self.clipper is not None:
            frame = self.clipper(frame)
        if all_as_car:
            frame = frame.with_columns(pl.lit(True).alias("loads_network"))
        self.add_frame(frame.with_columns(pl.lit(None, dtype=pl.Binary).alias("xml")))
        logger.info("SUMO demand: added %d columnar population trips without Python XML objects.", frame.height)

    def add_frame(self, frame):
        self.parts.append(frame.select(list(DEMAND_SCHEMA)).cast(DEMAND_SCHEMA))
        self.count += frame.height
        self.loading_count += int(frame["loads_network"].sum() or 0)

    def append(self, record):
        record = {**record, "parameters": {**record["parameters"],
                  "_sumo_loads_network": "true" if record["load"] else "false"}}
        attributes = record["attributes"]
        row = {"id": attributes["id"], "depart": float(attributes["depart"]),
               "loads_network": bool(record["load"]), "xml": etree.tostring(trip_element(record))}
        resolver = self.clipper or self.endpoint_resolver
        if resolver is not None:
            row["origin"], row["destination"] = resolver.xml_coordinates(attributes)
            row["person_id"] = str(record["parameters"].get("person_id") or attributes["id"])
            row["trip_index"] = str(record["parameters"].get("trip_index") or "0")
            row["source_trip_id"] = str(record["parameters"].get("source_trip_id") or attributes["id"])
            row["mode"] = str(record["parameters"].get("mode") or "car")
            reference = record["parameters"].get("source_travel_time_s")
            try:
                row["reference"] = float(reference)
            except (TypeError, ValueError):
                row["reference"] = None
        self.pending.append(row)
        self.count += 1
        self.loading_count += bool(record["load"])
        if len(self.pending) >= self.batch_size:
            self._flush()

    def write(self, path, types, write_xml=True):
        self._flush()
        frame = pl.concat(self.parts, how="vertical") if self.parts else pl.DataFrame(schema=DEMAND_SCHEMA)
        self.parts.clear()
        logger.info("SUMO demand: validating and sorting %d trips with Polars in RAM.", frame.height)
        if frame["id"].n_unique() != frame.height:
            raise ValueError("Duplicate trip IDs across SUMO assignment inputs")
        frame = frame.sort(["depart", "id"])
        table = write_table(frame, self.path)
        logger.info("SUMO demand: writing %d trips using %s.", frame.height, "C++/Arrow" if self.native_writer else "lxml")
        prelude = self.path.with_suffix(".types.xml")
        prelude.write_bytes(b"".join(etree.tostring(element) for element in types.values()))
        if not write_xml:
            pass
        elif self.native_writer:
            subprocess.run([self.native_writer, table, str(prelude), str(path)], check=True)
        else:
            def payloads():
                for batch in frame.iter_slices(100000):
                    population = batch.filter(pl.col("xml").is_null())
                    # Preserve the global departure order when external and population rows interleave.
                    records = iter(trip_records([population]))
                    for payload in batch["xml"]:
                        if payload is None:
                            record = next(records)
                            record["parameters"]["_sumo_loads_network"] = "true" if record["load"] else "false"
                            yield etree.tostring(trip_element(record))
                        else:
                            yield payload
            write_serialized_trips(path, types, payloads())
        logger.info("SUMO demand finished: %d trips in %.1f seconds.", frame.height, time.monotonic() - self.started)
        return {"queries": str(path) if write_xml else None, "types": str(prelude), "table": table, "count": frame.height, "loading_count": self.loading_count,
                "first_departure": frame["depart"][0] if frame.height else None,
                "last_departure": frame["depart"][-1] if frame.height else None}

    def _flush(self):
        if self.pending:
            frame = pl.DataFrame(self.pending, schema=DEMAND_SCHEMA)
            if self.clipper is not None:
                clipped = self.clipper(frame)
                self.count += clipped.height - frame.height
                self.loading_count += int(clipped["loads_network"].sum() or 0) - int(frame["loads_network"].sum() or 0)
                frame = clipped
            self.parts.append(frame)
            self.pending.clear()
