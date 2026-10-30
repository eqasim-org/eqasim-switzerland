"""Polars vehicle-demand preparation and Arrow IPC caches."""

import logging
from pathlib import Path
import time

import numpy as np
import polars as pl
import pyarrow as pa
import shapely

from sumo.tables import as_polars, id_expression
from sumo.scenario.population.tools import lonlat_transformer, scalar_text

logger = logging.getLogger("synpp")
CACHE_FILENAME = "vehicle_trips.arrow"
BATCH_SIZE = 250000
SCHEMA = {
    "id": pl.String, "depart": pl.Float64, "origin": pl.String, "destination": pl.String,
    "person_id": pl.String, "trip_index": pl.String, "source_trip_id": pl.String,
    "mode": pl.String, "reference": pl.Float64, "loads_network": pl.Boolean,
}


def write_trip_cache(population, path, crs, selected):
    count = 0
    started = time.monotonic()
    schema = pl.DataFrame(schema=SCHEMA).to_arrow().schema
    with pa.OSFile(str(path), "wb") as stream, pa.ipc.new_file(stream, schema) as writer:
        for batch in population_trip_batches(population, crs, selected):
            writer.write_table(batch.to_arrow())
            count += batch.height
    logger.info("Prepared SUMO IPC demand: %d trips in %.1f seconds (%.1f MB).",
                count, time.monotonic() - started, Path(path).stat().st_size / 1e6)
    return str(path)


def population_trip_batches(population, crs, selected, batch_size=BATCH_SIZE):
    activities, persons = as_polars(population["activities"]), as_polars(population["persons"])
    transformer = lonlat_transformer(crs)
    home_columns = [pl.col("person_id")]
    home_columns += [pl.col(name).cast(pl.Float64, strict=False).alias("_sumo_" + name) if name in persons.columns else pl.lit(None, dtype=pl.Float64).alias("_sumo_" + name)
                     for name in ("home_x", "home_y")]
    homes = persons.select(home_columns)
    # Build the person lookup once, not once per batch over millions of people.
    activities = activities.join(homes, on="person_id", how="left", validate="m:1", maintain_order="left")
    started, count = time.monotonic(), 0
    for begin in range(0, max(0, activities.height - 1), batch_size):
        frame = activities.slice(begin, batch_size + 1)
        same = frame["person_id"].to_numpy()[:-1] == frame["person_id"].to_numpy()[1:]
        mode = frame["following_mode"].cast(pl.String).slice(0, frame.height - 1) if "following_mode" in frame.columns else pl.Series([None] * (frame.height - 1), dtype=pl.String)
        if (same & mode.is_null().to_numpy()).any():
            raise ValueError("Missing source mode in SUMO activity chain")
        base = mode.str.strip_suffix("_loop")
        keep = same if selected is None else same & base.is_in(list(selected)).fill_null(False).to_numpy()
        positions = np.flatnonzero(keep)
        if not len(positions):
            continue
        origins, destinations = frame[positions], frame[positions + 1]
        needed = np.zeros(frame.height, dtype=bool)
        needed[positions] = needed[positions + 1] = True
        endpoint_positions = np.flatnonzero(needed)
        endpoints = frame[endpoint_positions]
        x, y, valid = _coordinates(endpoints)
        valid &= np.isfinite(x) & np.isfinite(y)
        if not valid.all():
            _location_error(frame, positions, endpoint_positions[np.flatnonzero(~valid)[0]], mode)
        lon, lat = transformer.transform(x, y)
        valid = np.isfinite(lon) & np.isfinite(lat) & (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
        if not valid.all():
            _location_error(frame, positions, endpoint_positions[np.flatnonzero(~valid)[0]], mode)
        coordinates = np.asarray([f"{a:.10f},{b:.10f}" for a, b in zip(lon, lat)], dtype=object)
        lookup = np.empty(frame.height, dtype=np.int64)
        lookup[endpoint_positions] = np.arange(len(endpoint_positions))
        depart = origins["end_time"].cast(pl.Float64, strict=False)
        if depart.is_null().any() or (~depart.is_finite() | (depart < 0)).any():
            raise ValueError("Departure must be finite, nonnegative seconds")
        arrival = destinations["start_time"].cast(pl.Float64, strict=False) if "start_time" in destinations.columns else pl.Series([None] * len(positions), dtype=pl.Float64)
        indices = origins["trip_index"].fill_null(origins["activity_index"]) if "trip_index" in origins.columns else origins["activity_index"]
        index_frame = pl.DataFrame({"trip_index": indices}).with_columns(id_expression("trip_index", indices.dtype))
        if index_frame["trip_index"].is_null().any() or index_frame["trip_index"].str.contains(r"^$|\s").any():
            raise ValueError("Invalid SUMO trip identifier")
        reference = pl.DataFrame({"arrival": arrival, "depart": depart}).select(
            pl.when(pl.col("arrival").is_finite() & (pl.col("arrival") >= pl.col("depart")))
            .then(pl.col("arrival") - pl.col("depart")).otherwise(None).alias("reference"))["reference"]
        batch = pl.DataFrame({
            "person_id": origins["person_id"].cast(pl.String), "trip_index": index_frame["trip_index"],
            "depart": depart, "origin": coordinates[lookup[positions]], "destination": coordinates[lookup[positions + 1]],
            "source_trip_id": origins["trip_id"].map_elements(scalar_text, return_dtype=pl.String) if "trip_id" in origins.columns else pl.Series([None] * len(positions), dtype=pl.String),
            "mode": mode[positions], "reference": reference,
            "loads_network": base[positions].is_in(["car", "taxi"]),
        }).with_columns(pl.concat_str(pl.lit("person_"), pl.col("person_id"), pl.lit("_"), pl.col("trip_index")).alias("id")).select(list(SCHEMA)).cast(SCHEMA)
        count += batch.height
        yield batch
        if (begin // batch_size) % 4 == 0:
            logger.info("Preparing SUMO demand: %d/%d activities, %d trips, %.1f seconds.",
                        min(begin + batch_size, activities.height), activities.height, count, time.monotonic() - started)


def trip_records(batches, all_as_car=False):
    for batch in batches:
        for trip in as_polars(batch).iter_rows(named=True):
            yield trip_record(trip, all_as_car)


def trip_record(trip, all_as_car=False):
    return {"attributes": {"id": trip["id"], "depart": str(trip["depart"]),
                          "fromLonLat": trip["origin"], "toLonLat": trip["destination"], "type": "sumo_query_car"},
           "parameters": {"person_id": trip["person_id"], "trip_index": trip["trip_index"],
                          "source_trip_id": trip["source_trip_id"], "mode": trip["mode"],
                          "source_travel_time_s": trip["reference"]},
           "load": all_as_car or trip["loads_network"]}


def _coordinates(frame):
    if "geometry" in frame.columns:
        column = frame["geometry"]
        geometry = shapely.from_wkb(column.to_numpy()) if column.dtype == pl.Binary else shapely.from_wkt(column.to_numpy())
    else:
        geometry = np.asarray([None] * frame.height, dtype=object)
    kind, empty = shapely.get_type_id(geometry), shapely.is_empty(geometry)
    fallback = (kind == -1) | empty
    x, y = shapely.get_x(geometry), shapely.get_y(geometry)
    for output, name in ((x, "destination_x"), (y, "destination_y")):
        if name in frame.columns:
            values = frame[name].cast(pl.Float64, strict=False).to_numpy()
            output[fallback] = values[fallback]
    valid = (kind <= 0) | empty
    home = frame["purpose"].eq("home").fill_null(False).to_numpy()
    if home.any():
        coordinates = frame.filter(pl.Series(home)).select("_sumo_home_x", "_sumo_home_y").to_numpy()
        replace = np.isfinite(coordinates).all(axis=1)
        indices = np.flatnonzero(home)[replace]
        x[indices], y[indices] = coordinates[replace].T
        valid[indices] = True
    return x, y, valid


def _location_error(frame, positions, endpoint, mode):
    origin = np.flatnonzero((positions == endpoint) | (positions + 1 == endpoint))[0]
    row = frame.row(int(endpoint), named=True)
    raise ValueError(f"Invalid location for SUMO person {row['person_id']}, activity {row['activity_index']}, "
                     f"selected trip mode {mode[int(positions[origin])]}: missing or invalid coordinates")
