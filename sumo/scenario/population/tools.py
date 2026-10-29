"""Polars population preparation, identifiers, coordinates and XML streams."""

from contextlib import contextmanager
from datetime import date, datetime, time as datetime_time, timedelta
import gzip
import logging
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import polars as pl
from pyproj import Transformer
from shapely import wkt, from_wkb

from sumo.tables import as_polars, id_expression

ACTIVITY_KEY = ["person_id", "activity_index"]
logger = logging.getLogger("synpp")


def as_frame(value):
    """Compatibility name for the shared external-to-Polars boundary."""
    return as_polars(value)


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def scalar_text(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (date, datetime, datetime_time, timedelta)):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return format(value, ".15g") if math.isfinite(value) else None
    return str(value) if np.isscalar(value) else None


def identifier(value):
    result = str(int(value)) if isinstance(value, Real) and math.isfinite(value) and float(value).is_integer() else scalar_text(value)
    if result is None or not result or any(character.isspace() for character in result):
        raise ValueError(f"Invalid SUMO identifier: {value!r}")
    return result


def seconds(value, label="time"):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite, nonnegative seconds: {value!r}")
    return result


def reference_travel_time(arrival, departure):
    try:
        arrival = float(arrival)
    except (TypeError, ValueError):
        return None
    return arrival - departure if math.isfinite(arrival) and arrival >= departure else None


def prepare_activities(activities, locations, trips):
    activities, locations, trips = map(as_polars, (activities, locations, trips))
    require_columns(activities, ACTIVITY_KEY + ["start_time", "end_time"], "Activities")
    require_columns(locations, ACTIVITY_KEY + ["geometry"], "Activity locations")
    require_columns(trips, ["person_id", "trip_index", "mode", "departure_time", "arrival_time"], "Trips")
    activities = _prepare_keys(activities, "activity_index")
    locations = _prepare_keys(locations, "activity_index")
    trips = _prepare_keys(trips, "trip_index")
    columns = ["person_id", "trip_index", "mode", "departure_time", "arrival_time"]
    columns += [column for column in ("trip_id", "passenger_vehicle_id") if column in trips.columns]
    following = trips.select(columns).rename({"trip_index": "activity_index", "mode": "following_mode"}).with_columns(
        pl.col("activity_index").alias("trip_index"), pl.lit(True).alias("_matched_trip"))
    activities = activities.drop([column for column in following.columns if column in activities.columns and column not in ACTIVITY_KEY])
    activities = activities.join(following, on=ACTIVITY_KEY, how="left", validate="1:1", maintain_order="left").with_columns(
        pl.when(pl.col("_matched_trip").fill_null(False)).then(pl.col("departure_time")).otherwise(pl.col("end_time")).alias("end_time")
    ).drop("_matched_trip").with_columns((pl.col("end_time") - pl.col("start_time")).alias("duration"))
    location_columns = [column for column in locations.columns if column not in activities.columns or column in ACTIVITY_KEY]
    return activities.join(locations.select(location_columns), on=ACTIVITY_KEY, how="left", validate="1:1", maintain_order="left")


def _prepare_keys(frame, index):
    values = frame[index].cast(pl.Float64, strict=False)
    if values.is_null().any() or (~values.is_finite() | (values < 0) | (values % 1 != 0)).any():
        raise ValueError(f"Invalid {index} in SUMO population inputs")
    return frame.with_columns(id_expression("person_id", frame.schema["person_id"]), pl.col(index).cast(pl.Int64))


def combine_populations(populations):
    person_frames, activity_frames = [], []
    for persons, activities, source in populations:
        persons, activities = map(as_polars, (persons, activities))
        require_columns(persons, ["person_id"], f"{source} persons")
        require_columns(activities, ACTIVITY_KEY + ["purpose"], f"{source} activities")
        persons = persons.with_columns(id_expression("person_id", persons.schema["person_id"]), pl.lit(source).alias("population_source"))
        activities = activities.with_columns(id_expression("person_id", activities.schema["person_id"]))
        indices = activities["activity_index"].cast(pl.Float64, strict=False)
        if indices.is_null().any() or (~indices.is_finite() | (indices < 0) | (indices % 1 != 0)).any():
            raise ValueError(f"Invalid activity_index in {source} population")
        activities = activities.with_columns(pl.col("activity_index").cast(pl.Int64))
        for frame in (persons, activities):
            if frame["person_id"].is_null().any() or frame["person_id"].str.contains(r"^$|\s").any():
                raise ValueError("Invalid SUMO person identifier")
        person_frames.append(persons)
        activity_frames.append(activities)
    persons = pl.concat(person_frames, how="diagonal_relaxed")
    activities = pl.concat(activity_frames, how="diagonal_relaxed")
    if persons["person_id"].is_duplicated().any():
        raise ValueError("Duplicate person_id across SUMO populations")
    if activities.select(pl.struct(ACTIVITY_KEY).is_duplicated().any()).item():
        raise ValueError("Duplicate person_id/activity_index in SUMO activities")
    ids = activities.select("person_id").unique()
    if ids.join(persons.select("person_id"), on="person_id", how="anti").height:
        raise ValueError("SUMO activities reference persons absent from the population")
    if persons.select("person_id").join(ids, on="person_id", how="anti").height:
        raise ValueError("SUMO persons are missing their activity chain")
    if "start_time" not in activities.columns:
        activities = activities.with_columns(pl.lit(None, dtype=pl.Float64).alias("start_time"))
    first = activities.group_by("person_id").agg(pl.col("start_time").sort_by("activity_index").first().fill_null(0).alias("sumo_depart"))
    persons = persons.join(first, on="person_id", how="left").with_columns(pl.col("sumo_depart").cast(pl.Float64))
    if persons.filter(~pl.col("sumo_depart").is_finite() | (pl.col("sumo_depart") < 0)).height:
        raise ValueError("Population departure must be finite, nonnegative seconds")
    persons = persons.sort(["sumo_depart", "person_id"])
    activities = activities.join(persons.select("person_id").with_row_index("_person_order"), on="person_id", how="left", validate="m:1").sort(["_person_order", "activity_index"]).drop("_person_order")
    return {"persons": persons, "activities": activities}


def coalesce_external_activities(activities):
    activities = as_polars(activities)
    duplicates = activities.filter(pl.struct(ACTIVITY_KEY).is_duplicated())
    if not duplicates.height:
        return activities
    compared = [column for column in activities.columns if column not in ACTIVITY_KEY + ["following_mode"]]
    expressions = [pl.struct(compared).n_unique().alias("_activities")]
    if "following_mode" in activities.columns:
        expressions.append(pl.col("following_mode").cast(pl.String).str.strip_suffix("_loop").n_unique().alias("_modes"))
    grouped = duplicates.group_by(ACTIVITY_KEY).agg(expressions)
    if grouped.filter((pl.col("_activities") > 1) | (pl.col("_modes") > 1) if "_modes" in grouped.columns else pl.col("_activities") > 1).height:
        raise ValueError("Duplicate person_id/activity_index in external SUMO activities with conflicting activity data or transport modes")
    result = activities.unique(subset=ACTIVITY_KEY, keep="first", maintain_order=True)
    logger.info("Collapsed %d equivalent external activity rows across %d keys for SUMO.", activities.height - result.height, grouped.height)
    return result


def activity_coordinates(activity, person):
    """Read source coordinates, using person home coordinates for home activities."""
    if activity.get("purpose") == "home":
        x, y = person.get("home_x"), person.get("home_y")
        if scalar_text(x) is not None and scalar_text(y) is not None:
            return finite_coordinates(x, y)
    geometry = activity.get("geometry")
    if isinstance(geometry, (bytes, bytearray, memoryview)):
        geometry = from_wkb(bytes(geometry))
    if isinstance(geometry, str):
        geometry = wkt.loads(geometry)
    if geometry is not None and hasattr(geometry, "geom_type") and not geometry.is_empty:
        if geometry.geom_type != "Point":
            raise ValueError("Activity geometry must be a Point")
        return finite_coordinates(geometry.x, geometry.y)
    return finite_coordinates(activity.get("destination_x"), activity.get("destination_y"))


def finite_coordinates(x, y):
    try:
        coordinates = float(x), float(y)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing activity coordinates: {(x, y)!r}") from error
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(f"Invalid activity coordinates: {coordinates!r}")
    return coordinates


def lonlat_transformer(crs):
    """Geographic demand coordinates let SUMO apply its own projection and offset."""
    return Transformer.from_crs(crs, "EPSG:4326", always_xy=True)


def to_lonlat(transformer, coordinates):
    longitude, latitude = transformer.transform(*coordinates, errcheck=True)
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError(f"Invalid geographic activity coordinates: {(longitude, latitude)!r}")
    return f"{longitude:.10f},{latitude:.10f}"


@contextmanager
def xml_stream(path, compresslevel=1):
    """Open a deterministic binary XML stream for plain or gzip output."""
    path = Path(path)
    with path.open("wb") as raw:
        if path.suffix == ".gz":
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=compresslevel, mtime=0) as stream:
                yield stream
        else:
            yield raw
