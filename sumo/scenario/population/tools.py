"""Shared population preparation, identifiers, coordinates and XML streams."""

from contextlib import contextmanager
import gzip
import math
from numbers import Integral, Real
from pathlib import Path

import pandas as pd
from pyproj import Transformer
from shapely import wkt

from synthesis.population.trips_with_departures import align_activities

ACTIVITY_KEY = ["person_id", "activity_index"]


def as_frame(value):
    """Copy upstream tables, including Polars results, before any modification."""
    if hasattr(value, "to_pandas"):
        value = value.to_pandas()
    return pd.DataFrame(value).copy()


def require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def scalar_text(value):
    """Serialize scalar metadata without losing IDs or writing nan/None."""
    if value is None or not pd.api.types.is_scalar(value) or pd.isna(value):
        return None
    if pd.api.types.is_bool(value):
        return "true" if value else "false"
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return format(value, ".15g") if math.isfinite(value) else None
    return str(value)


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


def prepare_activities(activities, locations, trips):
    """Join locations and final trip choices by keys, never by row position."""
    activities, locations, trips = map(as_frame, (activities, locations, trips))
    require_columns(activities, ACTIVITY_KEY + ["start_time", "end_time"], "Activities")
    require_columns(locations, ACTIVITY_KEY + ["geometry"], "Activity locations")
    require_columns(trips, ["person_id", "trip_index", "mode", "departure_time", "arrival_time"], "Trips")
    activities = align_activities(activities, trips)
    location_columns = [column for column in locations if column not in activities or column in ACTIVITY_KEY]
    activities = activities.merge(locations[location_columns], on=ACTIVITY_KEY, how="left", validate="1:1")
    trip_columns = ["person_id", "trip_index", "mode", "departure_time", "arrival_time"]
    trip_columns += [column for column in ("trip_id", "passenger_vehicle_id") if column in trips]
    following = trips[trip_columns].rename(columns={"trip_index": "activity_index", "mode": "following_mode"})
    following["trip_index"] = following["activity_index"]
    activities = activities.drop(columns=[column for column in following if column in activities and column not in ACTIVITY_KEY])
    return activities.merge(following, on=ACTIVITY_KEY, how="left", validate="1:1")


def combine_populations(populations):
    """Keep all people, including non-travellers, and validate their activity keys."""
    person_frames, activity_frames = [], []
    for persons, activities, source in populations:
        persons, activities = as_frame(persons), as_frame(activities)
        require_columns(persons, ["person_id"], f"{source} persons")
        require_columns(activities, ACTIVITY_KEY + ["purpose"], f"{source} activities")
        persons["person_id"] = persons["person_id"].map(identifier)
        activities["person_id"] = activities["person_id"].map(identifier)
        indices = pd.to_numeric(activities["activity_index"], errors="raise")
        if indices.isna().any() or (indices < 0).any() or (indices % 1 != 0).any():
            raise ValueError(f"Invalid activity_index in {source} population")
        activities["activity_index"] = indices.astype("int64")
        persons["population_source"] = source
        person_frames.append(persons)
        activity_frames.append(activities)
    persons = pd.concat(person_frames, ignore_index=True)
    activities = pd.concat(activity_frames, ignore_index=True)
    if persons["person_id"].duplicated().any():
        raise ValueError("Duplicate person_id across SUMO populations")
    if activities.duplicated(ACTIVITY_KEY).any():
        raise ValueError("Duplicate person_id/activity_index in SUMO activities")
    if set(activities["person_id"]) - set(persons["person_id"]):
        raise ValueError("SUMO activities reference persons absent from the population")
    if set(persons["person_id"]) - set(activities["person_id"]):
        raise ValueError("SUMO persons are missing their activity chain")
    activities = activities.sort_values(ACTIVITY_KEY, kind="stable").reset_index(drop=True)
    first = activities.drop_duplicates("person_id").set_index("person_id")
    starts = first["start_time"] if "start_time" in first else pd.Series(index=first.index, dtype=float)
    persons["sumo_depart"] = persons["person_id"].map(starts).fillna(0).map(seconds)
    persons = persons.sort_values(["sumo_depart", "person_id"], kind="stable").reset_index(drop=True)
    activities["_person_order"] = pd.Categorical(activities["person_id"], categories=persons["person_id"], ordered=True)
    activities = activities.sort_values(["_person_order", "activity_index"], kind="stable").drop(columns="_person_order")
    return {"persons": persons, "activities": activities.reset_index(drop=True)}


def activity_coordinates(activity, person):
    """Read source coordinates, using person home coordinates for home activities."""
    if activity.get("purpose") == "home":
        x, y = person.get("home_x"), person.get("home_y")
        if scalar_text(x) is not None and scalar_text(y) is not None:
            return finite_coordinates(x, y)
    geometry = activity.get("geometry")
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
