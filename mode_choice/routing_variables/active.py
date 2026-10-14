"""Common walk/bike routing, with beeline estimates only for the Pandana backend."""
import math
import polars as pl
from mode_choice.defaults import RUN_DEFAULTS
from mode_choice.routing.java import configure_mode, car_backend, route_mode


def configure(context, mode):
    configure_mode(context, mode)
    for name in (f"{mode}_speed_m_per_s", f"{mode}_distance_factor"):
        context.config(name, RUN_DEFAULTS[name])


def execute(context, mode):
    if car_backend(context) == "java":
        return route_mode(context, mode)
    speed, factor = (context.config(f"{mode}_{name}") for name in ("speed_m_per_s", "distance_factor"))
    if not all(math.isfinite(value) and value > 0 for value in (speed, factor)):
        raise ValueError(f"{mode} speed and distance factor must be positive and finite")
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(f"{mode}_{c}")
                                                    for c in ("distance_km", "travel_time_min")])
    distance = pl.col("euclidean_distance_km") * factor
    return trips.select(
        "trip_id", distance.alias(f"{mode}_distance_km"),
        (distance * 1000 / speed / 60).alias(f"{mode}_travel_time_min"))
