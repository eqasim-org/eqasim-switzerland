"""Vectorized walking/cycling distance and time from detour and speed factors."""
import math
from numbers import Real
import polars as pl


def route(context, mode):

    # Validate the speed and detour factor used by Python routing.

    speed, factor = (context.config(f"{mode}_{name}") for name in ("speed_m_per_s", "distance_factor"))
    if not all(isinstance(value, Real) and not isinstance(value, bool)
               and math.isfinite(value) and value > 0 for value in (speed, factor)):
        raise ValueError(f"{mode} speed and distance factor must be positive and finite")
    # Compute distance and time with vectorized expressions.

    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(f"{mode}_{c}")
                                                    for c in ("distance_km", "travel_time_min")])
    distance = pl.col("euclidean_distance_km") * factor
    return trips.select(
        "trip_id", distance.alias(f"{mode}_distance_km"),
        (distance * 1000 / speed / 60).alias(f"{mode}_travel_time_min"))
