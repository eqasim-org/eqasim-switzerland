"""Independent Java or vectorized detour/speed routing for walk and bike."""
import math
from numbers import Real
import polars as pl
from mode_choice.defaults import RUN_DEFAULTS, ACTIVE_ROUTING_DEFAULTS, ACTIVE_ROUTING_BACKENDS
from mode_choice.routing.java import route_mode


def configure(context, mode):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    name = f"mode_choice_{mode}_routing_backend"
    context.config(name, ACTIVE_ROUTING_DEFAULTS[name])
    if routing_backend(context, mode) == "java":
        context.stage("mode_choice.routing.server")
        context.stage("mode_choice.routing.requests")
    else:
        for name in (f"{mode}_speed_m_per_s", f"{mode}_distance_factor"):
            context.config(name, RUN_DEFAULTS[name])


def execute(context, mode):
    if routing_backend(context, mode) == "java":
        return route_mode(context, mode)
    speed, factor = (context.config(f"{mode}_{name}") for name in ("speed_m_per_s", "distance_factor"))
    if not all(isinstance(value, Real) and not isinstance(value, bool)
               and math.isfinite(value) and value > 0 for value in (speed, factor)):
        raise ValueError(f"{mode} speed and distance factor must be positive and finite")
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips.select("trip_id").with_columns([pl.lit(None, dtype=pl.Float64).alias(f"{mode}_{c}")
                                                    for c in ("distance_km", "travel_time_min")])
    distance = pl.col("euclidean_distance_km") * factor
    return trips.select(
        "trip_id", distance.alias(f"{mode}_distance_km"),
        (distance * 1000 / speed / 60).alias(f"{mode}_travel_time_min"))


def routing_backend(context, mode):
    name = f"mode_choice_{mode}_routing_backend"
    backend = context.config(name)
    if backend not in ACTIVE_ROUTING_BACKENDS:
        raise ValueError(f"{name} must be one of {ACTIVE_ROUTING_BACKENDS}")
    return backend
