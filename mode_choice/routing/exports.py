"""Collect selected Java itineraries for RunPopulationRouting --routes-path."""
import json
from pathlib import Path

import polars as pl

from mode_choice.defaults import ACTIVE_ROUTING_DEFAULTS

EXPORT_SCHEMA = {
    "person_id": pl.String, "trip_index": pl.Int64, "mode": pl.String,
    "origin_x": pl.Float64, "origin_y": pl.Float64,
    "destination_x": pl.Float64, "destination_y": pl.Float64,
    "departure_time": pl.Float64, "travel_time": pl.Float64,
    "travel_distance": pl.Float64, "route": pl.String,
}


def configure(context):
    context.stage("mode_choice.run")
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.routing_variables.pt")
    for mode in ("walk", "bike"):
        name = f"mode_choice_{mode}_routing_backend"
        context.config(name, ACTIVE_ROUTING_DEFAULTS[name])
        if context.config(name) == "java":
            context.stage(f"mode_choice.routing_variables.{mode}")


def population_routes(context):
    """Write only chosen modes and freight, keeping upstream CSVs untouched.

    Omit trip_id: mode choice uses a population-wide ID whereas Java's default
    trip_id is the local trip index. person_id + trip_index is the matching key.
    """
    choices = context.stage("mode_choice.run")
    choices = choices if isinstance(choices, pl.DataFrame) else pl.from_pandas(choices)
    choices = choices.select(pl.col("person_id").cast(pl.String), pl.col("trip_index").cast(pl.Int64),
                             pl.col("mode").alias("chosen_mode")).lazy()
    sources = []
    for mode in ("car", "pt", "walk", "bike"):
        if mode in ("walk", "bike") and context.config(f"mode_choice_{mode}_routing_backend") != "java":
            continue
        stage = f"mode_choice.routing_variables.{mode}"
        path = Path(context.path(stage)) / f"{mode}_routes.csv"
        if not path.exists() and context.stage(stage).is_empty():
            continue
        routes = pl.scan_csv(path, schema_overrides=EXPORT_SCHEMA).select(*EXPORT_SCHEMA)
        routes = routes.filter(
            pl.col("route").is_not_null() & ~pl.col("route").is_in(["", "NaN", "nan", "null"])
            & pl.col("travel_time").is_finite() & (pl.col("travel_time") >= 0)
            & pl.col("travel_distance").is_finite() & (pl.col("travel_distance") >= 0)
        ).join(choices, on=["person_id", "trip_index"], how="left", validate="m:1", maintain_order="left")
        if mode == "car":
            # Passengers reuse the same path without adding another vehicle to
            # the assignment. Retag only selected passenger itineraries.
            routes = routes.filter((pl.col("mode") == "truck") | (pl.col("mode") == pl.col("chosen_mode"))
                                   | ((pl.col("mode") == "car") & (pl.col("chosen_mode") == "car_passenger")))
            passenger = pl.col("chosen_mode") == "car_passenger"
            routes = routes.with_columns(
                pl.when(passenger).then(pl.lit("car_passenger")).otherwise(pl.col("mode")).alias("mode"),
                pl.coalesce(pl.when(passenger).then(pl.col("route")).otherwise(None)
                            .map_elements(passenger_route, return_dtype=pl.String), pl.col("route")).alias("route"),
            )
        else:
            routes = routes.filter(pl.col("mode") == pl.col("chosen_mode"))
        sources.append(routes.select(*EXPORT_SCHEMA))
    target = Path(context.path()) / "mode_choice_routes.csv"
    (pl.concat(sources) if sources else pl.DataFrame(schema=EXPORT_SCHEMA).lazy()).sink_csv(target)
    return str(target.resolve())


def passenger_route(value):
    """Equivalent to the server's car_passenger wrapper, without routing again."""
    try:
        itinerary = json.loads(value)
    except json.JSONDecodeError:
        return value
    if (not isinstance(itinerary, dict) or itinerary.get("mode") != "car"
            or not isinstance(itinerary.get("parts"), list) or len(itinerary["parts"]) != 1
            or not isinstance(itinerary["parts"][0], dict)):
        return value
    itinerary["mode"] = "car_passenger"
    for part in itinerary["parts"]:
        part["mode"] = "car_passenger"
        if part.get("routingMode") == "car":
            part["routingMode"] = "car_passenger"
    return json.dumps(itinerary, separators=(",", ":"))
