"""Convert final prepared scenario tables at the SUMO/Polars boundary."""

import polars as pl

from sumo.functions.scenario.population.tools import (
    combine_populations,
    require_columns,
)
from sumo.functions.tables import as_polars, id_expression

COORDINATES = ["origin_x", "origin_y", "destination_x", "destination_y"]
COMMERCIAL_SCHEMA = {
    "id": pl.String,
    "source_id": pl.String,
    "person_id": pl.String,
    "trip_index": pl.Int64,
    "departure_time": pl.Float64,
    **{name: pl.Float64 for name in COORDINATES},
    "mode": pl.String,
    "sumo_type": pl.String,
    "population_source": pl.String,
}


def people(bundle, exclude_external):
    """Reuse final plans, including the airport replacements supplied separately."""
    persons, activities = as_polars(bundle.persons), as_polars(bundle.activities)
    require_columns(persons, ["person_id"], "Prepared persons")
    require_columns(activities, ["person_id"], "Prepared activities")
    persons = persons.with_columns(
        id_expression("person_id", persons.schema["person_id"])
    )
    activities = activities.with_columns(
        id_expression("person_id", activities.schema["person_id"])
    )
    kind = (
        pl.col("person_type") if "person_type" in persons.columns else pl.lit("normal")
    )
    persons = persons.with_columns(
        pl.when(kind.eq("normal"))
        .then(pl.lit("resident"))
        .when(kind.eq("crossborder"))
        .then(pl.lit("crossborder"))
        .otherwise(pl.lit("external"))
        .alias("_source")
    )
    if exclude_external:
        persons = persons.filter(pl.col("_source") != "external")
    activities = activities.join(
        persons.select("person_id"), on="person_id", how="semi", maintain_order="left"
    )

    groups = []
    for source in ("resident", "external", "crossborder"):
        selected = persons.filter(pl.col("_source") == source).drop("_source")
        if selected.height:
            chains = activities.join(
                selected.select("person_id"), on="person_id", how="semi"
            )
            groups.append((selected, chains, source))
    airport = airport_people(bundle)
    if airport is not None:
        groups.append((*airport, "airport"))
    if not groups:
        # Empty typed inputs still pass through shared schema validation.
        groups = [(persons.drop("_source"), activities, "resident")]
    return combine_populations(groups)


def vehicles(bundle, selected):
    """Keep prepared freight/LCV IDs and apply the shared mode selection."""
    frames = []
    sources = [
        ("freight_trips", "freight_", "truck", "sumo_truck"),
        ("lcv_trips", "lcv_", "lcv", "sumo_delivery"),
    ]
    for name, prefix, mode, type_id in sources:
        table = getattr(bundle, name, None)
        if table is None:
            continue
        frame = as_polars(table)
        require_columns(frame, ["agent_id", "departure_time", *COORDINATES], name)
        frame = frame.with_columns(
            id_expression("agent_id", frame.schema["agent_id"]),
            pl.lit(mode).alias("mode"),
        )
        if selected is not None:
            frame = frame.filter(pl.col("mode").is_in(list(selected)))
        frames.append(
            frame.select(
                pl.concat_str(pl.lit(prefix), pl.col("agent_id")).alias("id"),
                pl.col("agent_id").alias("source_id"),
                pl.col("agent_id").alias("person_id"),
                pl.lit(0, dtype=pl.Int64).alias("trip_index"),
                *[
                    pl.col(column).cast(pl.Float64, strict=False)
                    for column in ["departure_time", *COORDINATES]
                ],
                pl.col("mode"),
                pl.lit(type_id).alias("sumo_type"),
                pl.lit(name.removesuffix("_trips")).alias("population_source"),
            )
        )
    return (
        pl.concat(frames, how="vertical")
        if frames
        else pl.DataFrame(schema=COMMERCIAL_SCHEMA)
    )


def location_pools(bundle):
    """Index prepared facilities without repeating upstream location assignment."""
    from sumo.functions.scenario.scaling import _points

    frame = as_polars(bundle.facilities[["x", "y", "activity_options"]])
    purposes = ["home", "work", "education", "shop", "leisure", "other"]
    return {
        purpose: _points(
            frame.filter(pl.col("activity_options").list.contains(purpose)), "x", "y"
        )
        for purpose in purposes
    }


def airport_people(bundle):
    """Represent each prepared access trip as a two-activity plan, exactly once."""
    table = getattr(bundle, "airport_trips", None)
    if table is None or not len(table):
        return None
    frame = as_polars(table)
    require_columns(
        frame,
        [
            "agent_id",
            "trip_id",
            "origin_purpose",
            "destination_purpose",
            "departure_time",
            "arrival_time",
            "mode",
            *COORDINATES,
        ],
        "Prepared airport trips",
    )
    frame = frame.with_columns(id_expression("agent_id", frame.schema["agent_id"]))
    persons = frame.select(
        pl.col("agent_id").alias("person_id"),
        *[
            pl.when(pl.col("origin_purpose") == "home")
            .then(pl.col(f"origin_{axis}"))
            .when(pl.col("destination_purpose") == "home")
            .then(pl.col(f"destination_{axis}"))
            .otherwise(None)
            .alias(f"home_{axis}")
            for axis in ("x", "y")
        ],
    )
    origin = frame.select(
        pl.col("agent_id").alias("person_id"),
        pl.lit(0, dtype=pl.Int64).alias("activity_index"),
        pl.lit(0.0).alias("start_time"),
        pl.col("departure_time").cast(pl.Float64).alias("end_time"),
        pl.col("mode").alias("following_mode"),
        pl.col("origin_purpose").alias("purpose"),
        pl.col("origin_x").alias("destination_x"),
        pl.col("origin_y").alias("destination_y"),
        pl.col("trip_id"),
    )
    destination = frame.select(
        pl.col("agent_id").alias("person_id"),
        pl.lit(1, dtype=pl.Int64).alias("activity_index"),
        pl.col("arrival_time").cast(pl.Float64).alias("start_time"),
        pl.lit(None, dtype=pl.Float64).alias("end_time"),
        pl.lit(None, dtype=pl.String).alias("following_mode"),
        pl.col("destination_purpose").alias("purpose"),
        pl.col("destination_x"),
        pl.col("destination_y"),
        pl.col("trip_id"),
    )
    return persons, pl.concat([origin, destination], how="vertical_relaxed")
