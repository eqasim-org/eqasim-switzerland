"""Freight and LCV requests with the identifiers used by the population writer."""
import polars as pl

REQUEST_SCHEMA = {
    "identifier": pl.String, "person_id": pl.String, "trip_index": pl.Int64,
    "origin_x": pl.Float64, "origin_y": pl.Float64,
    "destination_x": pl.Float64, "destination_y": pl.Float64,
    "departure_time": pl.Float64, "mode": pl.String,
}


def configure(context):
    context.config("use_freight", True)
    context.config("use_lcv", True)
    for kind in ("freight", "lcv"):
        if context.config(f"use_{kind}"):
            context.stage(f"synthesis.{kind}.trips")


def execute(context):

    # Read enabled vehicle populations and preserve their MATSim person IDs.

    frames = [requests(context.stage(f"synthesis.{kind}.trips"), kind)
              for kind in ("freight", "lcv") if context.config(f"use_{kind}")]

    # Return a typed empty frame when both vehicle populations are disabled.

    return pl.concat(frames) if frames else pl.DataFrame(schema=REQUEST_SCHEMA)


def requests(source, kind):
    frame = source if isinstance(source, pl.DataFrame) else pl.from_pandas(source)
    key = "agent_id" if kind == "freight" else "trip_id"
    person = pl.concat_str(pl.lit(kind + "_"), pl.col(key).cast(pl.String))
    result = frame.select(
        person.alias("identifier"), person.alias("person_id"), pl.lit(0).alias("trip_index"),
        "origin_x", "origin_y", "destination_x", "destination_y", "departure_time",
        (pl.col("vehicle_type") if kind == "freight" else pl.lit("truck")).alias("mode"),
    ).cast(REQUEST_SCHEMA)
    if result["identifier"].null_count() or result["identifier"].n_unique() != result.height:
        raise ValueError(f"Null or duplicate {kind} routing identifiers")
    if result.filter(~pl.col("mode").eq("truck").fill_null(False)).height:
        raise ValueError(f"Unsupported {kind} vehicle mode; expected truck, as in MATSim")
    numeric = [name for name, dtype in REQUEST_SCHEMA.items() if dtype == pl.Float64]
    if result.filter(~pl.all_horizontal(pl.col(numeric).is_finite()).fill_null(False)
                     | (pl.col("departure_time") < 0)).height:
        raise ValueError(f"Nonfinite coordinates or invalid departure times in {kind} trips")
    return result
