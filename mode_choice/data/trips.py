"""Read the population once and assign a stable mode-choice trip identifier."""
import polars as pl
from mode_choice.defaults import SOURCE_TRIP_ID


def configure(context):
    context.stage("mode_choice.routing.server")
    context.stage("synthesis.population.trips_with_departures")


def execute(context):
    from mode_choice.routing.server import ensure_started
    ensure_started(context)
    source = context.stage("synthesis.population.trips_with_departures")
    trips = source.clone() if isinstance(source, pl.DataFrame) else pl.from_pandas(source)
    if SOURCE_TRIP_ID in trips.columns:
        raise ValueError(f"Reserved mode-choice column: {SOURCE_TRIP_ID}")
    if "trip_id" in trips.columns:
        trips = trips.rename({"trip_id": SOURCE_TRIP_ID})
    if trips.select(pl.any_horizontal(pl.col("person_id", "trip_index").is_null()).any()).item():
        raise ValueError("Null person/trip keys")
    trips = trips.with_columns(pl.concat_str("person_id", "trip_index", separator="_").alias("trip_id"))
    if trips["trip_id"].n_unique() != trips.height:
        raise ValueError("Duplicate person/trip keys")
    return trips
