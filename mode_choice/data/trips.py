"""Read the population once and assign a stable mode-choice trip identifier."""
import polars as pl
from mode_choice.defaults import SOURCE_TRIP_ID
from mode_choice.routing.java.server import ensure_server_started

def configure(context):
    context.stage("mode_choice.routing.java.server")
    context.stage("synthesis.population.trips_with_departures")


def execute(context):
    # We make sure the server is starting
    ensure_server_started(context)

    # we get the trips for which we will perform mode choice, and we assign a stable trip_id to them
    source = context.stage("synthesis.population.trips_with_departures")[[
       'person_id', 'trip_id', 'trip_index', 'departure_time','arrival_time', 'preceding_purpose',
       'following_purpose','trip_duration', 'mode'
       ]]
    trips = source.clone() if isinstance(source, pl.DataFrame) else pl.from_pandas(source)

    # this id is only used for mode choice, and is not the same as the trip_id used in the population
    if SOURCE_TRIP_ID in trips.columns:
        raise ValueError(f"Reserved mode-choice column: {SOURCE_TRIP_ID}")

    if "trip_id" in trips.columns:
        trips = trips.rename({"trip_id": SOURCE_TRIP_ID})

    # we create the new trip_id used in mode choice module
    if trips.select(pl.any_horizontal(pl.col("person_id", "trip_index").is_null()).any()).item():
        raise ValueError("Null person/trip keys")

    trips = trips.with_columns(pl.concat_str("person_id", "trip_index", separator="_").alias("trip_id"))

    if trips["trip_id"].n_unique() != trips.height:
        raise ValueError("Duplicate person/trip keys")

    return trips
