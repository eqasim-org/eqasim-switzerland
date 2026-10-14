"""Parking duration, monetary cost and search time on complete person days."""
import polars as pl
from mode_choice import defaults as D
from mode_choice.model.parameters import load
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.defaults")
    context.stage("dmc.params")


def execute(context):
    trips = context.stage("mode_choice.data.prepare")
    if trips.is_empty():
        return trips.select("trip_id").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("parking_cost_CHF"),
            pl.lit(None, dtype=pl.Float64).alias("parking_searching_duration_min"))
    trips = join_variables(trips, context.stage("mode_choice.routing_variables.car"))
    return calculate(trips.sort("person_id", "trip_index"), load(context))


def calculate(trips, p):
    next_departure = pl.col("departure_time").shift(-1).over("person_id").clip(D.PARKING_START_S, D.PARKING_END_S)
    duration = ((next_departure - pl.col("departure_time").clip(D.PARKING_START_S, D.PARKING_END_S)) / 60
                - pl.col("car_travel_time_min")).fill_null(0).clip(0, D.PARKING_MAX_DURATION_MIN)
    parking_rate = pl.col("destination_municipality").cast(pl.String).replace_strict({
        "urban": p["urbanParkingCost_CHF_h"], "urbancore": p["urbancoreParkingCost_CHF_h"],
        "suburban": p["suburbanParkingCost_CHF_h"], "rural": 0.0,
    }, return_dtype=pl.Float64)
    parking_cost = pl.when(pl.col("destination_home")).then(0.0).otherwise(
        (duration / 60 - D.PARKING_FREE_HOURS).clip(lower_bound=0) * parking_rate
        * pl.when(pl.col("destination_work")).then(p["parkingPriceReductionForWork"]).otherwise(1.0)
    ).clip(0, D.PARKING_MAX_COST_CHF)
    search = pl.when(pl.col("destination_home") | pl.col("destination_work")).then(0.0).otherwise(
        pl.col("destination_municipality").cast(pl.String).replace_strict({
            "urban": p["parking.urbanParkingSearchDuration_min"],
            "urbancore": p["parking.urbancoreParkingSearchDuration_min"],
            "suburban": p["parking.suburbanParkingSearchDuration_min"], "rural": 0.0,
        }, return_dtype=pl.Float64)
    )
    return trips.select("trip_id", parking_cost.alias("parking_cost_CHF"),
                        search.alias("parking_searching_duration_min"))
