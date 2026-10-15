"""Compatibility helper; the pipeline uses the explicit costs/routing stages."""
import polars as pl
from mode_choice.costs.car import cost_expression
from mode_choice.costs.parking import calculate
from mode_choice.model.availability import prepare as availability
from mode_choice.data.frames import join_variables


def prepare(trips, p, *, car_cost_model, walk_speed, walk_factor, bike_speed, bike_factor, assume_sorted=False):
    if not assume_sorted:
        trips = trips.sort("person_id", "trip_index")
    generated_key = "trip_id" not in trips.columns
    if generated_key:
        trips = trips.with_columns(pl.concat_str("person_id", "trip_index", separator="_").alias("trip_id"))
    result = join_variables(trips, calculate(trips, p)).with_columns(
        cost_expression(p, car_cost_model).alias("car_cost_CHF"),
        pl.col("car_travel_time_min").alias("car_passenger_travel_time_min"),
        pl.col("car_distance_km").alias("car_passenger_distance_km"))
    for mode, speed, factor in (("walk", walk_speed, walk_factor), ("bike", bike_speed, bike_factor)):
        if f"{mode}_travel_time_min" not in result.columns:
            result = result.with_columns((pl.col("euclidean_distance_km") * factor * 1000 / speed / 60)
                                         .alias(f"{mode}_travel_time_min"))
    result = availability(result)
    return result.drop("trip_id") if generated_key else result
