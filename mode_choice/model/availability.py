"""Mode availability from household constraints and the routed variables."""
import polars as pl
from mode_choice import defaults as D


def prepare(trips):
    car_routed = (pl.col("car_travel_time_min").is_between(D.MIN_ROUTED_TIME_MIN, D.MAX_ROUTED_TIME_MIN)
                  & pl.col("car_distance_km").is_finite() & (pl.col("car_distance_km") >= 0)
                  & (pl.col("euclidean_distance_km") >= D.MIN_MOTORIZED_DISTANCE_KM)).fill_null(False)
    pt_routed = (pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in [
        "pt_in_vehicle_time_min", "pt_access_egress_time_min", "pt_transfer_time_min", "pt_transfers", "pt_cost_CHF"]])
        & pl.col("pt_in_vehicle_time_min").is_between(D.MIN_ROUTED_TIME_MIN, D.MAX_ROUTED_TIME_MIN)
        & (pl.col("pt_transfers") <= D.MAX_PT_TRANSFERS) & (pl.col("pt_transfer_time_min") <= D.MAX_PT_TRANSFER_TIME_MIN)
        & (pl.col("euclidean_distance_km") >= D.MIN_MOTORIZED_DISTANCE_KM)).fill_null(False)
    return trips.with_columns(
        (pl.col("car_availability") & car_routed).alias("car_availability"),
        car_routed.alias("car_passenger_availability"), pt_routed.alias("pt_availability"),
        (pl.col("bike_availability") & (pl.col("euclidean_distance_km") < D.MAX_BIKE_DISTANCE_KM)).alias("bike_availability"),
        pl.lit(True).alias("walk_availability"),
    )
