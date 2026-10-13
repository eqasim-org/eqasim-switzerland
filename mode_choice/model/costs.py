"""Costs, parking penalties and availability consistent with DMC training."""
import polars as pl
from mode_choice import defaults as D


def prepare(trips, p, *, car_cost_model, walk_speed, walk_factor, bike_speed, bike_factor, assume_sorted=False):
    distance = pl.col("car_distance_km")
    if car_cost_model == "weiss":
        car_cost = distance * pl.min_horizontal(pl.lit(D.WEISS_MAX_RATE), D.WEISS_BASE_RATE + D.WEISS_RATE_AMPLITUDE * (D.WEISS_DISTANCE_RATE * distance.pow(D.WEISS_DISTANCE_EXPONENT)).exp())
    elif car_cost_model == "simple":
        car_cost = distance * p["carCost_CHF_km"]
    else:
        raise ValueError(f"Unknown car_cost_model: {car_cost_model}")
    if not assume_sorted:
        trips = trips.sort(["person_id", "trip_index"])
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
    car_routed = (pl.col("car_travel_time_min").is_between(D.MIN_ROUTED_TIME_MIN, D.MAX_ROUTED_TIME_MIN)
                  & pl.col("car_distance_km").is_finite() & (pl.col("car_distance_km") >= 0)
                  & (pl.col("euclidean_distance_km") >= D.MIN_MOTORIZED_DISTANCE_KM)).fill_null(False)
    pt_routed = (pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) >= 0) for c in [
        "pt_in_vehicle_time_min", "pt_access_egress_time_min", "pt_transfer_time_min", "pt_transfers", "pt_cost_CHF"]])
        & pl.col("pt_in_vehicle_time_min").is_between(D.MIN_ROUTED_TIME_MIN, D.MAX_ROUTED_TIME_MIN)
        & (pl.col("pt_transfers") <= D.MAX_PT_TRANSFERS) & (pl.col("pt_transfer_time_min") <= D.MAX_PT_TRANSFER_TIME_MIN)
        & (pl.col("euclidean_distance_km") >= D.MIN_MOTORIZED_DISTANCE_KM)).fill_null(False)
    return trips.with_columns(
        car_cost.alias("car_cost_CHF"), parking_cost.alias("parking_cost_CHF"), search.alias("parking_searching_duration_min"),
        (pl.col("car_availability") & car_routed).alias("car_availability"), car_routed.alias("car_passenger_availability"),
        pt_routed.alias("pt_availability"),
        (pl.col("bike_availability") & (pl.col("euclidean_distance_km") < D.MAX_BIKE_DISTANCE_KM)).alias("bike_availability"),
        # Walking remains a last-resort alternative for tours with no motorised route.
        pl.lit(True).alias("walk_availability"),
        pl.col("car_travel_time_min").alias("car_passenger_travel_time_min"),
        pl.col("car_distance_km").alias("car_passenger_distance_km"),
        (pl.col("euclidean_distance_km") * walk_factor * 1000 / walk_speed / 60).alias("walk_travel_time_min"),
        (pl.col("euclidean_distance_km") * bike_factor * 1000 / bike_speed / 60).alias("bike_travel_time_min"),
    )
