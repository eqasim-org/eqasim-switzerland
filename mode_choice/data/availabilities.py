"""Mode availabilities from complete tours, route validity and person constraints."""
import polars as pl
from mode_choice import defaults as D
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.routing_variables.pt")
    context.stage("mode_choice.costs.pt")


def execute(context):
    # Read only the person and tour attributes needed for mode availability.
    trips = context.stage("mode_choice.data.prepare").select(
        "trip_id", "person_id", "tour_index", "euclidean_distance_km",
        *D.PERSON_AVAILABILITY_COLUMNS,
    )

    # Attach routed variables and PT fares without depending on the inputs stage.
    for stage in ("mode_choice.routing_variables.car", "mode_choice.routing_variables.pt", "mode_choice.costs.pt"):
        trips = join_variables(trips, context.stage(stage))

    # Keep a trip-keyed output so inputs can merge this stage like routing and costs.
    return calculate_availabilities(trips)


def calculate_availabilities(trips):
    # Check physical validity per trip. Short and zero-length connecting legs are valid.
    car_routed = (
        finite_nonnegative("car_travel_time_min", "car_distance_km")
        & (pl.col("car_travel_time_min") <= D.MAX_ROUTED_TIME_MIN)
    )
    pt_routed = (
        finite_nonnegative("pt_in_vehicle_time_min", "pt_access_egress_time_min",
                           "pt_transfer_time_min", "pt_transfers", "pt_cost_CHF")
        & (pl.col("pt_in_vehicle_time_min") <= D.MAX_ROUTED_TIME_MIN)
        & (pl.col("pt_transfers") <= D.MAX_PT_TRANSFERS)
        & (pl.col("pt_transfer_time_min") <= D.MAX_PT_TRANSFER_TIME_MIN)
    )

    # Compute minimum distance and time once per original tour, before any batching
    # or computational splitting. A short middle leg must not break a car chain.
    group = ["person_id", "tour_index"]
    tours = trips.group_by(group).agg(
        (pl.col("euclidean_distance_km").sum() >= D.MIN_MOTORIZED_TOUR_DISTANCE_KM).alias("motorized_tour"),
        (pl.col("car_travel_time_min").filter(car_routed).sum() >= D.MIN_MOTORIZED_TOUR_TIME_MIN).alias("car_tour"),
        (pl.col("pt_in_vehicle_time_min").filter(pt_routed).sum() >= D.MIN_MOTORIZED_TOUR_TIME_MIN).alias("pt_tour"),
    )
    trips = trips.join(tours, on=group, how="left", validate="m:1", maintain_order="left")

    # Apply the tour minimums to every leg, keeping ownership and route validity.
    # The sampler separately enforces car/bike locations across all tour segments.
    car_available = pl.col("motorized_tour") & pl.col("car_tour") & car_routed
    pt_available = pl.col("motorized_tour") & pl.col("pt_tour") & pt_routed
    return trips.select(
        "trip_id",
        (pl.col("car_availability") & car_available).fill_null(False).alias("car_availability"),
        pt_available.fill_null(False).alias("pt_availability"),
        (pl.col("bike_availability") & (pl.col("euclidean_distance_km") < D.MAX_BIKE_DISTANCE_KM))
        .fill_null(False).alias("bike_availability"),
        pl.lit(True).alias("walk_availability"),
        car_available.fill_null(False).alias("car_passenger_availability"),
    )


def finite_nonnegative(*columns):
    return pl.all_horizontal([
        pl.col(column).is_finite() & (pl.col(column) >= 0) for column in columns
    ]).fill_null(False)
