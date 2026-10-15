"""Read driving costs produced during routing and add parking costs."""
import polars as pl
from dmc.cost.car import car_cost_simple, car_cost_weiss
from mode_choice.costs.artifacts import read_costs
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.costs.parking")
    context.stage("mode_choice.defaults")


def execute(context):
    # Read driving costs cached by either router, then attach parking variables.
    return join_variables(read_costs(context, "car"), context.stage("mode_choice.costs.parking"))


def calculate_driving_costs(routes, parameters, cost_model):
    """Price Pandana distances with the same functions used for DMC estimation."""
    # DMC's cost functions operate on arrays; the stage input/output stays Polars.
    distances = routes["car_distance_km"].to_numpy()
    if cost_model.lower() == "simple":
        costs = car_cost_simple(distances, parameters["carCost_CHF_km"])
    elif cost_model.lower() == "weiss":
        costs = car_cost_weiss(distances)
    else:
        raise ValueError(f"Unknown car_cost_model: {cost_model}")

    # Reject missing or invalid prices before publishing the cost artifact.
    result = routes.with_columns(pl.Series("car_cost_CHF", costs, dtype=pl.Float64))
    if result.filter(~pl.col("car_cost_CHF").is_finite().fill_null(False) | (pl.col("car_cost_CHF") < 0)).height:
        raise ValueError("Invalid car costs")
    return result
