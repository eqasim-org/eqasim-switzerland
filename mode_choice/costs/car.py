"""Car operating costs, together with the separately computed parking costs."""
import polars as pl
from mode_choice import defaults as D
from mode_choice.model.parameters import load
from mode_choice.data.frames import join_variables


def configure(context):
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.costs.parking")
    context.stage("mode_choice.defaults")
    context.stage("dmc.params")
    context.config("car_cost_model", D.RUN_DEFAULTS["car_cost_model"])


def execute(context):
    trips = context.stage("mode_choice.routing_variables.car")
    costs = trips.select("trip_id", cost_expression(load(context), context.config("car_cost_model").lower()).alias("car_cost_CHF"))
    return join_variables(costs, context.stage("mode_choice.costs.parking"))


def cost_expression(p, car_cost_model):
    distance = pl.col("car_distance_km")
    if car_cost_model == "weiss":
        car_cost = distance * pl.min_horizontal(pl.lit(D.WEISS_MAX_RATE), D.WEISS_BASE_RATE + D.WEISS_RATE_AMPLITUDE * (D.WEISS_DISTANCE_RATE * distance.pow(D.WEISS_DISTANCE_EXPONENT)).exp())
    elif car_cost_model == "simple":
        car_cost = distance * p["carCost_CHF_km"]
    else:
        raise ValueError(f"Unknown car_cost_model: {car_cost_model}")
    return car_cost
