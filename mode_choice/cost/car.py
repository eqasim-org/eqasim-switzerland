"""
This is a very simple model to estimate car corst based on crowfly distance.
In future, this could be extended to use the routed distance instead.
We do not use it currently because of pandana current limitations.
"""

import numpy as np
from mode_choice.dmc_defaults import Defaults

def car_cost_weiss(distance_km, *args, **kwargs):
    # for the record:
    # this function can be used as car cost, fit from this paper:
    # Weis, C., Kowald, M., Danalet, A., Schmid, B., Vrtic, M., Axhausen, K.W. and Mathys, N., 2021. Surveying and analysing mode and route choices in Switzerland 2010–2015. Travel behaviour and society, 22, pp.10-21.
    cost_per_km = np.minimum(0.3, 0.104 + 0.6 * np.exp(-1.2*(distance_km**0.33)))
    return distance_km * cost_per_km

def car_cost_simple(distance_km, cost_per_km):
    return distance_km * cost_per_km


def car_cost(context, *args, **kwargs):
    car_cost_model = context.config("car_cost_model")
    if car_cost_model == "simple":
        return car_cost_simple(*args, **kwargs)
    elif car_cost_model == "weiss":
        return car_cost_weiss(*args, **kwargs)
    else:
        raise ValueError(f"Unknown car cost model: {car_cost_model}")

def configure(context):
    context.stage("mode_choice.variables.car")
    context.config("car_cost_per_km", default = Defaults.CAR_COST_PER_KM)
    context.config("car_cost_model", default = Defaults.CAR_COST_MODEL)

def execute(context):
    # read prepared trips
    df = context.stage("mode_choice.variables.car")[
        ["person_id", "trip_id","distance_km"]]
    
    # compute the cost
    car_cost_per_km = context.config("car_cost_per_km") 
    df["cost_CHF"] = car_cost(context, df["distance_km"], car_cost_per_km)

    return df[["person_id", "trip_id", "cost_CHF"]]