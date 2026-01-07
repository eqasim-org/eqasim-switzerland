import numpy as np

def car_cost_weiss(distance_km):
    # for the record:
    # this function can be used as car cost, fit from this paper:
    # Weis, C., Kowald, M., Danalet, A., Schmid, B., Vrtic, M., Axhausen, K.W. and Mathys, N., 2021. Surveying and analysing mode and route choices in Switzerland 2010–2015. Travel behaviour and society, 22, pp.10-21.
    cost_per_km = np.minimum(0.3, 0.104 + 0.6 * np.exp(-1.2*(distance_km**0.33)))
    return distance_km * cost_per_km

def car_cost_simple(distance_km, cost_per_km):
    return distance_km * cost_per_km


def car_cost(context, distance_km):
    car_cost_model = context.config("car_cost_model")
    if car_cost_model.lower() == "simple":
        car_cost_per_km = context.config("car_cost_per_km")
        return car_cost_simple(distance_km, car_cost_per_km)
    elif car_cost_model.lower() == "weiss":
        return car_cost_weiss(distance_km)
    else:
        raise ValueError(f"Unknown car cost model: {car_cost_model}")

def get_cost(df, context):    
    return car_cost(context, df["car_distance_km"])
