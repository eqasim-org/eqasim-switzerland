"""
This is a very simple model to estimate car corst based on crowfly distance.
In future, this could be extended to use the routed distance instead.
We do not use it currently because of pandana current limitations.
"""


def configure(context):
    context.stage("mode_choice.travel_times.car")
    context.config("car_cost_per_km", 0.26) #CHF per km


def execute(context):
    # read prepared trips
    df = context.stage("mode_choice.travel_times.car")[
        ["person_id", "trip_id","distance_km"]]
    
    # compute the cost
    car_cost_per_km = context.config("car_cost_per_km") 
    df["cost_CHF"] = df["distance_km"] * car_cost_per_km

    return df[["person_id", "trip_id", "cost_CHF"]]