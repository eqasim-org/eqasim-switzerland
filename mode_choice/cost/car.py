"""
This is a very simple model to estimate car corst based on crowfly distance.
In future, this could be extended to use the routed distance instead.
We do not use it currently because of pandana current limitations.
"""


def configure(context):
    context.stage("mode_choice.trips.prepare_trips")
    context.config("car_cost_per_km", 0.26) #CHF per km
    context.config("car_distance_factor", 1.4) #CHF per hour


def execute(context):
    # read prepared trips
    trips = context.stage("mode_choice.trips.prepare_trips")[
        ["person_id", "trip_index","trip_id","crowfly_distance"]]
    
    # compute the cost
    car_cost_per_km = context.config("car_cost_per_km") 
    distance_km = trips["crowfly_distance"] / 1000 * context.config("car_distance_factor")
    trips["car_cost"] = distance_km * car_cost_per_km
    
    return trips[["person_id", "trip_index","trip_id","car_cost"]]