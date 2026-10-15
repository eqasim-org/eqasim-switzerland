"""Car passengers reuse the car route; never make another routing request."""


def configure(context):
    context.stage("mode_choice.routing_variables.car")


def execute(context):
    return context.stage("mode_choice.routing_variables.car").rename({
        "car_travel_time_min": "car_passenger_travel_time_min", "car_distance_km": "car_passenger_distance_km"})
