
def configure(context):
    context.stage("mode_choice.travel_times.bike")
    context.stage("mode_choice.travel_times.car")
    context.stage("mode_choice.travel_times.pt")
    context.stage("mode_choice.travel_times.car_passenger")
    context.stage("mode_choice.travel_times.walk")

def execute(context):
    bike_tt = context.stage("mode_choice.travel_times.bike")
    car_tt = context.stage("mode_choice.travel_times.car")
    pt_tt = context.stage("mode_choice.travel_times.pt")
    car_passenger_tt = context.stage("mode_choice.travel_times.car_passenger")
    walk_tt = context.stage("mode_choice.travel_times.walk")

    return dict(
        bike=bike_tt,
        car=car_tt,
        pt=pt_tt,
        car_passenger=car_passenger_tt,
        walk=walk_tt
    )