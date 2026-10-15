"""Car time and distance, from Java or the hourly Pandana network."""
from mode_choice.routing.java import configure_mode, car_backend, route_mode, save_costs
from mode_choice.routing.congestion import input_path
from mode_choice.costs.car import calculate_driving_costs
from mode_choice.model.parameters import load
from mode_choice.defaults import RUN_DEFAULTS


def configure(context):
    configure_mode(context, "car")
    if car_backend(context) == "pandana":
        context.stage("matsim.scenario.network.mapped")
        context.stage("dmc.params")
        context.config("car_cost_model", RUN_DEFAULTS["car_cost_model"])


def execute(context):
    # If the backend is Java, we use the Java routing.
    if car_backend(context) == "java":
        return route_mode(context, "car")

    # Otherwise, pandana routing
    from mode_choice.routing.pandana import route

    trips = context.stage("mode_choice.data.prepare").rename({"trip_id": "identifier"})
    if trips.is_empty():
        raise ValueError("No trips eligible for mode choice")

    routes = route(context, trips, context.config("mode_choice_routing_batch_size")).rename({"identifier": "trip_id"})

    # Compute costs locally with DMC, keeping the same artifact contract as Java.
    routes = calculate_driving_costs(routes, load(context), context.config("car_cost_model"))
    return save_costs(context, routes, "car")


def validate(context):
    path = input_path(context, car_backend(context))
    if path is not None:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino
