"""Car time and distance, from Java or the hourly Pandana network."""
from mode_choice.routing.java import configure_mode, car_backend, route_mode, save_costs
from mode_choice.routing.congestion import input_path
from mode_choice.costs.car import calculate_driving_costs
from dmc.parameters import Parameters
from mode_choice.defaults import RUN_DEFAULTS
from mode_choice.routing.results import complete_routes


def configure(context):
    configure_mode(context, "car")
    if car_backend(context) == "pandana":
        context.stage("matsim.scenario.network.pt")
        context.stage("dmc.parameter_files")
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
    routes = complete_routes(trips.rename({"identifier": "trip_id"}), routes, ["car_travel_time_min"])

    # Compute costs locally with DMC, keeping the same artifact contract as Java.

    routes = calculate_driving_costs(routes, Parameters.from_context(context), context.config("car_cost_model"))
    return save_costs(context, routes, "car")


def validate(context):
    if car_backend(context) == "java" and context.config("use_marouter_in_mode_choice"):
        return None
    path = input_path(context, car_backend(context))
    if path is not None:
        stat = path.stat()
        return [str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino]
