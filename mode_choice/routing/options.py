"""Shared routing option declarations and compatibility checks."""
from mode_choice.defaults import CAR_ROUTING_BACKENDS


def configure(context):
    context.config("use_marouter_in_mode_choice", True)
    context.config("route_population_once", False)
    context.config("prepare_sumo", False)
    context.config("mode_choice_car_routing_backend", "java")
    context.config("java_memory", "50G", volatile=True)
    for name in ("use_marouter_in_mode_choice", "route_population_once", "prepare_sumo"):
        if not isinstance(context.config(name), bool):
            raise ValueError(f"{name} must be true or false")
    backend = context.config("mode_choice_car_routing_backend")
    if backend not in CAR_ROUTING_BACKENDS:
        raise ValueError(f"mode_choice_car_routing_backend must be one of {CAR_ROUTING_BACKENDS}")
    if context.config("prepare_sumo") and backend != "java":
        raise ValueError("prepare_sumo requires mode_choice_car_routing_backend: java for link-path export")
    if backend == "pandana" and (context.config("use_marouter_in_mode_choice")
                                 or context.config("route_population_once")):
        raise ValueError("Pandana car routing requires use_marouter_in_mode_choice=false "
                         "and route_population_once=false")
