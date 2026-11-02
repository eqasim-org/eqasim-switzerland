"""Independent Java or vectorized detour/speed routing for walk and bike."""
from mode_choice.defaults import RUN_DEFAULTS, ACTIVE_ROUTING_DEFAULTS, ACTIVE_ROUTING_BACKENDS
from mode_choice.routing.java.router import route_mode
from mode_choice.routing import options


def configure(context, mode):
    options.configure(context)
    context.stage("mode_choice.defaults")
    context.stage("mode_choice.data.prepare")
    name = f"mode_choice_{mode}_routing_backend"
    context.config(name, ACTIVE_ROUTING_DEFAULTS[name])
    if routing_backend(context, mode) == "java":
        context.stage("mode_choice.routing.java.server")
        context.stage("mode_choice.routing.java.requests")
        if context.config("route_population_once"):
            context.stage("mode_choice.routing.java.preparation")
            context.stage("mode_choice.routing.java.config")
    else:
        for name in (f"{mode}_speed_m_per_s", f"{mode}_distance_factor"):
            context.config(name, RUN_DEFAULTS[name])


def execute(context, mode):

    # Route with the shared Java server when requested.

    if routing_backend(context, mode) == "java":
        return route_mode(context, mode)

    from mode_choice.routing.python.active import route
    return route(context, mode)


def routing_backend(context, mode):
    name = f"mode_choice_{mode}_routing_backend"
    backend = context.config(name)
    if backend not in ACTIVE_ROUTING_BACKENDS:
        raise ValueError(f"{name} must be one of {ACTIVE_ROUTING_BACKENDS}")
    return backend
