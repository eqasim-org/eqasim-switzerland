"""Shared routing option declarations and compatibility checks."""
from pathlib import Path

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


def input_path(context, backend):
    key = "mode_choice_car_linkstats_path" if backend == "pandana" else "mode_choice_car_events_path"
    value = context.config(key)
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{key}: file does not exist: {path}")
    return path




def configure_pt(context):
    """Declare one consistent pricing default; resolve it from the selected router."""
    context.config('mode_choice_pt_routing_backend', 'java')
    context.config('mode_choice_pt_price_backend', None)
    pt_backend(context)
    pt_price_backend(context)


def pt_backend(context):
    backend = context.config('mode_choice_pt_routing_backend')
    if backend not in ('java', 'flash', 'flash_tb'):
        raise ValueError('mode_choice_pt_routing_backend must be java or flash (flash_tb is an alias)')
    return 'flash' if backend == 'flash_tb' else backend


def pt_price_backend(context):
    backend = context.config('mode_choice_pt_price_backend')
    if backend is None:
        return 'java' if pt_backend(context) == 'java' else 'python'
    if backend not in ('java', 'python'):
        raise ValueError('mode_choice_pt_price_backend must be java or python')
    return backend
