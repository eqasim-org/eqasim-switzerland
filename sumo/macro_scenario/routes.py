"""Realize macroscopic route shares and loaded edge clocks in a separate cache."""

from pathlib import Path

from sumo.functions.macro_scenario.routes import _native_routes
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.native import processing
from sumo.functions.scenario.assignment import materialize_assignment
from sumo.functions.source import source_digest


def configure(context):
    context.stage("sumo.macro_scenario.assignment")
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_assignment_interval", 3600)
    context.config("sumo_loading_method", "incremental")
    context.config("sumo_native_routes", True)
    if (
        context.config("sumo_native_routes")
        and context.config("sumo_loading_method") != "dua"
    ):
        context.stage("sumo.macro_scenario.allocation")
        processing.configure_stage(context, "realize")


@monitored
def execute(context):

    # Keep simulation-based loading on its experienced-route publication path.

    assigned = context.stage("sumo.macro_scenario.assignment")
    if assigned["method"] == "dua":
        return {
            **assigned,
            "convergence": assigned["report"],
            "timing": "experienced edge exit times",
        }
    if context.config("sumo_native_routes"):
        return _native_routes(context, assigned)

    # Retain the Python implementation for explicit debugging and compiler-free runs.

    target = Path(context.path()) / "assigned.routes.xml.gz"
    report = materialize_assignment(
        assigned["routes"],
        assigned["weights"],
        assigned["trips"],
        assigned["network"],
        target,
        context.config("random_seed"),
        context.config("sumo_begin"),
        context.config("sumo_assignment_interval"),
        context=context,
        metrics_path=Path(context.path()) / "trip_metrics.arrow",
    )
    return {
        **assigned,
        "routes": str(target),
        "convergence": {
            "method": assigned["method"],
            "equilibrium_guaranteed": False,
            "realization": report,
        },
        "timing": "macroscopic estimated edge exit times",
    }


def validate(context):
    import hashlib

    from sumo.functions.native import kernels
    from sumo.functions.scenario import assignment, assignment_spool, metrics, route_io
    from sumo.functions.scenario.network import tools as network_tools

    return {
        "inputs": (
            [
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (
                    assignment,
                    assignment_spool,
                    kernels,
                    metrics,
                    network_tools,
                    route_io,
                )
            ]
        ),
        "helpers": source_digest("sumo.functions.macro_scenario.routes"),
    }
