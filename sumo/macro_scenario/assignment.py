"""Run the routing engine, keeping its outputs separate from route realization."""

from sumo.functions.monitoring.runtime import monitored
from sumo.functions.simulation.loading import (
    configure_loading,
    run_iterations,
    run_macroscopic,
)


def configure(context):
    configure_loading(context)
    context.stage("sumo.macro_scenario.mapped")


@monitored
def execute(context):

    # Consume prepared endpoints without repeating mapping on a routing rerun.

    prepared = context.stage("sumo.macro_scenario.mapped")
    method = context.config("sumo_loading_method")
    if method == "dua":
        routes, weights, report = run_iterations(
            context, prepared["network"], prepared["trips"]
        )
    else:
        routes, weights, report = run_macroscopic(
            context, prepared["network"], prepared["trips"], method, realize=False
        )
    return {
        **prepared,
        "routes": str(routes),
        "weights": str(weights),
        "method": method,
        "report": report,
    }


def validate(context):
    import hashlib
    from pathlib import Path

    from sumo.functions.config import macroscopic, tools
    from sumo.functions.simulation import loading

    return [
        tools.external_token(context, ("sumo_assignment_previous_weights",)),
        *[
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (loading, macroscopic)
        ],
    ]
