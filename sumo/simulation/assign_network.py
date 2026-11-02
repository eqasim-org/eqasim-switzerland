"""Run the routing engine, keeping its outputs separate from route realization."""

from sumo.simulation.loading import configure_loading, run_macroscopic, run_iterations


def configure(context):
    configure_loading(context)
    context.stage("sumo.simulation.prepare_loading")


def execute(context):

    # Consume prepared endpoints without repeating mapping on a routing rerun.

    prepared = context.stage("sumo.simulation.prepare_loading")
    method = context.config("sumo_loading_method")
    if method == "dua":
        routes, weights, report = run_iterations(context, prepared["network"], prepared["trips"])
    else:
        routes, weights, report = run_macroscopic(context, prepared["network"], prepared["trips"], method, realize=False)
    return {**prepared, "routes": str(routes), "weights": str(weights), "method": method, "report": report}


def validate(context):
    import hashlib
    from pathlib import Path
    from sumo.simulation import loading
    from sumo.config import macroscopic
    from sumo.config import tools
    return [tools.external_token(context, ("sumo_assignment_previous_weights",)), *[hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (loading,macroscopic)]]
