"""Realize macroscopic route shares and loaded edge clocks in a separate cache."""

from pathlib import Path

from sumo.scenario.assignment import materialize_assignment


def configure(context):
    context.stage("sumo.simulation.assign_network")
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_assignment_interval", 3600)
    context.config("sumo_native_routes", True)
    if context.config("sumo_native_routes"):
        context.stage("sumo.native.routes")


def execute(context):

    # Keep seeded route allocation in Python and use native numerical edge clocks.

    assigned = context.stage("sumo.simulation.assign_network")
    if assigned["method"] == "dua":
        return {**assigned, "convergence": assigned["report"], "timing": "experienced edge exit times"}
    kernel = None
    if context.config("sumo_native_routes"):
        from sumo.native.kernels import RouteKernel
        kernel = RouteKernel(context.stage("sumo.native.routes"))
    target = Path(context.path()) / "assigned.routes.xml.gz"
    report = materialize_assignment(assigned["routes"], assigned["weights"], assigned["trips"],
        assigned["network"], target, context.config("random_seed"), context.config("sumo_begin"),
        context.config("sumo_assignment_interval"), context=context, kernel=kernel)
    return {**assigned, "routes": str(target), "convergence": {"method": assigned["method"],
            "equilibrium_guaranteed": False, "realization": report}, "timing": "macroscopic estimated edge exit times"}


def validate(context):
    import hashlib
    from sumo.scenario import assignment
    from sumo.native import kernels
    return [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (assignment,kernels)]
