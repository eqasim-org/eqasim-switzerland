"""Generate a native marouter configuration for flow-dependent road assignment."""

from pathlib import Path

from sumo.config import tools


def configure(context):
    context.stage("sumo.router.mapped")
    configure_profile(context)


def execute(context):

    # Write the shared assignment recipe for mapped scenario demand.

    return write(context, context.stage("sumo.router.mapped"))


def configure_profile(context):
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_end", 108000)
    context.config("sumo_assignment_method", "incremental")
    context.config("sumo_assignment_iterations", 50)
    context.config("sumo_assignment_interval", 3600)
    context.config("sumo_assignment_scale", 1.0)
    context.config("sumo_assignment_previous_weights", None)
    context.config("sumo_assignment_weight_adaption", 0.0)
    context.config("sumo_assignment_options", None)


def write(context, demand, method=None):

    # Load mapped road demand, retaining separate queries for non-driving trips.

    method = context.config("sumo_assignment_method") if method is None else method
    if method not in {"incremental", "SUE"}:
        raise ValueError("sumo_assignment_method must be 'incremental' or 'SUE'; native UE is not implemented")
    begin = float(context.config("sumo_begin"))
    end = tools.positive(context.config("sumo_end"), "sumo_end")
    if not 0 <= begin < end:
        raise ValueError("Require 0 <= sumo_begin < sumo_end")
    if demand.get("first_departure", begin) < begin or demand.get("last_departure", begin) >= end:
        raise ValueError("sumo_begin/sumo_end must include every trip departure")
    adaption = float(context.config("sumo_assignment_weight_adaption"))
    if not 0 <= adaption < 1:
        raise ValueError("sumo_assignment_weight_adaption must be in [0, 1) so new flow can update costs")
    previous = context.config("sumo_assignment_previous_weights")

    # Configure incremental loading or stochastic assignment using native volume-delay functions.

    sections = {
        "input": {"net-file": demand["network"], "route-files": demand["loading"],
                  "weight-files": None if previous is None else tools.existing_file(previous),
                  "weight-attribute": "traveltime", "weight-adaption": adaption},
        "time": {"begin": begin, "end": end},
        "processing": {
            "assignment-method": method,
            "max-iterations": tools.positive(context.config("sumo_assignment_iterations"), "sumo_assignment_iterations", integer=True),
            "aggregation-interval": tools.positive(context.config("sumo_assignment_interval"), "sumo_assignment_interval"),
            "scale": tools.positive(context.config("sumo_assignment_scale"), "sumo_assignment_scale"),
            "routing-algorithm": "dijkstra", "capacities.default": True,
        },
        "output": {"output-file": "assigned.flows.xml.gz", "netload-output": "loaded.edges.xml.gz", "precision": 8},
        "report": {"ignore-errors": False, "log": "assignment.log"},
        "random_number": {"seed": context.config("random_seed")},
    }
    return tools.write_config(Path(context.path()) / "macroscopic.marocfg", "marouterConfiguration", sections,
                              context.config("sumo_assignment_options"))


def validate(context):
    return tools.external_token(context, ("sumo_assignment_previous_weights",))
