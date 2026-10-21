"""Prepare assignment followed by per-trip routing on final loaded edge weights."""

from pathlib import Path

from sumo.config import tools


def configure(context):
    context.stage("sumo.config.macroscopic")
    context.stage("sumo.router.mapped")
    context.config("random_seed")


def execute(context):

    # Copy the assignment recipe so its outputs belong to this workflow's directory.

    return write(context, context.stage("sumo.config.macroscopic"), context.stage("sumo.router.mapped"))


def write(context, assignment_config, demand):
    cache = Path(context.path()).resolve()
    assignment = tools.relocate_config(assignment_config, cache / "assignment.marocfg")

    # Route every query using final assignment costs, without loading those queries again.

    routing = tools.write_config(cache / "loaded.duarcfg", "duarouterConfiguration", {
        "input": {"net-file": demand["network"], "route-files": demand["queries"],
                  "weight-files": str(cache / "loaded.edges.xml.gz"), "weight-attribute": "traveltime"},
        "processing": {"routing-algorithm": "dijkstra", "weights.interpolate": False, "weights.expand": False,
                       "weights.minor-penalty": 0, "weights.tls-penalty": 0, "weights.turnaround-penalty": 0},
        "output": {"output-file": "loaded.routes.xml.gz", "write-costs": True, "route-length": True,
                   "exit-times": True, "precision": 8},
        "report": {"ignore-errors": False, "log": "loaded_routing.log"},
        "random_number": {"seed": context.config("random_seed")},
    })

    # Return both native executable configs and the expected result locations.

    return {"assignment_config": assignment, "routing_config": routing,
            "weights": str(cache / "loaded.edges.xml.gz"), "routes": str(cache / "loaded.routes.xml.gz"),
            "count": demand["count"]}
