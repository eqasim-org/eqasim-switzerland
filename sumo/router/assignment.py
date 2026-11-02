"""Run native macroscopic loading, then report a loaded route metric for every trip."""

import json
from pathlib import Path

from lxml import etree

from sumo.config import tools
from sumo.router.tools import write_trip_metrics
from sumo.runtime import sumo


def configure(context):
    context.stage("sumo.runtime.sumo")
    context.stage("sumo.native.marouter")
    context.stage("sumo.config.loaded")


def execute(context):

    # Materialize configs locally so cached configuration stages remain read-only.

    settings = context.stage("sumo.config.loaded")
    cache = Path(context.path()).resolve()
    assignment = tools.relocate_config(settings["assignment_config"], cache / "assignment.marocfg")
    routing = tools.relocate_config(settings["routing_config"], cache / "loaded.duarcfg", {
        Path(settings["weights"]).resolve(): cache / "loaded.edges.xml.gz",
    })

    # Assign road demand before computing any of the final query routes.

    sumo.run(context, ["--configuration-file", assignment], binary="marouter",
             executable=context.stage("sumo.native.marouter"))
    sumo.run(context, ["--configuration-file", routing], binary="duarouter")

    # Export source identifiers, reference times, loaded travel times and distances.

    routes = cache / "loaded.routes.xml.gz"
    metrics = cache / "trip_metrics.arrow"
    count = write_trip_metrics(routes, metrics, settings["count"])
    recipe = etree.parse(assignment)
    result = {"metrics": str(metrics), "weights": str(cache / "loaded.edges.xml.gz"),
              "flows": str(cache / "assigned.flows.xml.gz"), "routes": str(routes), "count": count,
              "assignment_method": recipe.find("processing/assignment-method").get("value"),
              "demand_scale": float(recipe.find("processing/scale").get("value")),
              "cost_model": "SUMO marouter native volume-delay"}
    (cache / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
