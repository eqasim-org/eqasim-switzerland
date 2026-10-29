"""Classify original SUMO trips by endpoints before any routing or expansion."""

import json
import logging
from pathlib import Path

from sumo.scenario import selection
from sumo.tables import read_table, write_table

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("sumo.scenario.unrouted_demand")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.stage("sumo.scenario.border")


def execute(context):

    # Classify original endpoints with native vectorized polygon predicates.

    raw = context.stage("sumo.scenario.unrouted_demand")
    if not context.config("clip_trips_to_border_in_sumo"):
        return raw
    frame = read_table(raw["table"])
    border = context.stage("sumo.scenario.border")
    domestic, crossing, outside = selection.classify(frame, border)
    cache = Path(context.path()).resolve()
    retained = frame.filter(domestic | crossing)
    report = {"input": frame.height, "domestic": int(domestic.sum()),
              "crossing": int(crossing.sum()), "outside": int(outside.sum())}
    (cache / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("SUMO original-trip Swiss endpoint selection: %s", report)

    # Only crossing requests require extended-network itineraries.

    return {**raw, "table": write_table(retained, cache / "selected.arrow"),
            "domestic_table": write_table(frame.filter(domestic), cache / "domestic.arrow"),
            "crossing_table": write_table(frame.filter(crossing), cache / "crossing.arrow"),
            "count": retained.height, "border": border, "selection": str(cache / "selection.json")}


def validate(context):
    import hashlib
    return hashlib.sha256(Path(selection.__file__).read_bytes()).hexdigest()
