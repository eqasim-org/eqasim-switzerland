"""Sort and export cached regional route parts without repeating national cutting."""

import hashlib
import json
from pathlib import Path

import polars as pl
from lxml import etree

from sumo.functions.micro_scenario.demand import _native_export
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.native import processing
from sumo.functions.scenario.network.tools import read_network, validate_routes
from sumo.functions.scenario.rerouting import (
    retain_departure_routes,
    retention_settings,
)
from sumo.functions.scenario.tools import iter_routes, write_serialized_trips
from sumo.functions.source import source_digest


def configure(context):
    context.stage("sumo.micro_scenario.routes")
    context.config("random_seed")
    context.config("sumo_initial_route_retention", 0.8)
    context.config("sumo_native_routes", True)
    if context.config("sumo_native_routes"):
        processing.configure_stage(context, "export")


@monitored
def execute(context):

    # Merge only retained regional Arrow rows and sort with Polars.

    retention_settings(context)
    source = context.stage("sumo.micro_scenario.routes")
    if source.get("route_format") == "arrow":
        return _native_export(context, source)
    frames = [pl.scan_ipc(path) for path in source["parts"]]
    frame = (
        pl.concat(frames).sort("depart", "id").collect()
        if frames
        else pl.DataFrame(
            schema={"id": pl.String, "depart": pl.Float64, "xml": pl.Binary}
        )
    )
    if frame["id"].n_unique() != frame.height:
        raise ValueError("Duplicate regional vehicle IDs")
    types = {
        element.get("id"): element
        for element in etree.fromstring(
            b"<routes>" + Path(source["types"]).read_bytes() + b"</routes>"
        )
    }

    # Write and validate regional routes, retaining every contiguous routed visit.

    cache = Path(context.path()).resolve()
    output = cache / "region.routes.xml.gz"
    write_serialized_trips(
        output,
        types,
        retain_departure_routes(frame["xml"], *retention_settings(context)),
    )
    data = read_network(source["network"], context=context)
    state = {}
    for tag, vehicle in iter_routes(output):
        if tag == "vehicle":
            validate_routes([vehicle], data, types, state=state)
    result = {
        key: value for key, value in source.items() if key not in {"parts", "types"}
    }
    result["routes"] = str(output)
    (cache / "cut_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate(context):
    from sumo.functions.scenario.network import tools

    return {
        "inputs": (hashlib.sha256(Path(tools.__file__).read_bytes()).hexdigest()),
        "helpers": source_digest("sumo.functions.micro_scenario.demand"),
    }
