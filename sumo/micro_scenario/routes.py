"""Stream national routes into bounded, independently processed regional Arrow parts."""

import hashlib
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import polars as pl
from lxml import etree

from sumo.functions.native import processing
from sumo.functions.scenario.network.tools import read_network
from sumo.functions.scenario.tools import iter_routes

SCHEMA = {"id": pl.String, "depart": pl.Float64, "xml": pl.Binary}

logger = logging.getLogger("synpp")

from sumo.functions.micro_scenario.routes import _native_parts, _process
from scripts.monitoring import monitored
from sumo.functions.source import source_digest


def configure(context):
    context.stage("sumo.macro_scenario")
    context.stage("sumo.micro_scenario.network")
    context.config("threads", volatile=True)
    context.config("sumo_native_routes", True)
    if context.config("sumo_native_routes"):
        context.stage("sumo.native.routes")
        processing.configure_stage(context, "cut")


@monitored
def execute(context):

    # Read reusable network caches and immutable mapping once for all worker threads.

    started = time.monotonic()
    loaded = context.stage("sumo.macro_scenario")
    region = context.stage("sumo.micro_scenario.network")
    if context.config("sumo_native_routes") and loaded.get("route_format") == "arrow":
        return _native_parts(context, loaded, region)
    source = read_network(loaded["network"], context=context)
    cut = read_network(region["network"], context=context)
    mapping = {
        key: value
        for key, value in json.loads(Path(region["mapping"]).read_text()).items()
        if value
    }
    transitions = {(c["from"], c["to"]) for c in cut["connections"]}
    kernel = None
    if context.config("sumo_native_routes"):
        from sumo.functions.native.kernels import RouteKernel

        kernel = RouteKernel(
            context.stage("sumo.native.routes"), mapping, source, transitions
        )
    cache = Path(context.path()).resolve()
    workers = max(1, int(context.config("threads")))
    types = {}
    pending = deque()
    parts = []
    report = {
        "source_count": 0,
        "count": 0,
        "outside_or_discarded": 0,
        "repeated_visits": 0,
        "interpolated_visits": 0,
    }

    # Keep only a bounded number of batches in flight, with deterministic collection order.

    def collect(future):
        rows, counts = future.result()
        path = cache / f"part_{len(parts):06d}.arrow"
        pl.DataFrame(rows, schema=SCHEMA).write_ipc(path)
        parts.append(str(path))
        for key, value in counts.items():
            report[key] += value

    selected_edges = set(mapping)
    with ExitStack() as resources, ThreadPoolExecutor(max_workers=workers) as pool:
        route_filter = None
        if kernel is not None:
            from sumo.functions.native.kernels import RouteFilter

            route_filter = resources.enter_context(
                RouteFilter(context.stage("sumo.native.routes"), selected_edges)
            )
        batch = []
        for tag, vehicle in iter_routes(
            loaded["routes"], selected_edges=selected_edges, route_filter=route_filter
        ):
            if tag == "vType":
                types[vehicle.get("id")] = vehicle
                continue
            report["source_count"] += 1
            if report["source_count"] % 100000 == 0:
                logger.info(
                    "SUMO regional cutting: scanned %d national routes; %d visits written; %.1fs",
                    report["source_count"],
                    report["count"],
                    time.monotonic() - started,
                )
            if tag == "outside":
                report["outside_or_discarded"] += 1
                continue
            batch.append(vehicle)
            if len(batch) >= 1024:
                pending.append(
                    pool.submit(
                        _process, batch, mapping, source, cut, transitions, kernel
                    )
                )
                batch = []
                if len(pending) >= 2 * workers:
                    collect(pending.popleft())
        if batch:
            pending.append(
                pool.submit(_process, batch, mapping, source, cut, transitions, kernel)
            )
        while pending:
            collect(pending.popleft())

    # Publish cached parts; the export stage validates the complete regional result.

    definitions = cache / "types.xml"
    definitions.write_bytes(b"".join(etree.tostring(value) for value in types.values()))
    result = {
        "parts": parts,
        "types": str(definitions),
        "network": region["network"],
        **report,
    }
    (cache / "parts.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "SUMO regional cutting complete: %d source routes, %d visits; %.1fs",
        report["source_count"],
        report["count"],
        time.monotonic() - started,
    )
    return result


def validate(context):
    from sumo.functions.cutter import tools
    from sumo.functions.native import kernels
    from sumo.functions.scenario import route_io
    from sumo.functions.scenario import tools as scenario_tools

    return {
        "inputs": (
            [
                hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (tools, scenario_tools, kernels, route_io)
            ]
        ),
        "helpers": source_digest("sumo.functions.micro_scenario.routes"),
    }
