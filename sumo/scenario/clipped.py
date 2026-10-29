"""Clip original cross-border trips before demand expansion."""

import hashlib
import json
import logging
from pathlib import Path

import polars as pl
from lxml import etree

from sumo.config import tools as config_tools
from sumo.scenario import modes
from sumo.scenario.demand_tools import TripSpool
from sumo.scenario.selection import fixed_endpoints
from sumo.tables import read_table, write_table
from sumo.cutter.parallel import clip_batches, merge_reports, worker_count

logger = logging.getLogger("synpp")


def configure(context):
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.scenario.selected")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.config("threads", volatile=True)
        context.config("sumo_demand_workers", None)
        context.stage("sumo.routing.paths")
        context.stage("sumo.scenario.border")
        config_tools.configure_network(context)
        context.config("sumo_mapmatch_distance", None)


def execute(context):

    # Pass through domestic trips; route and clip only original cross-border trips.

    raw = context.stage("sumo.scenario.selected")
    if not context.config("clip_trips_to_border_in_sumo"):
        return raw
    cache = Path(context.path()).resolve()
    crossing = read_table(raw["crossing_table"])
    routed = context.stage("sumo.routing.paths")
    crossing = crossing.join(read_table(routed["routes"]), on="id", how="left", validate="1:1", maintain_order="left")
    types = {element.get("id"): element for element in
             etree.fromstring(b"<routes>" + Path(raw["types"]).read_bytes() + b"</routes>")}
    report = None
    fixed = []

    # Cache clipped originals as Arrow; expansion and final XML writing happen later.

    with TripSpool(cache / "demand.arrow") as records:
        records.add_frame(read_table(raw["domestic_table"]))
        if crossing.height:
            settings = {"border": raw["border"], "links": routed["links"], "routes": routed["routes"],
                        "network": config_tools.network_path(context),
                        "distance": context.config("sumo_mapmatch_distance"), "types": raw["types"]}
            workers = worker_count(context.config("sumo_demand_workers"), context.config("threads"), crossing.height)
            for batch, partial in clip_batches(crossing, settings, cache, workers):
                records.add_frame(batch)
                fixed.append(fixed_endpoints(batch, crossing, raw["border"]))
                report = merge_reports(report, partial)
        result = records.write(cache / "unused.xml", types, write_xml=False)
    fixed = pl.concat(fixed).unique() if fixed else pl.DataFrame(schema={"x": pl.Float64, "y": pl.Float64})
    fixed_path = write_table(fixed, cache / "fixed_endpoints.arrow")
    target = cache / "border_clipping.json"
    target.write_text(json.dumps(report or {"input": 0, "output": 0}, indent=2) + "\n")
    logger.info("SUMO original-trip Swiss-border clipping: %s", report)
    return {**result, "border": raw["border"], "fixed_endpoints": fixed_path, "border_clipping": str(target)}


def validate(context):
    fingerprint = hashlib.sha256()
    from sumo.cutter import route_tools, demand_tools as cutter_tools, parallel
    from sumo.scenario import selection
    from sumo.scenario import demand_tools
    from sumo.scenario.network import search
    for module in (route_tools, cutter_tools, demand_tools, parallel, search, selection):
        fingerprint.update(Path(module.__file__).read_bytes())
    return fingerprint.hexdigest()
