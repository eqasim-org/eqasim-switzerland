"""Write selected road demand, clipping completed road paths to Switzerland."""

import hashlib
import json
import logging
from pathlib import Path

from lxml import etree

from sumo.config import tools as config_tools
from sumo.scenario import modes
from sumo.scenario.demand_tools import TripSpool
from sumo.scenario.population.vehicle_tools import BATCH_SIZE
from sumo.tables import read_table
from sumo.cutter.parallel import clip_batches, merge_reports, worker_count

logger = logging.getLogger("synpp")


def configure(context):
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.scenario.unrouted_demand")
    context.config("sumo_native_xml", True)
    if context.config("sumo_native_xml"):
        context.stage("sumo.native.runtime")
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo"):
        context.config("threads", volatile=True)
        context.config("sumo_demand_workers", None)
        context.stage("mode_choice.routing.sumo")
        context.stage("sumo.scenario.border")
        config_tools.configure_network(context)
        context.config("sumo_mapmatch_distance", None)


def execute(context):

    # Read compact source manifests and the already selected raw trip columns.

    raw = context.stage("sumo.scenario.unrouted_demand")
    frame = read_table(raw["table"])
    types = {element.get("id"): element for element in
             etree.fromstring(b"<routes>" + Path(raw["types"]).read_bytes() + b"</routes>")}
    cache = Path(context.path()).resolve()
    native = context.stage("sumo.native.runtime")["writer"] if context.config("sumo_native_xml") else None

    # Join completed itineraries once, then keep every domestic routed visit.

    settings = None
    report = None
    if context.config("clip_trips_to_border_in_sumo"):
        routed = context.stage("mode_choice.routing.sumo")
        frame = frame.join(read_table(routed["routes"]), on="id", how="left", validate="1:1", maintain_order="left")
        settings = {"border": context.stage("sumo.scenario.border"), "links": routed["links"],
                    "routes": routed["routes"], "network": config_tools.network_path(context),
                    "distance": context.config("sumo_mapmatch_distance"), "types": raw["types"]}

    # Sort the final departures and export XML only once, using the shared writer.

    with TripSpool(cache / "demand.arrow", native_writer=native) as records:
        if settings is not None:
            workers = worker_count(context.config("sumo_demand_workers"), context.config("threads"), frame.height)
            for batch, partial in clip_batches(frame, settings, cache, workers):
                records.add_frame(batch)
                report = merge_reports(report, partial)
        else:
            for batch in frame.iter_slices(BATCH_SIZE):
                records.add_frame(batch)
        result = records.write(cache / "queries.trips.xml.gz", types)
    if report is not None:
        target = cache / "border_clipping.json"
        target.write_text(json.dumps(report, indent=2) + "\n")
        result["border_clipping"] = str(target)
        logger.info("SUMO routed Swiss-border clipping: %s", report)
    return result


def validate(context):
    fingerprint = hashlib.sha256()
    from sumo.cutter import route_tools, demand_tools as cutter_tools, parallel
    from sumo.scenario import demand_tools
    from sumo.scenario.network import search
    for module in (route_tools, cutter_tools, demand_tools, parallel, search):
        fingerprint.update(Path(module.__file__).read_bytes())
    return fingerprint.hexdigest()
