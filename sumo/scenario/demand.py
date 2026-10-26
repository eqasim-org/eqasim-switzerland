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
        context.stage("mode_choice.routing.sumo")
        context.stage("sumo.scenario.border")
        config_tools.configure_network(context)
        context.config("sumo_mapmatch_distance", 2000)


def execute(context):

    # Read compact source manifests and the already selected raw trip columns.

    raw = context.stage("sumo.scenario.unrouted_demand")
    frame = read_table(raw["table"])
    types = {element.get("id"): element for element in
             etree.fromstring(b"<routes>" + Path(raw["types"]).read_bytes() + b"</routes>")}
    cache = Path(context.path()).resolve()
    native = context.stage("sumo.native.runtime")["writer"] if context.config("sumo_native_xml") else None

    # Join completed itineraries once, then keep every domestic routed visit.

    clipper = None
    if context.config("clip_trips_to_border_in_sumo"):
        from sumo.cutter.route_tools import RoutedBorderClipper
        routed = context.stage("mode_choice.routing.sumo")
        clipper = RoutedBorderClipper(context.stage("sumo.scenario.border"), routed["links"], routed["routes"],
            lambda: config_tools.network_path(context), context.config("sumo_mapmatch_distance"), types)
        frame = frame.join(clipper.paths, on="id", how="left", validate="1:1", maintain_order="left")

    # Sort the final departures and export XML only once, using the shared writer.

    with TripSpool(cache / "demand.arrow", native_writer=native) as records:
        for batch in frame.iter_slices(BATCH_SIZE):
            records.add_frame(clipper(batch) if clipper is not None else batch)
        result = records.write(cache / "queries.trips.xml.gz", types)
    if clipper is not None:
        report = cache / "border_clipping.json"
        report.write_text(json.dumps(clipper.report, indent=2) + "\n")
        result["border_clipping"] = str(report)
        logger.info("SUMO routed Swiss-border clipping: %s", clipper.report)
    return result


def validate(context):
    fingerprint = hashlib.sha256()
    from sumo.cutter import route_tools, demand_tools as cutter_tools
    from sumo.scenario import demand_tools
    for module in (route_tools, cutter_tools, demand_tools):
        fingerprint.update(Path(module.__file__).read_bytes())
    return fingerprint.hexdigest()
