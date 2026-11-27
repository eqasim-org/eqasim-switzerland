"""Complete selected SUMO road paths without changing anyone's mode choice."""

import json
import logging
from pathlib import Path

import polars as pl

from mode_choice.routing import options
from sumo.functions.routing.java import SERVER_STAGE, RoutingServer
from sumo.functions.routing.network import attach_road_endpoints

logger = logging.getLogger("synpp")

ROUTE_SCHEMA = {
    "id": pl.String,
    "route": pl.String,
    "travel_time": pl.Float64,
    "travel_distance": pl.Float64,
}

from sumo.functions.monitoring.runtime import monitored
from sumo.functions.routing.paths import (
    completed_routes,
    reusable_routes,
    start_completion_server,
    trip_requests,
)
from sumo.functions.source import source_digest


def configure(context):
    options.configure(context)
    context.config("enable_mode_choice", False)
    if not context.config("enable_mode_choice"):
        raise ValueError("SUMO requires enable_mode_choice: true")
    if context.config("mode_choice_car_routing_backend") != "java":
        raise ValueError(
            "SUMO route reuse requires mode_choice_car_routing_backend: java"
        )
    context.stage("sumo.macro_scenario.selected")
    context.stage("mode_choice.routing_variables.car")
    context.stage(SERVER_STAGE)
    context.stage("matsim.scenario.network.pt")
    context.stage("sumo.routing.network")


@monitored
def execute(context):

    # Read only request columns, leaving XML payloads in the raw demand cache.

    cache = Path(context.path()).resolve()
    raw = context.stage("sumo.macro_scenario.selected")
    requests = trip_requests(raw["crossing_table"])
    reused_path = cache / "reused.arrow"
    source = Path(context.path("mode_choice.routing_variables.car")) / "car_routes.csv"
    reused = reusable_routes(source, requests)
    reused.sink_ipc(reused_path, compression="uncompressed")
    reused_ids = pl.read_ipc(reused_path, columns=["id"])
    if reused_ids["id"].n_unique() != reused_ids.height:
        raise ValueError("Duplicate reusable mode-choice paths for the same SUMO trip")
    missing = requests.join(reused_ids, on="id", how="anti")

    # Route only gaps, using the same extended network and Java road router.

    settings = context.stage(SERVER_STAGE)
    network = settings["command"][settings["command"].index("--network-path") + 1]
    additions = cache / "additional_routes.arrow"
    try:
        if missing.height:
            logger.info(
                "SUMO paths: reusing %d; routing %d missing/changed road trips.",
                reused_ids.height,
                missing.height,
            )
            request_file = cache / "additional_requests.csv"
            routable = attach_road_endpoints(
                missing, context.stage("sumo.routing.network")["links"]
            )
            routable.rename({"id": "identifier", "router_mode": "mode"}).write_csv(
                request_file
            )
            output = cache / "additional_routes.csv"
            start_completion_server(settings)
            RoutingServer.route_csv(request_file, output, None, compute_cost=False)
            extra = completed_routes(output, missing)
        else:
            extra = pl.DataFrame(schema=ROUTE_SCHEMA)
        extra.write_ipc(additions, compression="uncompressed")
    finally:
        RoutingServer.stop_server()

    # Publish small manifests and Arrow paths; do not pickle itinerary tables.

    target = cache / "road_routes.arrow"
    pl.concat([pl.scan_ipc(reused_path), pl.scan_ipc(additions)]).sink_ipc(
        target, compression="uncompressed"
    )
    report = {
        "requested": requests.height,
        "reused": reused_ids.height,
        "routed": missing.height,
    }
    (cache / "routing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("SUMO path preparation: %s", report)
    geometry = context.stage("matsim.scenario.network.pt").get("geometry")
    return {
        "routes": str(target),
        "network": str(network),
        "geometry": geometry,
        "links": context.stage("sumo.routing.network")["links"],
        "report": str(cache / "routing_report.json"),
        **report,
    }


def validate(context):
    return source_digest("sumo.functions.routing.paths")
