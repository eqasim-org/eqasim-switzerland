"""Complete selected SUMO road paths without changing anyone's mode choice."""

import json
import logging
from pathlib import Path

import polars as pl
from pyproj import Transformer

from mode_choice.routing import options
from mode_choice.routing.sumo_network import attach_road_endpoints
from mode_choice.routing.RoutingServer import RoutingServer

logger = logging.getLogger("synpp")
ROUTE_SCHEMA = {"id": pl.String, "route": pl.String, "travel_time": pl.Float64, "travel_distance": pl.Float64}


def configure(context):
    options.configure(context)
    context.config("enable_mode_choice", False)
    if not context.config("enable_mode_choice"):
        raise ValueError("SUMO requires enable_mode_choice: true")
    if context.config("mode_choice_car_routing_backend") != "java":
        raise ValueError("SUMO route reuse requires mode_choice_car_routing_backend: java")
    context.stage("sumo.scenario.unrouted_demand")
    context.stage("mode_choice.routing_variables.car")
    context.stage("mode_choice.routing.server")
    context.stage("matsim.scenario.network.pt")
    context.stage("mode_choice.routing.sumo_network")


def execute(context):

    # Read only request columns, leaving XML payloads in the raw demand cache.

    cache = Path(context.path()).resolve()
    raw = context.stage("sumo.scenario.unrouted_demand")
    requests = trip_requests(raw["table"])
    reused_path = cache / "reused.arrow"
    source = Path(context.path("mode_choice.routing_variables.car")) / "car_routes.csv"
    reused = reusable_routes(source, requests)
    reused.sink_ipc(reused_path, compression="uncompressed")
    reused_ids = pl.read_ipc(reused_path, columns=["id"])
    if reused_ids["id"].n_unique() != reused_ids.height:
        raise ValueError("Duplicate reusable mode-choice paths for the same SUMO trip")
    missing = requests.join(reused_ids, on="id", how="anti")

    # Route only gaps, using the same extended network and Java road router.

    settings = context.stage("mode_choice.routing.server")
    network = settings["command"][settings["command"].index("--network-path") + 1]
    additions = cache / "additional_routes.arrow"
    try:
        if missing.height:
            logger.info("SUMO paths: reusing %d; routing %d missing/changed road trips.", reused_ids.height, missing.height)
            request_file = cache / "additional_requests.csv"
            routable = attach_road_endpoints(missing, context.stage("mode_choice.routing.sumo_network")["links"])
            routable.rename({"id": "identifier", "router_mode": "mode"}).write_csv(request_file)
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
    pl.concat([pl.scan_ipc(reused_path), pl.scan_ipc(additions)]).sink_ipc(target, compression="uncompressed")
    report = {"requested": requests.height, "reused": reused_ids.height, "routed": missing.height}
    (cache / "routing_report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("SUMO path preparation: %s", report)
    geometry = context.stage("matsim.scenario.network.pt").get("geometry")
    return {"routes": str(target), "network": str(network), "geometry": geometry,
            "links": context.stage("mode_choice.routing.sumo_network")["links"],
            "report": str(cache / "routing_report.json"), **report}


def trip_requests(path):
    columns = ["id", "person_id", "trip_index", "depart", "origin", "destination", "mode"]
    frame = pl.read_ipc(path, columns=columns)
    transform = Transformer.from_crs(4326, 2056, always_xy=True)
    for source, prefix in (("origin", "origin"), ("destination", "destination")):
        coords = frame[source].str.split_exact(",", 1)
        lon = coords.struct.field("field_0").cast(pl.Float64, strict=False).to_numpy()
        lat = coords.struct.field("field_1").cast(pl.Float64, strict=False).to_numpy()
        x, y = transform.transform(lon, lat, errcheck=True)
        frame = frame.with_columns(pl.Series(prefix + "_x", x), pl.Series(prefix + "_y", y))
    frame = frame.with_columns(
        pl.col("person_id").cast(pl.String), pl.col("trip_index").cast(pl.Int64),
        pl.when(pl.col("mode").str.strip_suffix("_loop").is_in(["truck", "lcv", "delivery"]))
          .then(pl.lit("truck")).otherwise(pl.lit("car")).alias("router_mode"),
    ).rename({"depart": "departure_time"}).drop("origin", "destination", "mode")
    numeric = ["origin_x", "origin_y", "destination_x", "destination_y", "departure_time"]
    if frame.filter(~pl.all_horizontal(pl.col(numeric).is_finite()).fill_null(False)
                    | pl.col("person_id").is_null() | pl.col("trip_index").is_null()).height:
        raise ValueError("SUMO routing requests have invalid coordinates or person/trip keys")
    return frame


def reusable_routes(path, requests):
    if not path.is_file():
        return pl.DataFrame(schema=ROUTE_SCHEMA).lazy()
    columns = pl.read_csv_schema(path) if hasattr(pl, "read_csv_schema") else pl.scan_csv(path).collect_schema()
    required = {"person_id", "trip_index", "mode", "route", "travel_time", "travel_distance",
                "origin_x", "origin_y", "destination_x", "destination_y", "departure_time"}
    if not required.issubset(columns):
        return pl.DataFrame(schema=ROUTE_SCHEMA).lazy()
    routes = pl.scan_csv(path, schema_overrides={"person_id": pl.String, "trip_index": pl.Int64,
                                                "route": pl.String, "mode": pl.String})
    routes = routes.select(*required).rename({"mode": "router_mode"})
    joined = routes.join(requests.lazy(), on=["person_id", "trip_index", "router_mode"], how="inner", suffix="_requested")
    valid = pl.col("route").is_not_null() & pl.col("route").str.starts_with("{")
    valid &= pl.col("travel_time").is_finite() & (pl.col("travel_time") >= 0)
    valid &= pl.col("travel_distance").is_finite() & (pl.col("travel_distance") >= 0)
    for prefix in ("origin", "destination"):
        valid &= ((pl.col(prefix + "_x") - pl.col(prefix + "_x_requested")).pow(2)
                  + (pl.col(prefix + "_y") - pl.col(prefix + "_y_requested")).pow(2)) <= 4
    valid &= (pl.col("departure_time") - pl.col("departure_time_requested")).abs() <= 1
    return joined.filter(valid.fill_null(False)).select(*ROUTE_SCHEMA)


def completed_routes(path, requests):
    frame = pl.read_csv(path, schema_overrides={"identifier": pl.String, "route": pl.String})
    required = {"identifier", "route", "travel_time", "travel_distance"}
    if not required.issubset(frame.columns):
        raise ValueError("Java must export actual link itineraries for SUMO routing completion")
    frame = frame.select(pl.col("identifier").alias("id"), "route", "travel_time", "travel_distance").cast(ROUTE_SCHEMA)
    valid = frame.filter(pl.col("route").is_not_null() & pl.col("route").str.starts_with("{")
                         & pl.col("travel_time").is_finite() & (pl.col("travel_time") >= 0)
                         & pl.col("travel_distance").is_finite() & (pl.col("travel_distance") >= 0))
    absent = requests.select("id").join(valid.select("id"), on="id", how="anti")
    if (valid.height != frame.height or absent.height or valid["id"].n_unique() != requests.height
            or valid.join(requests.select("id"), on="id", how="anti").height):
        raise ValueError(f"Incomplete SUMO road routing; {absent.height} missing trips, examples: {absent['id'].head(5).to_list()}")
    return valid


def start_completion_server(settings):
    command = list(settings["command"])
    for key, value in (("--export-routes", "true"), ("--marouter", "false")):
        if key in command:
            command[command.index(key) + 1] = value
        else:
            command.extend([key, value])
    updated = {**settings, "command": command}
    # Mode choice normally stops its server on completion; a cache-only choice
    # may leave an asynchronously started server owned by this Python process.
    if RoutingServer.is_server_running() and RoutingServer._settings != updated:
        RoutingServer.stop_server()
    RoutingServer.run_server_if_not_running(updated)
