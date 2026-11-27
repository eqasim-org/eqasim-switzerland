"""Complete selected SUMO road paths without changing anyone's mode choice."""

import logging

import polars as pl
from pyproj import Transformer

from sumo.functions.routing.java import RoutingServer

logger = logging.getLogger("synpp")

ROUTE_SCHEMA = {
    "id": pl.String,
    "route": pl.String,
    "travel_time": pl.Float64,
    "travel_distance": pl.Float64,
}


def trip_requests(path):
    columns = [
        "id",
        "person_id",
        "trip_index",
        "depart",
        "origin",
        "destination",
        "mode",
    ]
    frame = pl.read_ipc(path, columns=columns)
    transform = Transformer.from_crs(4326, 2056, always_xy=True)
    for source, prefix in (("origin", "origin"), ("destination", "destination")):
        coords = frame[source].str.split_exact(",", 1)
        lon = coords.struct.field("field_0").cast(pl.Float64, strict=False).to_numpy()
        lat = coords.struct.field("field_1").cast(pl.Float64, strict=False).to_numpy()
        x, y = transform.transform(lon, lat, errcheck=True)
        frame = frame.with_columns(
            pl.Series(prefix + "_x", x), pl.Series(prefix + "_y", y)
        )
    frame = (
        frame.with_columns(
            pl.col("person_id").cast(pl.String),
            pl.col("trip_index").cast(pl.Int64),
            pl.when(
                pl.col("mode")
                .str.strip_suffix("_loop")
                .is_in(["truck", "lcv", "delivery"])
            )
            .then(pl.lit("truck"))
            .otherwise(pl.lit("car"))
            .alias("router_mode"),
        )
        .rename({"depart": "departure_time"})
        .drop("origin", "destination", "mode")
    )
    numeric = [
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
        "departure_time",
    ]
    if frame.filter(
        ~pl.all_horizontal(pl.col(numeric).is_finite()).fill_null(False)
        | pl.col("person_id").is_null()
        | pl.col("trip_index").is_null()
    ).height:
        raise ValueError(
            "SUMO routing requests have invalid coordinates or person/trip keys"
        )
    return frame


def reusable_routes(path, requests):
    if not path.is_file():
        return pl.DataFrame(schema=ROUTE_SCHEMA).lazy()
    columns = (
        pl.read_csv_schema(path)
        if hasattr(pl, "read_csv_schema")
        else pl.scan_csv(path).collect_schema()
    )
    required = {
        "person_id",
        "trip_index",
        "mode",
        "route",
        "travel_time",
        "travel_distance",
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
        "departure_time",
    }
    if not required.issubset(columns):
        return pl.DataFrame(schema=ROUTE_SCHEMA).lazy()
    routes = pl.scan_csv(
        path,
        schema_overrides={
            "person_id": pl.String,
            "trip_index": pl.Int64,
            "route": pl.String,
            "mode": pl.String,
        },
    )
    routes = routes.select(*required).rename({"mode": "router_mode"})
    joined = routes.join(
        requests.lazy(),
        on=["person_id", "trip_index", "router_mode"],
        how="inner",
        suffix="_requested",
    )
    valid = pl.col("route").is_not_null() & pl.col("route").str.starts_with("{")
    valid &= pl.col("travel_time").is_finite() & (pl.col("travel_time") >= 0)
    valid &= pl.col("travel_distance").is_finite() & (pl.col("travel_distance") >= 0)
    for prefix in ("origin", "destination"):
        valid &= (
            (pl.col(prefix + "_x") - pl.col(prefix + "_x_requested")).pow(2)
            + (pl.col(prefix + "_y") - pl.col(prefix + "_y_requested")).pow(2)
        ) <= 4
    valid &= (pl.col("departure_time") - pl.col("departure_time_requested")).abs() <= 1
    return joined.filter(valid.fill_null(False)).select(*ROUTE_SCHEMA)


def completed_routes(path, requests):
    frame = pl.read_csv(
        path, schema_overrides={"identifier": pl.String, "route": pl.String}
    )
    required = {"identifier", "route", "travel_time", "travel_distance"}
    if not required.issubset(frame.columns):
        raise ValueError(
            "Java must export actual link itineraries for SUMO routing completion"
        )
    frame = frame.select(
        pl.col("identifier").alias("id"), "route", "travel_time", "travel_distance"
    ).cast(ROUTE_SCHEMA)
    valid = frame.filter(
        pl.col("route").is_not_null()
        & pl.col("route").str.starts_with("{")
        & pl.col("travel_time").is_finite()
        & (pl.col("travel_time") >= 0)
        & pl.col("travel_distance").is_finite()
        & (pl.col("travel_distance") >= 0)
    )
    absent = requests.select("id").join(valid.select("id"), on="id", how="anti")
    if (
        valid.height != frame.height
        or absent.height
        or valid["id"].n_unique() != requests.height
        or valid.join(requests.select("id"), on="id", how="anti").height
    ):
        raise ValueError(
            f"Incomplete SUMO road routing; {absent.height} missing trips, examples: {absent['id'].head(5).to_list()}"
        )
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
