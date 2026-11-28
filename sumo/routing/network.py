"""Cache directed road shapes from the exact extended routing network."""

from pathlib import Path

import polars as pl
import shapely
from pyproj import Transformer

from matsim.readers import read_network
from matsim.scenario.network.utils.link_geometry import (
    GEOMETRY_COLUMN,
    attach_link_geometries,
)
from scripts.monitoring import monitored
from sumo.functions.routing.java import SERVER_STAGE
from sumo.functions.routing.network import (
    attach_road_endpoints as attach_road_endpoints,
)
from sumo.functions.source import source_digest


def configure(context):
    context.stage(SERVER_STAGE)
    context.stage("matsim.scenario.network.pt")
    context.stage("sumo.macro_scenario.border")
    context.config("sumo_matsim_crs", "EPSG:2056")


@monitored
def execute(context):

    # Read the same network used by Java, reusing MATSim's binary reader cache.

    command = context.stage(SERVER_STAGE)["command"]
    path = command[command.index("--network-path") + 1]
    network = read_network(path, context=context)
    links = network.links
    road = (
        links["modes"]
        .str.split(",")
        .map(lambda values: bool({"car", "truck"}.intersection(values)))
    )
    network.links = links[road].copy()
    source_geometry = context.stage("matsim.scenario.network.pt").get("geometry")
    if source_geometry and Path(source_geometry).is_file():
        attach_link_geometries(network, source_geometry)
        shapes = network.links[GEOMETRY_COLUMN].to_numpy()
    else:
        shapes = network.as_geo().geometry.to_numpy()

    # Preserve directed detailed shapes and normalize them to the Swiss boundary CRS.

    crs = network.network_attrs.get("coordinateReferenceSystem") or context.config(
        "sumo_matsim_crs"
    )
    transformer = Transformer.from_crs(crs, 2056, always_xy=True)
    shapes = shapely.transform(shapes, transformer.transform, interleaved=False)
    links = network.links
    table = pl.DataFrame(
        {
            "link_id": links["link_id"].astype(str).to_numpy(),
            "from_node": links["from_node"].astype(str).to_numpy(),
            "to_node": links["to_node"].astype(str).to_numpy(),
            "length": links["length"].to_numpy(),
            "freespeed": links["freespeed"].to_numpy(),
            "modes": links["modes"].str.split(",").to_list(),
            "geometry": pl.Series(shapely.to_wkb(shapes), dtype=pl.Binary),
        }
    )
    if table.filter(
        ~pl.col("length").is_finite()
        | ~pl.col("freespeed").is_finite()
        | (pl.col("length") <= 0)
        | (pl.col("freespeed") <= 0)
    ).height:
        raise ValueError("Extended road network contains invalid lengths or speeds")

    # Index the Swiss boundary once per directed link, including exact crossing fractions.

    import json

    from sumo.functions.cutter.demand_tools import BoundaryIndex

    boundary = BoundaryIndex(context.stage("sumo.macro_scenario.border"))
    inside = shapely.covers(boundary.region, shapes)
    touches = shapely.intersects(boundary.region, shapes)
    intervals = [None] * len(shapes)
    import numpy as np

    for index in np.flatnonzero(touches & ~inside):
        intervals[index] = json.dumps(boundary.intervals(shapes[index]))
    table = table.with_columns(
        pl.Series("swiss_inside", inside),
        pl.Series("swiss_touches", touches),
        pl.Series("swiss_intervals", intervals, dtype=pl.String),
    )

    # Return only the Arrow path and routing network identity.

    output = Path(context.path()).resolve() / "road_links.arrow"
    table.write_ipc(output, compression="uncompressed")
    return {"links": str(output), "network": str(path), "crs": "EPSG:2056"}


def validate(context):
    return source_digest("sumo.functions.routing.network")
