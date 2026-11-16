"""Prepare directed routing and foreign road access from convert_osm's pickle."""

import hashlib
import logging
from pathlib import Path
import re
from types import SimpleNamespace

import geopandas as gpd
import igraph as ig
import numpy as np
import pandas as pd
from shapely import covers, intersects, prepare
from shapely.geometry import Point

from matsim.scenario.network.utils.link_geometry import attach_link_geometries, get_network_geometry
from matsim.scenario.network.utils.border_crossings import crossing_links
from synthesis.freight.npvm.crossings import ROAD_HIERARCHY, highway_type, NETWORK_STAGE

logger = logging.getLogger("synpp")


def configure(context):
    context.stage(NETWORK_STAGE)
    context.stage("data.spatial.swiss_border")


def execute(context):

    # Read the converted Python network directly, never reparsing its XML.

    context.stage(NETWORK_STAGE)
    directory = Path(context.path(NETWORK_STAGE))
    network = pd.read_pickle(directory / "network.pkl")
    country = context.stage("data.spatial.swiss_border").to_crs(2056).union_all()

    # Preserve directed curves while dropping unrelated attributes from the working copy.

    network = prepare_network(network, directory / "detailed_network.csv")
    return build_access_network(network, country)


def validate(context):
    from synthesis.freight.npvm import crossings as freight_crossings
    from matsim.scenario.network.utils import border_crossings, link_geometry
    code = b"".join(Path(module.__file__).read_bytes() for module in
                    (freight_crossings, border_crossings, link_geometry))
    directory = Path(context.path(NETWORK_STAGE))
    inputs = []
    for name in ("network.pkl", "detailed_network.csv"):
        path = directory / name
        stat = path.stat() if path.is_file() else None
        inputs.append([name, stat.st_size, stat.st_mtime_ns] if stat else [name, None])
    return {"helpers": hashlib.sha256(code).hexdigest(), "inputs": inputs}


def prepare_network(network, geometry_path):
    modes = network.links.modes.fillna("").map(lambda value: set(re.split(r"[,;\s]+", str(value))))
    mode = "truck" if modes.map(lambda values: "truck" in values).any() else "car"
    links = network.links.loc[modes.map(lambda values: mode in values)].copy()
    links["highway"] = links.attributes.map(lambda values: highway_type(
        values.get("osm:way:highway") if isinstance(values, dict) else None))
    columns = ["link_id", "from_node", "to_node", "modes", "highway", "length", "freespeed"]
    links = links[columns].reset_index(drop=True)
    used = set(links.from_node.astype(str)) | set(links.to_node.astype(str))
    nodes = network.nodes.loc[network.nodes.node_id.astype(str).isin(used), ["node_id", "x", "y"]].copy()
    result = SimpleNamespace(links=links, nodes=nodes, network_attrs=dict(network.network_attrs))
    attach_link_geometries(result, geometry_path)
    return result


def build_access_network(network, country):
    """Use all permitted roads for routing, major wholly foreign roads for access."""
    links = network.links.copy().reset_index(drop=True)
    nodes = network.nodes.reset_index(drop=True).copy()
    nodes["node_id"] = nodes.node_id.astype(str)
    lookup = pd.Series(np.arange(len(nodes)), index=nodes.node_id)
    links["from_index"] = links.from_node.astype(str).map(lookup).astype(int)
    links["to_index"] = links.to_node.astype(str).map(lookup).astype(int)
    links["time"] = links.length.to_numpy(dtype=float) / links.freespeed.to_numpy(dtype=float)
    if not np.isfinite(links.time).all() or not links.time.gt(0).all():
        raise ValueError("Routing requires finite positive link travel times")
    shapes = get_network_geometry(network)
    links = links.drop(columns="_detailed_geometry", errors="ignore")
    crossings = crossing_links(links, shapes, country)
    prepare(country)
    inside = covers(country, gpd.points_from_xy(nodes.x, nodes.y))
    source, target = links.from_index.to_numpy(), links.to_index.to_numpy()
    internal = inside[source] & inside[target]
    crossing_ids = set(crossings.link_id.astype(str))
    crossing_mask = links.link_id.astype(str).isin(crossing_ids).to_numpy()
    uncertain = np.flatnonzero(internal & crossing_mask)
    internal[uncertain] = covers(country, shapes.iloc[uncertain].array)
    exterior = ~inside[source] & ~inside[target] & ~crossing_mask
    allowed_crossing = links.highway.map(ROAD_HIERARCHY).lt(max(ROAD_HIERARCHY.values())).to_numpy()
    permitted = ~crossing_mask | allowed_crossing
    graph = routing_graph(links, permitted, len(nodes))
    crossings = crossings[crossings.link_id.isin(set(links.loc[allowed_crossing, "link_id"].astype(str)))].copy()
    inside_graph = routing_graph(links, internal, len(nodes))
    outside_graph = routing_graph(links, exterior, len(nodes))

    # A foreign access link stays entirely outside Switzerland, so its partial traversal is unambiguous.

    major = links.highway.map(ROAD_HIERARCHY).le(2).to_numpy()
    eligible = np.flatnonzero(major)
    outside = ~intersects(country, shapes.iloc[eligible].array)
    chosen = eligible[outside]
    roads = gpd.GeoDataFrame(links.iloc[chosen].copy(), geometry=shapes.iloc[chosen].array, crs=2056)
    roads["hierarchy"] = roads.highway.map(ROAD_HIERARCHY)
    if roads.empty:
        raise ValueError("No eligible foreign motorway/trunk/primary/secondary access roads")
    logger.info("Freight routing: %d nodes, %d permitted links, %d foreign access links",
                len(nodes), len(links), len(roads))
    return {"graph": graph, "inside_graph": inside_graph, "outside_graph": outside_graph, "nodes": nodes, "links": links.drop(columns="_detailed_geometry", errors="ignore"),
            "crossings": crossings, "roads": roads, "inside": inside, "country": country}


def routing_graph(links, keep, node_count):
    """Preserve original link indices in each directed routing graph."""
    part = links.loc[keep]
    graph = ig.Graph(n=node_count, edges=list(zip(part.from_index, part.to_index)), directed=True)
    graph.es["time"] = part.time.tolist()
    graph.es["link_index"] = np.flatnonzero(keep).tolist()
    return graph


def sample_access_points(zones, roads, rng, count=3):
    """Sample major-road positions in a zone, or use its three nearest road projections."""
    ordered = roads.sort_values(["hierarchy", "link_id"], kind="stable")
    physical = pd.MultiIndex.from_arrays([np.minimum(ordered.from_index, ordered.to_index),
                                         np.maximum(ordered.from_index, ordered.to_index)])
    fallback_roads = ordered.loc[~physical.duplicated()]
    records = []
    for i, zone in enumerate(zones.itertuples()):
        if i % 50 == 0:
            logger.info("Foreign road access: %d/%d zones", i + 1, len(zones))
        positions = (roads.sindex.query(zone.geometry, predicate="intersects")
                     if zone.geometry.geom_type in {"Polygon", "MultiPolygon"} else [])
        candidates = roads.iloc[positions].copy()
        if not candidates.empty and zone.geometry.geom_type in {"Polygon", "MultiPolygon"}:
            candidates.geometry = candidates.geometry.intersection(zone.geometry)
            candidates = candidates[candidates.geometry.length.gt(0)].copy()
        if not candidates.empty:
            weights = 2.0 ** (-candidates.hierarchy.to_numpy(dtype=float))
            candidates = candidates.iloc[rng.choice(len(candidates), count,
                replace=len(candidates) < count, p=weights / weights.sum())]
            method = "sampled_road_in_zone"
        else:
            # Exact line-to-region distances, rather than centroid or node distances.
            # An indexed distance search still gives the exact nearest three, without scanning every road.
            _, nearest = fallback_roads.sindex.nearest(zone.geometry, return_distance=True, return_all=False)
            padding = 100.0
            positions = []
            while len(positions) < min(count, len(fallback_roads)):
                positions = fallback_roads.sindex.query(zone.geometry, predicate="dwithin",
                                                       distance=float(nearest[0]) + padding)
                padding *= 2
            candidates = fallback_roads.iloc[positions].copy()
            candidates["distance_to_zone_m"] = candidates.geometry.distance(zone.geometry)
            candidates = candidates.sort_values(["distance_to_zone_m", "hierarchy", "link_id"],
                                                kind="stable").head(count)
            method = "nearest_three_foreign_roads"
        for row in candidates.itertuples():
            original = roads.loc[row.Index].geometry
            if method == "sampled_road_in_zone":
                point = row.geometry.interpolate(rng.random() * row.geometry.length)
                distance = 0.0
            else:
                from shapely.ops import nearest_points
                point = nearest_points(original, zone.geometry)[0]
                distance = row.distance_to_zone_m
            fraction = original.project(point) / original.length
            records.append({"zone_id": zone.zone_id, "link_id": str(row.link_id), "highway": row.highway,
                "hierarchy": row.hierarchy, "x": point.x, "y": point.y,
                "from_index": row.from_index, "to_index": row.to_index, "fraction": fraction,
                "link_time": row.time, "distance_to_zone_m": distance, "selection_method": method,
                "weight": 2.0 ** (-row.hierarchy) / (1 + distance / 1000)})
    return pd.DataFrame(records)
