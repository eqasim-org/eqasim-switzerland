"""Mode-specific turn connectivity and reachable SUMO trip endpoints."""

import copy
import json
import logging
import time
from pathlib import Path

import igraph as ig
import numpy as np
import polars as pl
import shapely
from lxml import etree
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import Point
from shapely.ops import transform

from sumo.scenario.network.tools import lane_allows, position, read_network
from sumo.scenario.network.search import candidate_windows, distance_limit, SEARCH_STEP
from sumo.scenario.population.tools import xml_stream
from sumo.scenario.tools import elements

logger = logging.getLogger("synpp")


def build_index(network, directory, classes):
    """Cache geometry and role masks; never remove roads from the network."""
    data = read_network(network)
    ids = sorted(data["edges"])
    lookup = {edge: index for index, edge in enumerate(ids)}
    edges = [data["edges"][edge] for edge in ids]
    shapes = np.asarray([edge["shape"] for edge in edges], dtype=object)
    geometric_lengths = shapely.length(shapes)
    geometric = np.isfinite(geometric_lengths) & (geometric_lengths > 1e-12)
    classes = sorted(classes)
    lanes = [[sum((1 << index) for index, name in enumerate(classes) if lane_allows(lane, name))
              for lane in edge["lanes"]] for edge in edges]
    pairs, permissions = [], []
    for connection in data["connections"]:
        first, last = lookup[connection["from"]], lookup[connection["to"]]
        permission = lanes[first][int(connection["fromLane"])] & lanes[last][int(connection["toLane"])]
        if permission:
            pairs.append((first, last))
            permissions.append(permission)
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    permissions = np.asarray(permissions, dtype=np.uint32)
    table = {"edge_id": ids, "geometry": pl.Series(shapely.to_wkb(shapes), dtype=pl.Binary),
             "length": [edge["length"] for edge in edges]}
    report = {}
    for index, name in enumerate(classes):
        allowed = np.asarray([any(value & (1 << index) for value in values) for values in lanes])
        graph = ig.Graph(n=len(ids), edges=pairs[(permissions & (1 << index)) != 0], directed=True)
        sources, targets, detail = endpoint_masks(graph, allowed)
        table["from_" + name], table["to_" + name] = sources & geometric, targets & geometric
        report[name] = {**detail, "origins": int((sources & geometric).sum()),
                        "destinations": int((targets & geometric).sum())}
        logger.info("SUMO endpoint connectivity (%s): %s", name, report[name])
    directory = Path(directory)
    target = directory / "routing_edges.arrow"
    pl.DataFrame(table).write_ipc(target, compression="uncompressed")
    return {"edges": str(target), "network": str(network), "location": data["location"], "connectivity": report}


def endpoint_masks(graph, allowed):
    """Origins must reach the main directed core; destinations must be reachable from it.

    Keep one-way entry/exit branches too. Small acyclic test/extract networks
    have no directed core: use their main weak component and endpoint degrees.
    """
    sources, targets = np.zeros(graph.vcount(), dtype=bool), np.zeros(graph.vcount(), dtype=bool)
    eligible = np.flatnonzero(allowed)
    if not len(eligible):
        return sources, targets, {"core_edges": 0}
    strong = graph.connected_components(mode="strong")
    sizes = np.asarray(strong.sizes())
    membership = np.asarray(strong.membership)
    core = max(np.unique(membership[eligible]), key=lambda component: (sizes[component], -component))
    representative = int(np.flatnonzero(membership == core)[0])
    if sizes[core] > 1:
        sources[graph.subcomponent(representative, mode="in")] = True
        targets[graph.subcomponent(representative, mode="out")] = True
    else:
        weak = graph.connected_components(mode="weak")
        sizes_weak = np.asarray(weak.sizes())
        memberships = np.asarray(weak.membership)
        main = max(np.unique(memberships[eligible]), key=lambda component: (sizes_weak[component], -component))
        retained = memberships == main
        sources = retained & ((np.asarray(graph.outdegree()) > 0) | (sizes_weak[main] == 1))
        targets = retained & ((np.asarray(graph.indegree()) > 0) | (sizes_weak[main] == 1))
    return sources & allowed, targets & allowed, {"core_edges": int(sizes[core])}


class EndpointIndex:
    def __init__(self, manifest, radius, border=None):
        self.frame = pl.read_ipc(manifest["edges"])
        self.ids = self.frame["edge_id"].to_list()
        self.lookup = {edge: index for index, edge in enumerate(self.ids)}
        self.shapes = shapely.from_wkb(self.frame["geometry"].to_numpy())
        self.lengths = self.frame["length"].to_numpy()
        self.radius = distance_limit(radius)
        self.offset = np.asarray([float(value) for value in manifest["location"].get("netOffset", "0,0").split(",")])
        projection = manifest["location"].get("projParameter")
        self.forward = None if projection in {None, "!", "-"} else Transformer.from_crs(4326, projection, always_xy=True)
        self.region = None
        if border is not None:
            if self.forward is None:
                raise ValueError("Swiss endpoint matching requires a georeferenced SUMO network")
            region = shapely.from_wkb(Path(border["geometry"]).read_bytes())
            transformer = Transformer.from_crs(border["crs"], projection, always_xy=True)
            self.region = translate(transform(transformer.transform, region), *self.offset)
            shapely.prepare(self.region)
        self.trees, self.cache = {}, {}

    def match(self, attributes, side, vehicle_class):
        column = side + "_" + vehicle_class
        if column not in self.frame.columns:
            raise ValueError(f"No routing connectivity index for vehicle class {vehicle_class}")
        explicit = attributes.get(side)
        direction = None
        position_key = "departPos" if side == "from" else "arrivalPos"
        if explicit is not None:
            if explicit not in self.lookup:
                raise ValueError(f"Unknown SUMO endpoint edge {explicit}")
            index = self.lookup[explicit]
            if self.frame[column][index]:
                return None
            length = self.lengths[index]
            distance = position(attributes.get(position_key, "0" if side == "from" else "max"), length)
            fraction = distance / length
            line = self.shapes[index]
            point = line.interpolate(fraction, normalized=True)
            before = np.asarray(line.interpolate(max(0., fraction-.01), normalized=True).coords[0])[:2]
            after = np.asarray(line.interpolate(min(1., fraction+.01), normalized=True).coords[0])[:2]
            vector = after-before
            norm = np.linalg.norm(vector)
            if norm > 1e-12:
                direction = vector / norm
        elif side + "LonLat" in attributes:
            if self.forward is None:
                raise ValueError("Geographic trips require a georeferenced SUMO network")
            lonlat = [float(value) for value in attributes[side + "LonLat"].split(",")]
            point = Point(np.asarray(self.forward.transform(*lonlat, errcheck=True)) + self.offset)
        elif side + "XY" in attributes:
            point = Point([float(value) for value in attributes[side + "XY"].split(",")])
        else:
            # Native junction/TAZ inputs do not identify a single endpoint edge.
            return None
        key = (column, *point.coords[0], *(direction if direction is not None else ()))
        if key in self.cache:
            return self.cache[key]
        if column not in self.trees:
            indices = np.flatnonzero(self.frame[column].to_numpy())
            shapes = self.shapes[indices]
            self.trees[column] = indices, shapely.STRtree(shapes), shapely.total_bounds(shapes) if len(shapes) else np.zeros(4)
        indices, tree, bounds = self.trees[column]
        if not len(indices):
            raise ValueError(f"Network has no reachable {vehicle_class} {side} edges")
        nearest, distances = tree.query_nearest(point, all_matches=True, return_distance=True)
        best = self.choose(point, indices[nearest], direction)
        # For ordinary points the nearest permitted road is an exact solution:
        # skip empty 500 m rings without scanning a large neighbourhood.
        exact_nearest = (direction is None and best is not None
                         and abs(best["distance_m"] - float(distances.min())) < 1e-8)
        if not exact_nearest:
            for fresh in candidate_windows(tree, point, bounds,
                    lambda: None if best is None else best["score_m"], self.radius, seen=nearest):
                candidate = self.choose(point, indices[fresh], direction)
                if candidate is not None and (best is None or (candidate["score_m"], candidate["edge"])
                                             < (best["score_m"], best["edge"])):
                    best = candidate
        if best is None:
            scope = "on the supplied network" if self.radius is None else f"within {self.radius:g} m"
            raise ValueError(f"No reachable {vehicle_class} {side} edge {scope} satisfying Swiss-side and direction constraints at {point.wkt}")
        best["search_radius_m"] = max(SEARCH_STEP, float(np.ceil(best["distance_m"]/SEARCH_STEP)*SEARCH_STEP))
        if self.radius is not None:
            best["search_radius_m"] = min(self.radius, best["search_radius_m"])
        self.cache[key] = best
        return best

    def choose(self, point, candidates, direction=None):
        best = None
        for index in candidates:
            line = self.shapes[index]
            fraction = line.project(point, normalized=True)
            snap = line.interpolate(fraction, normalized=True)
            if self.region is not None and not self.region.covers(snap):
                parts = shapely.get_parts(line.intersection(self.region))
                parts = parts[shapely.get_type_id(parts) == 1]
                if not len(parts):
                    continue
                part = min(parts, key=lambda piece: piece.distance(point))
                snap = part.interpolate(part.project(point))
                fraction = line.project(snap, normalized=True)
            distance = point.distance(snap)
            alignment = 1.
            if direction is not None:
                before = np.asarray(line.interpolate(max(0., fraction-.01), normalized=True).coords[0])[:2]
                after = np.asarray(line.interpolate(min(1., fraction+.01), normalized=True).coords[0])[:2]
                heading = after-before
                norm = np.linalg.norm(heading)
                alignment = np.dot(direction, heading) / norm if norm > 1e-12 else 0.
                if alignment <= 0:
                    continue
            score = (distance + 20 * (1-alignment), self.ids[index])
            if (self.radius is None or distance <= self.radius) and (best is None or score < best[0]):
                best = score, {"edge": self.ids[index], "position": fraction * self.lengths[index], "distance_m": distance, "score_m": score[0]}
        return None if best is None else best[1]


def map_trips(source, target, manifest, radius, border=None):
    """Map all geographic inputs and repair unreachable explicit endpoints, preserving demand."""
    started = time.monotonic()
    logger.info("Loading cached SUMO road geometry for endpoint mapping")
    index = EndpointIndex(manifest, radius, border)
    logger.info("Loaded SUMO endpoint index in %.1f seconds; maximum distance: %s",
                time.monotonic() - started, "unlimited" if radius is None else radius)
    types = {}
    report = {"trips": 0, "mapped_endpoints": 0, "repaired_endpoints": 0, "maximum_mapping_distance_m": 0., "expanded_endpoints": 0, "maximum_search_radius_m": 0., "examples": []}
    with xml_stream(target) as stream, etree.xmlfile(stream, encoding="utf-8") as writer:
        writer.write_declaration()
        with writer.element("routes"):
            for element in elements(source, ("vType", "trip")):
                element = copy.deepcopy(element)
                if element.tag == "vType":
                    types[element.get("id")] = element.get("vClass", "passenger")
                else:
                    vehicle_class = types.get(element.get("type"), "passenger")
                    for side, pos in (("from", "departPos"), ("to", "arrivalPos")):
                        previous = element.get(side)
                        try:
                            matched = index.match(element.attrib, side, vehicle_class)
                        except ValueError as error:
                            raise ValueError(f"Cannot map SUMO trip {element.get('id')}: {error}") from error
                        if matched is None:
                            continue
                        if previous is not None:
                            report["repaired_endpoints"] += 1
                            if len(report["examples"]) < 20:
                                report["examples"].append({"trip": element.get("id"), "side": side,
                                    "old_edge": previous, "new_edge": matched["edge"], "distance_m": matched["distance_m"]})
                        report["mapped_endpoints"] += 1
                        report["expanded_endpoints"] += matched["distance_m"] > SEARCH_STEP
                        report["maximum_search_radius_m"] = max(report["maximum_search_radius_m"], matched["search_radius_m"])
                        report["maximum_mapping_distance_m"] = max(report["maximum_mapping_distance_m"], matched["distance_m"])
                        for suffix in ("LonLat", "XY"):
                            element.attrib.pop(side + suffix, None)
                        element.set(side, matched["edge"])
                        element.set(pos, format(matched["position"], ".12g"))
                    report["trips"] += 1
                    if report["trips"] % 10000 == 0:
                        logger.info("SUMO endpoint mapping: %d trips, %d expanded endpoints, %.1f seconds",
                                    report["trips"], report["expanded_endpoints"], time.monotonic() - started)
                writer.write(element)
    Path(target).with_name("endpoint_mapping.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("SUMO reachable endpoint mapping: %s", report)
    return report
