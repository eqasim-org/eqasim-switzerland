"""Clip extended MATSim road itineraries and match directed SUMO endpoints."""

from functools import lru_cache
import json
import math
import logging
import time

import numpy as np
import polars as pl
import shapely
from lxml import etree
from pyproj import Transformer
from shapely.affinity import translate
from shapely.geometry import Point, box
from shapely.ops import transform

from sumo.cutter.demand_tools import BoundaryIndex
from sumo.scenario.network.tools import read_network, lane_allows
from sumo.scenario.network.search import candidate_windows, distance_limit
from sumo.scenario.population.vehicle_tools import trip_record
from sumo.scenario.tools import trip_element
from sumo.tables import read_table


logger = logging.getLogger("synpp")


class RoutedBorderClipper:
    def __init__(self, border, links, routes, sumo_network, distance=None, types=None, link_ids=None, paths=None):
        self.geometry = BoundaryIndex(border)
        self.region = self.geometry.region
        frame = read_table(links)
        if link_ids is not None:
            frame = frame.filter(pl.col("link_id").is_in(list(link_ids)))
        self.shapes = shapely.from_wkb(frame["geometry"].to_numpy())
        self.lookup = {link: index for index, link in enumerate(frame["link_id"])}
        self.free_times = (frame["length"] / frame["freespeed"]).to_numpy()
        self.from_nodes, self.to_nodes = frame["from_node"].to_numpy(), frame["to_node"].to_numpy()
        self.permissions = np.asarray([("car" in modes, "truck" in modes) for modes in frame["modes"]])
        indexed = {"swiss_inside", "swiss_touches", "swiss_intervals"}.issubset(frame.columns)
        self.inside = frame["swiss_inside"].to_numpy() if indexed else shapely.covers(self.region, self.shapes)
        self.touches = frame["swiss_touches"].to_numpy() if indexed else shapely.intersects(self.region, self.shapes)
        self.border_intervals = frame["swiss_intervals"].to_list() if indexed else None
        self.paths = read_table(routes) if paths is None else paths
        if self.paths["id"].n_unique() != self.paths.height:
            raise ValueError("Duplicate completed road routes")
        self.matcher = SumoEndpointMatcher(sumo_network, self.region, distance)
        self.types = types or {}
        self.report = {"method": "extended mode-choice road paths", "input": 0, "output": 0,
                       "outside": 0, "clipped": 0, "multiple_visits": 0,
                       "full_link_endpoint_fallbacks": 0,
                       "timing": "routed total time distributed over links in proportion to free-flow costs"}

    @lru_cache(maxsize=10000)
    def path_indices(self, payload, road_mode):
        try:
            itinerary = json.loads(payload)
            parts = [part for part in itinerary["parts"] if part.get("kind") == "leg"]
            if itinerary.get("version") != 1 or len(parts) != 1 or parts[0].get("mode") != road_mode:
                raise ValueError("Expected one car/truck network leg")
            route = parts[0]["route"]
            if route.get("type") != "links":
                raise ValueError("Expected an explicit link route")
            links = route["links"]
            if not isinstance(links, list) or any(not isinstance(link, str) for link in links):
                raise ValueError("Expected a list of link identifiers")
            links = list(links)
            start, end = route.get("startLinkId"), route.get("endLinkId")
            if start and (not links or links[0] != start):
                links.insert(0, start)
            if end and (not links or links[-1] != end):
                links.append(end)
            if not links:
                raise ValueError("Empty road path")
            indices = np.asarray([self.lookup[link] for link in links], dtype=np.int64)
            if not self.permissions[indices, 1 if road_mode == "truck" else 0].all():
                raise ValueError(f"Road route contains links forbidden to {road_mode}")
            if not (self.to_nodes[indices[:-1]] == self.from_nodes[indices[1:]]).all():
                raise ValueError("Disconnected or reversed directed road path")
            return indices
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid or mismatched extended-network itinerary: {error}") from error

    @lru_cache(maxsize=None)
    def link_intervals(self, index):
        if self.border_intervals is not None:
            if self.inside[index]:
                return ((0., 1.),)
            if not self.touches[index]:
                return ()
            return tuple(map(tuple, json.loads(self.border_intervals[index])))
        return tuple(self.geometry.intervals(self.shapes[index]))

    def __call__(self, frame, defer_matching=False):
        if defer_matching:
            frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("_sumo_clip_visit"))
        joined = (frame if "route" in frame.columns else
                  frame.join(self.paths, on="id", how="left", validate="1:1", maintain_order="left"))
        if joined["route"].null_count():
            raise ValueError("SUMO demand is missing completed extended-network road paths")
        self.report["input"] += frame.height
        unchanged, changed = [], []
        started = last_log = time.monotonic()
        for index, row in enumerate(joined.iter_rows(named=True)):
            if index % 1000 == 0 and time.monotonic() - last_log >= 30:
                logger.info("SUMO Swiss clipping: %d/%d trips checked, %d retained visits, %.1f seconds",
                            index, frame.height, len(unchanged) + len(changed), time.monotonic() - started)
                last_log = time.monotonic()
            road_mode = "truck" if row["mode"].removesuffix("_loop") in {"truck", "lcv", "delivery"} else "car"
            try:
                path = self.path_indices(row["route"], road_mode)
                if not self.touches[path].any():
                    self.report["outside"] += 1
                    continue
                origin = self.geometry.forward.transform(*map(float, row["origin"].split(",")))
                destination = self.geometry.forward.transform(*map(float, row["destination"].split(",")))
                # Common domestic paths require no per-link geometry processing.
                if (self.inside[path].all() and self.region.covers(Point(origin))
                        and self.region.covers(Point(destination))):
                    unchanged.append(index)
                    continue
                visits = self.visits(path, origin, destination, row["travel_time"])
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Cannot clip SUMO trip {row['id']} on its road path: {error}") from error
            if not visits:
                self.report["outside"] += 1
                continue
            self.report["clipped"] += 1
            self.report["multiple_visits"] += len(visits) > 1
            for number, visit in enumerate(visits, 1):
                result = {key: row[key] for key in frame.columns}
                result["id"] = row["id"] + (f"__swiss_{number}" if len(visits) > 1 else "")
                result["source_trip_id"] = row["source_trip_id"] or row["id"]
                result["depart"] = row["depart"] + visit["begin"]
                result["reference"] = max(0., visit["end"] - visit["begin"])
                if defer_matching:
                    result["_sumo_clip_visit"] = json.dumps({
                        key: np.asarray(visit[key]).tolist()
                        for key in ("first", "last", "first_direction", "last_direction")})
                else:
                    result = self.matched_trip(result, visit)
                changed.append(result)
        parts = [frame[unchanged]] if unchanged else []
        if changed:
            parts.append(pl.DataFrame(changed, schema=frame.schema))
        output = pl.concat(parts) if parts else frame.clear()
        self.report["output"] += output.height
        return output

    def visits(self, path, origin, destination, travel_time):
        if not math.isfinite(travel_time) or travel_time < 0:
            raise ValueError("Invalid routed travel time")
        lower, upper = np.zeros(len(path)), np.ones(len(path))
        lower[0] = self.shapes[path[0]].project(Point(origin), normalized=True)
        upper[-1] = self.shapes[path[-1]].project(Point(destination), normalized=True)
        if len(path) == 1 and upper[0] <= lower[0]:
            # Java's explicit same-link requests export one complete directed
            # link, regardless of the activity coordinates along that link.
            # Reversed/coincident projections cannot define a forward subpath;
            # retain the exported full-link traversal and its routed travel time.
            lower[0], upper[0] = 0., 1.
            self.report["full_link_endpoint_fallbacks"] += 1
        weights = self.free_times[path]
        total = float(np.sum(weights * (upper - lower)))
        if total <= 1e-12:
            return []
        scale, elapsed, visits = travel_time / total, 0., []
        last_position = None
        for sequence, (link, low, high, weight) in enumerate(zip(path, lower, upper, weights)):
            for a, b in self.link_intervals(int(link)):
                a, b = max(a, low), min(b, high)
                if b - a < 1e-10:
                    continue
                begin, end = elapsed + weight * (a - low) * scale, elapsed + weight * (b - low) * scale
                if visits and last_position is not None and abs(last_position - (sequence + a)) < 1e-9:
                    visits[-1].update(end=end, last_link=link, last_fraction=b)
                else:
                    visits.append({"begin": begin, "end": end, "first_link": link, "first_fraction": a,
                                   "last_link": link, "last_fraction": b})
                last_position = sequence + b
            elapsed += weight * (high - low) * scale
        for visit in visits:
            visit["first"], visit["first_direction"] = endpoint(self.shapes[visit["first_link"]], visit["first_fraction"])
            visit["last"], visit["last_direction"] = endpoint(self.shapes[visit["last_link"]], visit["last_fraction"])
        return visits

    def matched_trip(self, row, visit):
        if row["xml"] is None:
            record = trip_record(row)
            record["parameters"]["_sumo_loads_network"] = "true" if row["loads_network"] else "false"
            element = trip_element(record)
            etree.SubElement(element, "param", key="_sumo_population", value="true")
        else:
            element = etree.fromstring(row["xml"])
        if any(key in element.attrib for key in ("via", "viaXY", "viaLonLat")):
            raise ValueError("Border clipping cannot preserve unrouted via constraints")
        definition = self.types.get(element.get("type"))
        vehicle_class = (definition.get("vClass", "passenger") if definition is not None else
                         {"truck": "truck", "lcv": "delivery"}.get(row["mode"], "passenger"))
        first = self.matcher.match(visit["first"], visit["first_direction"], vehicle_class)
        last = self.matcher.match(visit["last"], visit["last_direction"], vehicle_class)
        for key in ("fromLonLat", "toLonLat", "fromXY", "toXY", "departLane", "arrivalLane"):
            element.attrib.pop(key, None)
        element.attrib.update({"id": row["id"], "depart": format(row["depart"], ".15g"),
            "from": first["edge"], "to": last["edge"], "departPos": format(first["position"], ".12g"),
            "arrivalPos": format(last["position"], ".12g"), "departLane": "best"})
        for param in list(element.findall("param")):
            if param.get("key") in {"source_travel_time_s", "source_trip_id"}:
                element.remove(param)
        for key, value in (("source_travel_time_s", row["reference"]), ("source_trip_id", row["source_trip_id"]),
                           ("sumo_border_timing", "routed_time_scaled_by_link_freeflow")):
            etree.SubElement(element, "param", key=key, value=str(value))
        row["origin"], row["destination"] = first["lonlat"], last["lonlat"]
        row["xml"] = etree.tostring(element)
        return row


class SumoEndpointMatcher:
    def __init__(self, source, region, distance):
        self.source, self.swiss_region, self.distance = source, region, distance_limit(distance)
        self.network = None
        self.cache = {}

    def initialize(self):
        started = time.monotonic()
        logger.info("Reading SUMO network for border endpoint matching")
        self.network = read_network(self.source())
        location = self.network["location"]
        projection = location.get("projParameter")
        if projection in {None, "!", "-"}:
            raise ValueError("Routed Swiss clipping requires a georeferenced SUMO network")
        self.forward = Transformer.from_crs(2056, projection, always_xy=True)
        self.reverse = Transformer.from_crs(projection, 4326, always_xy=True)
        self.offset = np.asarray([float(x) for x in location.get("netOffset", "0,0").split(",")])
        self.region = translate(transform(self.forward.transform, self.swiss_region), *self.offset)
        shapely.prepare(self.region)
        self.ids = list(self.network["edges"])
        self.shapes = np.asarray([self.network["edges"][key]["shape"] for key in self.ids], dtype=object)
        # Some imported lanes have positive simulated lengths but collapsed shapes.
        # They have no geographic heading and cannot be endpoint candidates.
        lengths = shapely.length(self.shapes)
        usable = np.isfinite(lengths) & (lengths > 1e-12)
        if not usable.all():
            logger.warning("SUMO endpoint matching: excluding %d edges with degenerate lane shapes",
                           int((~usable).sum()))
        self.ids = [key for key, keep in zip(self.ids, usable) if keep]
        self.shapes = self.shapes[usable]
        self.tree = shapely.STRtree(self.shapes)
        self.bounds = shapely.total_bounds(self.shapes) if len(self.shapes) else np.zeros(4)
        self.domestic = shapely.intersects(self.region, self.shapes)
        self.class_trees = {}
        logger.info("Indexed %d SUMO border-matching edges in %.1f seconds", len(self.ids), time.monotonic() - started)

    def match(self, xy, direction, vehicle_class):
        if self.network is None:
            self.initialize()
        key = (*np.round(xy, 3), *np.round(direction, 5), vehicle_class)
        if key in self.cache:
            return self.cache[key]
        point = Point(np.asarray(self.forward.transform(*xy)) + self.offset)
        ahead = np.asarray(self.forward.transform(*(np.asarray(xy) + np.asarray(direction)))) + self.offset
        wanted = ahead - np.asarray(point.coords[0])
        wanted /= np.linalg.norm(wanted)
        if vehicle_class not in self.class_trees:
            eligible = [index for index, key in enumerate(self.ids) if self.domestic[index]
                        and any(lane_allows(lane, vehicle_class) for lane in self.network["edges"][key]["lanes"])]
            indices = np.asarray(eligible, dtype=np.int64)
            self.class_trees[vehicle_class] = indices, shapely.STRtree(self.shapes[indices])
        indices, tree = self.class_trees[vehicle_class]
        if not len(indices):
            raise ValueError(f"No directed {vehicle_class} SUMO edge on the Swiss side: network has no usable permitted geometry")
        best = None
        for index in indices[tree.query_nearest(point, all_matches=True)]:
            candidate = self.evaluate(index, point, wanted, vehicle_class)
            if candidate is not None and (best is None or candidate[0] < best[0]):
                best = candidate
        for index in self.candidates(point, lambda: None if best is None else best[0][0]):
            candidate = self.evaluate(index, point, wanted, vehicle_class)
            if candidate is not None and (best is None or candidate[0] < best[0]):
                best = candidate
        if best is None:
            scope = "on the supplied network" if self.distance is None else f"within {self.distance:g} m"
            raise ValueError(f"No directed {vehicle_class} SUMO edge on the Swiss side {scope} at {tuple(xy)}")
        self.cache[key] = best[1]
        return best[1]

    def evaluate(self, index, point, wanted, vehicle_class):
        edge_id = self.ids[index]
        edge = self.network["edges"][edge_id]
        if not any(lane_allows(lane, vehicle_class) for lane in edge["lanes"]):
            return None
        line = edge["shape"]
        fraction = line.project(point, normalized=True)
        snap, heading = endpoint(line, fraction)
        if not self.region.covers(Point(snap)):
            inside = line.intersection(self.region)
            if inside.is_empty:
                return None
            parts = shapely.get_parts(inside)
            parts = parts[(shapely.get_type_id(parts) == 1) & ~shapely.is_empty(parts)
                          & (shapely.length(parts) > 1e-12)]
            if not len(parts):
                return None
            closest = min(parts, key=lambda part: part.distance(point))
            snap = np.asarray(closest.interpolate(closest.project(point)).coords[0])
            fraction = line.project(Point(snap), normalized=True)
            snap, heading = endpoint(line, fraction)
        alignment = float(np.dot(wanted, heading))
        distance = float(np.linalg.norm(snap - np.asarray(point.coords[0])))
        if not np.isfinite([alignment, distance, fraction]).all() or alignment <= 0 or (self.distance is not None and distance > self.distance):
            return None
        score = (distance + 20 * (1 - alignment), edge_id)
        lon, lat = self.reverse.transform(*(snap - self.offset))
        return score, {"edge": edge_id, "position": fraction * edge["length"],
                       "lonlat": f"{lon:.10f},{lat:.10f}"}

    def candidates(self, point, best_score):
        for fresh in candidate_windows(self.tree, point, self.bounds, best_score, self.distance):
            yield from fresh


def endpoint(line, fraction):
    fraction = min(1., max(0., float(fraction)))
    point = np.asarray(line.interpolate(fraction, normalized=True).coords[0])[:2]
    step = min(.01, 1 / max(line.length, 1))
    before = np.asarray(line.interpolate(max(0, fraction-step), normalized=True).coords[0])[:2]
    after = np.asarray(line.interpolate(min(1, fraction+step), normalized=True).coords[0])[:2]
    direction = after - before
    norm = np.linalg.norm(direction)
    if norm <= 1e-12:
        # A tight reversal can cancel a centred tangent on a valid road shape.
        # Use the outgoing direction, or the incoming direction at the end.
        direction = after - point
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            direction = point - before
            norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise ValueError("Degenerate directed road geometry")
    return point, direction / norm
