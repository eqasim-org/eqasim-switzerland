import csv
import gzip
import heapq
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict

import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

CRS = "EPSG:2056"
ARTIFICIAL_MODES = {"artificial", "stopFacilityLink"}

OUTSIDE_PENALTY    = 20.0    # cost multiplier for links outside the official corridor
NO_MODE_PENALTY    = 1.3     # cost multiplier for links that do not allow the route's mode yet
OPPOSITE_DIRECTION_COST = 30.0   # cost of a stop on a link that runs against the direction of travel
U_TURN_COST        = 100.0    # cost of turning back onto the reverse of the link just driven
STOP_DISTANCE_COST = 0.1     # cost per metre between a stop and the link it is put on
OUTSIDE_STOP_COST  = 200.0   # cost of putting a stop on a link outside the corridor
MAX_PATH_COST      = 6000.0
BOUNDS_MARGIN      = 1000.0
 
PLATFORM_SWITCH_DISTANCE    = 100.0  # extra "distance" for using another platform of the same station
STATION_SWITCH_MIN_DISTANCE = 50.0   # only switch platform if the stop is this far outside the corridor

NEW_LINK_RATIO         = 1.5     # a leg gets new links if its path is longer than this x the official distance ...
NEW_LINK_SLACK         = 100.0   # ... plus this many metres
NEW_LINK_SNAP_DISTANCE = 60.0    # stops further than this from the official route get no new links
NEW_LINK_MAX_LENGTH    = 3000.0
NEW_LINK_SPEED         = 11.1    # m/s
NEW_LINK_MIN_SPACING   = 3.0     # metres between the nodes of a new link
NEW_LINK_SIMPLIFY      = 2.0     # metres: tolerance when simplifying the official line into new links


def read_line_agencies(routes_path):
    """GTFS route_id -> agency_id. The line ids of the mapped schedule are the GTFS route ids."""
    with open(routes_path, newline = "", encoding = "utf-8-sig") as f:
        return {row["route_id"]: row.get("agency_id", "") for row in csv.DictReader(f)}


def normalize_name(name):
    """Casefolded name without accents, punctuation or repeated spaces, so that e.g.
    "Vernier, Lignon-Tours" (shapefile) matches "Vernier, Lignon Tours" (GTFS)."""
    name = unicodedata.normalize("NFKD", str(name or ""))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", name.casefold()).split())


def normalize_line(name):
    name = str(name).strip().upper()
    return name.lstrip("0") or name


# ---------------------------------------------------------------------------
# Official line routes
# ---------------------------------------------------------------------------

class Corridors:
    def __init__(self, shapefile_path, buffer_m):
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(CRS) if gdf.crs is not None else gdf.set_crs(CRS)

        self.features = defaultdict(list)  # normalized line -> [(directions, buffered geometry)]
        self.centerlines = defaultdict(list)  # normalized line -> [unbuffered geometry]
        for line, direction, geometry in zip(gdf["LIGNE"], gdf["DIRECTION"], gdf.geometry):
            if geometry is None or geometry.is_empty:
                continue
            geometry = shapely.force_2d(geometry)
            directions = {normalize_name(d) for d in str(direction).split("/")}
            self.features[normalize_line(line)].append((directions, shapely.buffer(geometry, buffer_m)))
            self.centerlines[normalize_line(line)].append(geometry)

        minx, miny, maxx, maxy = gdf.total_bounds
        self.bounds = (minx - buffer_m - BOUNDS_MARGIN, miny - buffer_m - BOUNDS_MARGIN,
                       maxx + buffer_m + BOUNDS_MARGIN, maxy + buffer_m + BOUNDS_MARGIN)
        self._geometries = {}
        self._graphs = {}


    def centerline_graph(self, key):
        """Graph of the official route(s) selected by a lookup key (None if not a single route)."""
        if key not in self._graphs:
            line, selected = key
            self._graphs[key] = CenterlineGraph(self.centerlines[line][selected[0]]) if len(selected) == 1 else None
        return self._graphs[key]


    def lookup(self, line_name, destination_name):
        """Returns (key, geometry, matched_direction) or None if the line has no official route.
        If no route runs towards the destination, all routes of the line are used."""
        features = self.features.get(normalize_line(line_name))
        if not features:
            return None

        destination = normalize_name(destination_name)
        selected = [i for i, (directions, _) in enumerate(features) if destination in directions]
        matched = bool(selected)
        selected = selected or list(range(len(features)))

        key = (normalize_line(line_name), tuple(selected))
        if key not in self._geometries:
            geometry = shapely.union_all([features[i][1] for i in selected])
            shapely.prepare(geometry)
            self._geometries[key] = geometry
        return key, self._geometries[key], matched


class CenterlineGraph:
    """The vertices of an official route as a graph, so that the route between two points can be
    followed even when the shapefile geometry is split in several parts."""

    JOIN_TOLERANCE = 20.0

    def __init__(self, geometry):
        parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
        index, self.coords = {}, []
        self.adjacency = defaultdict(list)

        def node(c):
            key = (round(c[0], 1), round(c[1], 1))
            if key not in index:
                index[key] = len(self.coords)
                self.coords.append(key)
            return index[key]

        endpoints = []
        for part in parts:
            ids = [node(c) for c in part.coords]
            for a, b in zip(ids, ids[1:]):
                if a != b:
                    w = math.dist(self.coords[a], self.coords[b])
                    self.adjacency[a].append((b, w))
                    self.adjacency[b].append((a, w))
            endpoints += [ids[0], ids[-1]]

        points = shapely.points(self.coords)
        self._tree = STRtree(points)
        for e in set(endpoints):
            for other in self._tree.query(Point(self.coords[e]).buffer(self.JOIN_TOLERANCE)):
                other = int(other)
                if other != e and all(other != n for n, _ in self.adjacency[e]):
                    w = math.dist(self.coords[e], self.coords[other])
                    if w <= self.JOIN_TOLERANCE:
                        self.adjacency[e].append((other, w))
                        self.adjacency[other].append((e, w))


    def snap(self, x, y, max_distance):
        i = int(self._tree.nearest(Point(x, y)))
        return i if math.dist(self.coords[i], (x, y)) <= max_distance else None


    def path(self, source, target):
        """(coordinates, length) of the shortest way along the route, or None."""
        best, previous, queue = {source: 0.0}, {}, [(0.0, source)]
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > best.get(i, math.inf):
                continue
            if i == target:
                nodes = [i]
                while nodes[-1] != source:
                    nodes.append(previous[nodes[-1]])
                return [self.coords[n] for n in reversed(nodes)], cost
            for j, w in self.adjacency[i]:
                if cost + w < best.get(j, math.inf):
                    best[j] = cost + w
                    previous[j] = i
                    heapq.heappush(queue, (cost + w, j))
        return None


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class Network:
    """The part of a MATSim network inside `bounds` (nodes are written before links)."""

    def __init__(self, network_path, bounds, routing_modes, fallback_modes = frozenset()):
        minx, miny, maxx, maxy = bounds
        nodes = {}
        self.node_z = {}
        self.links = {}  # id -> (from, to, freespeed, modes)
        self.new_nodes = {}  # id -> (x, y, z) of nodes of the new links
        self.new_links = {}  # id -> (from, to, length) of links added along the official route
        self._new_links_cache = {}      # (from node, to node) -> ids of the new links between them

        opener = gzip.open if network_path.endswith(".gz") else open
        with opener(network_path, "rb") as f:
            for _, elem in ET.iterparse(f, events = ["end"]):
                if elem.tag == "node":
                    x, y = float(elem.attrib["x"]), float(elem.attrib["y"])
                    if minx <= x <= maxx and miny <= y <= maxy:
                        nodes[elem.attrib["id"]] = (x, y)
                        self.node_z[elem.attrib["id"]] = float(elem.attrib.get("z", 0.0))
                elif elem.tag == "link":
                    from_node, to_node = elem.attrib["from"], elem.attrib["to"]
                    if from_node in nodes and to_node in nodes:
                        self.links[elem.attrib["id"]] = (
                            from_node, to_node, float(elem.attrib["freespeed"]),
                            frozenset(elem.attrib.get("modes", "").split(",")),
                        )
                if elem.tag in ("node", "link"):
                    elem.clear()

        self.nodes = nodes
        self.segments = {
            link_id: LineString([nodes[link[0]], nodes[link[1]]]) for link_id, link in self.links.items()
        }
        self.length = {link_id: segment.length for link_id, segment in self.segments.items()}

        # Links the re-router may use: real (non-artificial) links open to the routing modes, or
        # to the fallback modes (e.g. car): buses run on such roads in reality, the network just
        # lacks the bus mode there. Using one means the mode has to be added to the link.
        self.routing_modes = frozenset(routing_modes)
        self.routing_ids = [
            link_id for link_id, link in self.links.items()
            if link[3] & (self.routing_modes | frozenset(fallback_modes)) and not link[3] & ARTIFICIAL_MODES
        ]
        self.routing_ids_set = set(self.routing_ids)
        self.adjacency = defaultdict(list)  # node -> [link ids leaving it]
        for link_id in self.routing_ids:
            self.adjacency[self.links[link_id][0]].append(link_id)
        self._tree = STRtree([self.segments[link_id] for link_id in self.routing_ids])
        self._inside = {}


    def direction_cosine(self, link_id, vector):
        """Cosine between the direction of a link and a vector (0 if undefined)."""
        link = self.links.get(link_id)
        if link is None:
            return 0.0
        (x0, y0), (x1, y1) = self.nodes[link[0]], self.nodes[link[1]]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy) * math.hypot(*vector)
        return (dx * vector[0] + dy * vector[1]) / norm if norm > 1e-6 else 0.0


    def is_reverse(self, link_a, link_b):
        """True if link_b is the same road as link_a driven the other way (a U-turn)."""
        a, b = self.links.get(link_a), self.links.get(link_b)
        return a is not None and b is not None and a[0] == b[1] and a[1] == b[0]


    def count_uturns(self, link_ids):
        return sum(1 for a, b in zip(link_ids, link_ids[1:]) if self.is_reverse(a, b))


    def is_artificial(self, link_id):
        link = self.links.get(link_id)
        return link is not None and bool(link[3] & ARTIFICIAL_MODES)


    def lacks_mode(self, link_id):
        link = self.links.get(link_id)
        return link is not None and not link[3] & self.routing_modes and not link[3] & ARTIFICIAL_MODES


    def add_bus_links(self, source_node, target_node, coords, inside):
        """Links that only allow the routing modes from source_node to target_node through the
        given coordinates (which lie on the official route). Returns the new link ids."""
        if (source_node, target_node) in self._new_links_cache:
            for link_id in self._new_links_cache[(source_node, target_node)]:
                inside.setdefault(link_id, self.length[link_id])
            return self._new_links_cache[(source_node, target_node)]

        z0, z1 = self.node_z.get(source_node, 0.0), self.node_z.get(target_node, 0.0)
        points = [self.nodes[source_node]]
        for c in coords:
            if math.dist(c, points[-1]) >= NEW_LINK_MIN_SPACING and math.dist(c, self.nodes[target_node]) >= NEW_LINK_MIN_SPACING:
                points.append(c)
        points.append(self.nodes[target_node])

        # The official line has a vertex every few metres: keep the ones that matter, so that a
        # new road is a few links and not dozens of tiny ones.
        if len(points) > 2:
            points = [tuple(c) for c in LineString(points).simplify(NEW_LINK_SIMPLIFY).coords]

        node_ids = [source_node]
        for i, c in enumerate(points[1:-1], start = 1):
            node_id = "pt_bus_node_%d" % (len(self.new_nodes) + 1)
            z = z0 + (z1 - z0) * i / (len(points) - 1)
            self.nodes[node_id] = c
            self.node_z[node_id] = z
            self.new_nodes[node_id] = (c[0], c[1], z)
            node_ids.append(node_id)
        node_ids.append(target_node)

        link_ids = []
        for a, b in zip(node_ids, node_ids[1:]):
            link_id = "pt_bus_link_%d" % (len(self.new_links) + 1)
            segment = LineString([self.nodes[a], self.nodes[b]])
            self.links[link_id] = (a, b, NEW_LINK_SPEED, self.routing_modes)
            self.segments[link_id] = segment
            self.length[link_id] = segment.length
            self.new_links[link_id] = (a, b, segment.length)
            self.adjacency[a].append(link_id)
            self.routing_ids_set.add(link_id)
            inside[link_id] = segment.length
            link_ids.append(link_id)

        self._new_links_cache[(source_node, target_node)] = link_ids
        return link_ids


    def inside_lengths(self, key, geometry):
        """Length of each routing link that lies inside the corridor (only links with > 0)."""
        if key not in self._inside:
            candidates = self._tree.query(geometry, predicate = "intersects")
            segments = [self._tree.geometries[i] for i in candidates]
            lengths = shapely.length(shapely.intersection(segments, geometry)) if segments else []
            self._inside[key] = {
                self.routing_ids[i]: float(length) for i, length in zip(candidates, lengths) if length > 0
            }
        return self._inside[key]


    def link_is_inside(self, link_id, inside):
        return inside.get(link_id, 0.0) >= 0.5 * max(self.length[link_id], 1e-6)


    def route_metrics(self, link_ids, key, geometry):
        total = inside_total = artificial_total = 0.0
        known = [l for l in link_ids if l in self.segments]
        if not known:
            return 0.0, 0.0
        segments = [self.segments[l] for l in known]
        lengths = shapely.length(shapely.intersection(segments, geometry))
        for link_id, inside_length in zip(known, lengths):
            length = self.length[link_id]
            total += length
            inside_total += min(float(inside_length), length)
            if self.is_artificial(link_id):
                artificial_total += length
        if total == 0:
            return 0.0, 0.0
        return inside_total / total, artificial_total / total

    # -- routing ------------------------------------------------------------

    def shortest_path(self, source_node, target_node, inside, cache):
        """Cheapest list of links from source_node to target_node (None if unreachable)."""
        if source_node == target_node:
            return [], 0.0
        cache_key = (id(inside), source_node, target_node)
        if cache_key in cache:
            return cache[cache_key]

        best = {source_node: 0.0}
        previous = {}
        queue = [(0.0, source_node)]
        result = (None, math.inf)
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > best.get(node, math.inf):
                continue
            if node == target_node:
                path = []
                while node != source_node:
                    link_id = previous[node]
                    path.append(link_id)
                    node = self.links[link_id][0]
                result = (path[::-1], cost)
                break
            if cost > MAX_PATH_COST:
                break
            for link_id in self.adjacency.get(node, ()):
                _, head, freespeed, _ = self.links[link_id]
                step = self.length[link_id] / max(freespeed, 1.0)
                if not self.link_is_inside(link_id, inside):
                    step *= OUTSIDE_PENALTY
                if self.lacks_mode(link_id):
                    step *= NO_MODE_PENALTY
                new_cost = cost + step
                if new_cost < best.get(head, math.inf):
                    best[head] = new_cost
                    previous[head] = link_id
                    heapq.heappush(queue, (new_cost, head))

        cache[cache_key] = result
        return result

    def nearby_routing_links(self, x, y, radius, inside):
        """Routing links within `radius` of a point that lie in the corridor."""
        point = Point(x, y)
        found = self._tree.query(point.buffer(radius), predicate = "intersects")
        return [
            (self.routing_ids[i], self._tree.geometries[i].distance(point))
            for i in found if self.link_is_inside(self.routing_ids[i], inside)
        ]


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class Schedule:
    def __init__(self, schedule_path):
        opener = gzip.open if schedule_path.endswith(".gz") else open
        with opener(schedule_path, "rt", encoding = "utf-8") as f:
            text = f.read()
        self.header = text[:text.index("<transitSchedule")]
        self.root = ET.fromstring(text)
        del text

        self.stops_element = self.root.find("transitStops")
        self.facilities = {f.attrib["id"]: f for f in self.stops_element.iter("stopFacility")}
        self.copies = defaultdict(list)  # base stop id -> facility ids
        self.station_copies = defaultdict(list)  # station id (before any ":") -> facility ids
        for facility_id in self.facilities:
            base_id = facility_id.split(".link:")[0]
            self.copies[base_id].append(facility_id)
            self.station_copies[base_id.split(":")[0]].append(facility_id)


    def routes(self):
        for line in self.root.iter("transitLine"):
            for route in line.iter("transitRoute"):
                yield line, route


    def stop_name(self, facility_id):
        facility = self.facilities.get(facility_id)
        return facility.attrib.get("name", facility_id) if facility is not None else facility_id


    def add_copy(self, base_id, link_id):
        facility_id = "%s.link:%s" % (base_id, link_id)
        if facility_id not in self.facilities:
            template = self.facilities[self.copies[base_id][0]]
            attributes = dict(template.attrib)
            attributes.update(id = facility_id, linkRefId = link_id)
            self.facilities[facility_id] = ET.SubElement(self.stops_element, "stopFacility", attributes)
            self.copies[base_id].append(facility_id)
            self.station_copies[base_id.split(":")[0]].append(facility_id)
        return facility_id


    def write(self, path):
        ET.indent(self.root, "\t")
        text = self.header + ET.tostring(self.root, encoding = "unicode") + "\n"
        with gzip.open(path, "wt", encoding = "utf-8") as f:
            f.write(text)


# ---------------------------------------------------------------------------
# Check + correction
# ---------------------------------------------------------------------------

REPORT_COLUMNS = [
    "line_name", "line_id", "route_id", "destination", "mode", "n_stops", "n_links", "length_m",
    "reference_matched", "share_inside", "share_artificial", "status", "new_share_inside",
    "n_stops_moved", "n_new_links", "n_uturns", "new_n_uturns",
]


def check_and_correct(schedule, network, corridors, modes, min_share, correct, stop_radius,
                      add_links = True, log = print, line_agencies = None, agencies = ()):
    """Returns (report rows, links that need the route mode added, new link links used).
    If agencies are given, only the lines operated by one of them (line_agencies maps the line id
    to its agency) are compared with the official routes: lines of other operators can have the
    same name as an official line (e.g. line 7 in Geneva and in France).
    With correct = True the schedule is modified in place."""
    rows = []
    patched_links = set()
    used_new_links = set()
    corrections = {}  # (corridor key, stop refIds) -> (new stop refIds, new link ids) or None
    path_cache = {}
    counts = defaultdict(int)

    for line, route in schedule.routes():
        mode = (route.findtext("transportMode") or "").strip().lower()
        if mode not in modes:
            continue

        profile = route.find("routeProfile")
        stop_elements = list(profile.findall("stop")) if profile is not None else []
        route_element = route.find("route")
        if len(stop_elements) < 2 or route_element is None:
            continue

        stop_ids = [s.attrib["refId"] for s in stop_elements]
        link_ids = [l.attrib["refId"] for l in route_element.findall("link")]
        line_name = line.attrib.get("name") or line.attrib["id"]
        destination = schedule.stop_name(stop_ids[-1])

        if agencies and line_agencies.get(line.attrib["id"]) not in agencies:
            counts["other_agency"] += 1
            continue

        reference = corridors.lookup(line_name, destination)
        if reference is None:
            counts["no_reference"] += 1
            continue
        key, geometry, matched = reference

        # Lines with the same name exist in other cities: only routes serving the area of the
        # official route are compared.
        stop_points = shapely.points([
            (float(schedule.facilities[s].attrib["x"]), float(schedule.facilities[s].attrib["y"]))
            for s in stop_ids
        ])
        if shapely.dwithin(stop_points, geometry, stop_radius).mean() < 0.5:
            counts["other_area"] += 1
            continue

        share_inside, share_artificial = network.route_metrics(link_ids, key, geometry)
        n_uturns = network.count_uturns(link_ids)
        length = sum(network.length.get(l, 0.0) for l in link_ids)
        row = dict(
            line_name = line_name, line_id = line.attrib["id"], route_id = route.attrib["id"],
            destination = destination, mode = mode, n_stops = len(stop_ids), n_links = len(link_ids),
            length_m = round(length), reference_matched = matched,
            share_inside = round(share_inside, 3), share_artificial = round(share_artificial, 3),
            status = "ok" if share_inside >= min_share and n_uturns == 0 else "deviates",
            new_share_inside = "", n_stops_moved = "", n_new_links = "",
            n_uturns = n_uturns, new_n_uturns = "",
        )

        if row["status"] == "deviates" and correct:
            cache_key = (key, tuple(stop_ids))
            if cache_key not in corrections:
                corrections[cache_key] = _reroute_with_bus_links(
                    schedule, network, corridors, key, geometry, stop_ids, stop_radius, path_cache, add_links
                )
            result = corrections[cache_key]

            if result is None:
                row["status"] = "unchanged"
            else:
                new_stop_ids, new_links, moved = result[:3]
                new_share, _ = network.route_metrics(new_links, key, geometry)
                new_uturns = network.count_uturns(new_links)
                # better: closer to the official route, or fewer U-turns without leaving it
                if new_share > share_inside + 1e-6 or (new_uturns < n_uturns and new_share >= share_inside - 0.005):
                    _apply(route_element, stop_elements, new_stop_ids, new_links)
                    patched_links.update(l for l in new_links if network.lacks_mode(l))
                    added = [l for l in new_links if l in network.new_links]
                    used_new_links.update(added)
                    row.update(
                        status = "corrected" if new_share >= min_share and new_uturns == 0 else "improved",
                        new_share_inside = round(new_share, 3), new_n_uturns = new_uturns, n_stops_moved = moved,
                        n_new_links = len(set(added)),
                    )
                else:
                    row["status"] = "unchanged"

        counts[row["status"]] += 1
        rows.append(row)

    log("PT route check: %s" % ", ".join("%s: %d" % item for item in sorted(counts.items())))
    return rows, patched_links, used_new_links


def _apply(route_element, stop_elements, new_stop_ids, new_links):
    for element, stop_id in zip(stop_elements, new_stop_ids):
        element.set("refId", stop_id)
    for link in list(route_element):
        route_element.remove(link)
    for link_id in new_links:
        ET.SubElement(route_element, "link", refId = link_id)


def _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache):
    inside = network.inside_lengths(key, geometry)

    # candidate (facility id or None for a new copy, base id, link id, cost) for each stop
    candidates = []
    stop_xy = [(float(schedule.facilities[s].attrib["x"]), float(schedule.facilities[s].attrib["y"])) for s in stop_ids]
    for index, stop_id in enumerate(stop_ids):
        base_id = stop_id.split(".link:")[0]
        facility = schedule.facilities[stop_id]
        x, y = stop_xy[index]
        point = Point(x, y)

        # Direction of travel at the stop (from the previous to the next stop): a stop on the
        # other carriageway makes the bus overshoot and turn back.
        before, after = stop_xy[max(index - 1, 0)], stop_xy[min(index + 1, len(stop_xy) - 1)]
        travel = (after[0] - before[0], after[1] - before[1])
        if math.hypot(*travel) < 20.0:
            travel = None

        options = {}  # link id -> (facility id or None for a new copy, base id, distance)
        for copy_id in schedule.copies[base_id]:
            link_id = schedule.facilities[copy_id].attrib.get("linkRefId")
            if link_id in network.segments and link_id in network.routing_ids_set:
                distance = network.segments[link_id].distance(point)
                options[link_id] = (copy_id, base_id, distance)
        for link_id, distance in network.nearby_routing_links(x, y, stop_radius, inside):
            options.setdefault(link_id, (None, base_id, distance))

        # The schedule may refer to a station as a whole (e.g. the centroid of Cornavin) although
        # its platforms are what the line serves: if this stop is not on the official route,
        # platforms of the same station that are can be used instead.
        if geometry.distance(point) > STATION_SWITCH_MIN_DISTANCE:
            for copy_id in schedule.station_copies[base_id.split(":")[0]]:
                facility = schedule.facilities[copy_id]
                copy_base = copy_id.split(".link:")[0]
                link_id = facility.attrib.get("linkRefId")
                platform = Point(float(facility.attrib["x"]), float(facility.attrib["y"]))
                if (copy_base != base_id and link_id not in options and link_id in network.segments
                        and link_id in network.routing_ids_set and geometry.contains(platform)):
                    options[link_id] = (
                        copy_id, copy_base,
                        network.segments[link_id].distance(platform) + PLATFORM_SWITCH_DISTANCE,
                    )

        # No usable road near the stop (pt2matsim put it on an artificial link of its own): it
        # stays where it is, and the legs to it are connected with new links along the official route.
        if not options:
            own_link = schedule.facilities[stop_id].attrib.get("linkRefId")
            if own_link in network.segments:
                options[own_link] = (stop_id, base_id, network.segments[own_link].distance(point))

        candidates.append([
            (copy_id, copy_base, link_id, STOP_DISTANCE_COST * distance
             + (0.0 if network.link_is_inside(link_id, inside) else OUTSIDE_STOP_COST)
             + (OPPOSITE_DIRECTION_COST if travel is not None and network.direction_cosine(link_id, travel) < -0.3 else 0.0))
            for link_id, (copy_id, copy_base, distance) in options.items()
        ])
        if not candidates[-1]:
            return None

    # Viterbi over the stop copies
    best = [[c[3] for c in candidates[0]]]
    back = []
    legs = {}
    for i in range(1, len(candidates)):
        costs, pointers = [], []
        for j, (_, _, link_j, cost_j) in enumerate(candidates[i]):
            best_cost, best_k = math.inf, None
            for k, (_, _, link_k, _) in enumerate(candidates[i - 1]):
                if best[i - 1][k] == math.inf:
                    continue
                if link_k == link_j:
                    path, leg_cost = [], 0.0
                else:
                    path, leg_cost = network.shortest_path(
                        network.links[link_k][1], network.links[link_j][0], inside, path_cache
                    )
                    if path is None:
                        continue
                legs[(i, k, j)] = path
                total = best[i - 1][k] + leg_cost + cost_j
                first = path[0] if path else link_j
                last = path[-1] if path else link_k
                if network.is_reverse(link_k, first) or network.is_reverse(last, link_j):
                    total += U_TURN_COST
                if total < best_cost:
                    best_cost, best_k = total, k
            costs.append(best_cost)
            pointers.append(best_k)
        best.append(costs)
        back.append(pointers)

    last = min(range(len(candidates[-1])), key = lambda j: best[-1][j])
    if best[-1][last] == math.inf:
        return None

    chosen = [last]
    for pointers in reversed(back):
        chosen.append(pointers[chosen[-1]])
    chosen.reverse()

    new_stop_ids, new_links, moved = [], [], 0
    stop_links, leg_lengths = [], []
    for i, j in enumerate(chosen):
        copy_id, base_id, link_id, _ = candidates[i][j]
        if copy_id is None:
            copy_id = schedule.add_copy(base_id, link_id)
        moved += copy_id != stop_ids[i]
        new_stop_ids.append(copy_id)
        stop_links.append(link_id)
        if i == 0:
            new_links.append(link_id)
        else:
            leg = legs[(i, chosen[i - 1], j)]
            leg_lengths.append(sum(network.length[l] for l in leg))
            new_links.extend(leg)
            if link_id != candidates[i - 1][chosen[i - 1]][2]:
                new_links.append(link_id)
    return new_stop_ids, new_links, moved, stop_links, leg_lengths


def _add_bus_links(schedule, network, graph, key, geometry, stop_ids, stop_links, leg_lengths, path_cache):
    """Adds new bus links along the official route for legs without a sensible connection in the
    network. `leg_lengths[i]` is the length of the path between stop i and i + 1 (inf if none).
    Returns the number of new links added."""
    inside = network.inside_lengths(key, geometry)
    added = 0

    for i, leg_length in enumerate(leg_lengths):
        endpoints = []
        for stop_id in (stop_ids[i], stop_ids[i + 1]):
            facility = schedule.facilities[stop_id]
            endpoints.append(graph.snap(float(facility.attrib["x"]), float(facility.attrib["y"]), NEW_LINK_SNAP_DISTANCE))
        if None in endpoints or endpoints[0] == endpoints[1]:
            continue

        official = graph.path(*endpoints)
        if official is None or official[1] > NEW_LINK_MAX_LENGTH:
            continue
        coords, official_length = official
        if leg_length <= NEW_LINK_RATIO * official_length + NEW_LINK_SLACK:
            continue

        source_node, target_node = network.links[stop_links[i]][1], network.links[stop_links[i + 1]][0]
        if source_node == target_node or (source_node, target_node) in network._new_links_cache:
            continue
        network.add_bus_links(source_node, target_node, coords, inside)
        added += 1

    if added:
        path_cache.clear()
    return added


def _reroute_with_bus_links(schedule, network, corridors, key, geometry, stop_ids, stop_radius, path_cache, add_links):
    result = _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache)
    graph = corridors.centerline_graph(key) if add_links else None
    if graph is None:
        return result

    if result is not None:
        stop_links, leg_lengths = result[3], result[4]
    else:
        # No connection at all: look for the legs without a path between the current stop links
        stop_links = [schedule.facilities[s].attrib.get("linkRefId") for s in stop_ids]
        if any(l not in network.segments for l in stop_links):
            return None
        inside = network.inside_lengths(key, geometry)
        leg_lengths = []
        for a, b in zip(stop_links, stop_links[1:]):
            if a == b:
                leg_lengths.append(0.0)
                continue
            path, _ = network.shortest_path(network.links[a][1], network.links[b][0], inside, path_cache)
            leg_lengths.append(math.inf if path is None else sum(network.length[l] for l in path))

    chosen_stop_ids = result[0] if result is not None else stop_ids
    if _add_bus_links(schedule, network, graph, key, geometry, chosen_stop_ids, stop_links, leg_lengths, path_cache):
        result = _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache)
    return result


def write_patched_network(network_path, output_path, link_ids, mode, new_nodes = None,
                          new_links = None, new_link_modes = None):
    """Copy of a MATSim network file where `mode` is added to the given links and the new
    nodes and links (id -> (x, y, z) and id -> (from, to, length)) are added."""
    link_ids = set(link_ids)
    new_nodes, new_links = new_nodes or {}, new_links or {}
    pattern = re.compile(r'^(\s*<link id=")([^"]*)(".*? modes=")([^"]*)(")')
    patched = 0
    opener = gzip.open if network_path.endswith(".gz") else open
    with opener(network_path, "rt", encoding = "utf-8") as source, \
            gzip.open(output_path, "wt", encoding = "utf-8") as target:
        for line in source:
            if new_nodes and "</nodes>" in line:
                for node_id, (x, y, z) in new_nodes.items():
                    target.write('\t\t<node id="%s" x="%.3f" y="%.3f" z="%.1f" />\n' % (node_id, x, y, z))
            elif new_links and "</links>" in line:
                for link_id, (a, b, length) in new_links.items():
                    target.write(
                        '\t\t<link id="%s" from="%s" to="%s" length="%.3f" freespeed="%.3f" capacity="600.0" '
                        'permlanes="1.0" oneway="1" modes="%s" />\n' % (link_id, a, b, length, NEW_LINK_SPEED, new_link_modes)
                    )
            match = pattern.match(line) if "<link id=" in line else None
            if match and match.group(2) in link_ids:
                modes = match.group(4).split(",")
                if mode not in modes:
                    modes.append(mode)
                    line = match.group(1) + match.group(2) + match.group(3) + ",".join(modes) \
                        + match.group(5) + line[match.end():]
                    patched += 1
            target.write(line)
    return patched


def write_report(rows, path):
    with open(path, "w", newline = "", encoding = "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames = REPORT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run(schedule_path, network_path, shapefile_path, report_path, corrected_schedule_path = None,
        patched_network_path = None, modes = ("bus",), fallback_modes = ("car",), buffer_m = 30.0,
        min_share = 0.9, stop_radius = 80.0, add_links = True, log = print,
        gtfs_routes_path = None, agencies = ()):
    """Checks (and with corrected_schedule_path also corrects) the mapped routes. If routes had
    to use links without the route mode, or links added along the official route, a network with
    the mode added / the new links is written to patched_network_path.
    Returns (rows, number of network changes: links patched + links added).
    agencies (with gtfs_routes_path, the GTFS routes.txt) restricts the check to the lines of these
    agencies."""
    agencies = set(map(str, agencies or ()))
    line_agencies = None
    if agencies:
        assert gtfs_routes_path, "gtfs_routes_path is needed to filter the lines by agency"
        line_agencies = read_line_agencies(gtfs_routes_path)
    corridors = Corridors(shapefile_path, buffer_m)
    log("Reading network inside the area of the official routes ...")
    network = Network(
        network_path, corridors.bounds, set(modes),
        fallback_modes = set(fallback_modes) if corrected_schedule_path else set(),
    )
    log("  -> %d links (%d usable for re-routing)" % (len(network.links), len(network.routing_ids)))

    log("Reading schedule ...")
    schedule = Schedule(schedule_path)

    rows, patched_links, used_new_links = check_and_correct(
        schedule, network, corridors, set(modes), min_share,
        correct = corrected_schedule_path is not None, stop_radius = stop_radius,
        add_links = add_links, log = log, line_agencies = line_agencies, agencies = agencies,
    )
    write_report(rows, report_path)

    n_patched = 0
    if corrected_schedule_path is not None:
        schedule.write(corrected_schedule_path)
        if patched_network_path and (patched_links or used_new_links):
            new_links = {l: network.new_links[l] for l in used_new_links}
            used_nodes = {n for a, b, _ in new_links.values() for n in (a, b)}
            new_nodes = {n: v for n, v in network.new_nodes.items() if n in used_nodes}
            log("Adding mode %s to %d link(s) and %d new %s-only link(s) used by corrected routes ..." % (
                ",".join(modes), len(patched_links), len(new_links), ",".join(modes)))
            n_patched = write_patched_network(
                network_path, patched_network_path, patched_links, modes[0],
                new_nodes, new_links, ",".join(sorted(modes)),
            ) + len(new_links)
    return rows, n_patched


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--schedule", required = True)
    parser.add_argument("--network", required = True)
    parser.add_argument("--shapefile", required = True)
    parser.add_argument("--report", required = True)
    parser.add_argument("--corrected-schedule", default = None,
                         help = "If given, deviating routes are re-routed and the schedule written here")
    parser.add_argument("--patched-network", default = None,
                         help = "Where to write the network with the bus mode added to links used by corrected routes")
    parser.add_argument("--no-new-links", action = "store_true",
                         help = "Do not add links along the official route where the network has no connection")
    parser.add_argument("--gtfs-routes", default = None, help = "GTFS routes.txt (to filter by agency)")
    parser.add_argument("--agency", action = "append", default = [],
                         help = "Only check the lines of this GTFS agency_id (repeatable)")
    parser.add_argument("--buffer", type = float, default = 30.0)
    parser.add_argument("--min-share", type = float, default = 0.9)
    args = parser.parse_args()

    run(args.schedule, args.network, args.shapefile, args.report, args.corrected_schedule,
        patched_network_path = args.patched_network, buffer_m = args.buffer, min_share = args.min_share,
        add_links = not args.no_new_links, gtfs_routes_path = args.gtfs_routes, agencies = args.agency)
