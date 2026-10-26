import csv
import gzip
import heapq
import itertools
import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict

import geopandas as gpd
import numpy as np
import shapely
import shapely.ops
from shapely.geometry import LineString, Point
from pyproj import Transformer
from shapely.strtree import STRtree

CRS = "EPSG:2056"
ARTIFICIAL_MODES = {"artificial", "stopFacilityLink"}

# The cost of a link is its length (metres) times 1 + d / LANE_SCALE, d its distance to the reference route (see
# LANE_SCALE), times the multipliers below; the other costs are in metres too.
OUTSIDE_PENALTY    = 20.0    # cost multiplier for links outside the official corridor
NO_MODE_PENALTY    = 1.3     # cost multiplier for links that do not allow the route's mode yet
CONTRAFLOW_PENALTY = 3.0     # cost multiplier for driving a one-way link of the corridor against its direction
OPPOSITE_DIRECTION_COST = 300.0  # cost of a stop on a link that runs against the direction of travel
U_TURN_COST        = 3000.0   # cost of turning back onto the reverse of the link just driven
U_TURN_MAX_SHARE_LOSS = 0.02  # a correction with fewer U-turns or repeated metres is kept if its share inside drops by at most this
# A route that drives more than this many metres on links it has already driven goes round a loop (e.g. a stop
# on the carriageway of the other direction). An out-and-back spur uses the two carriageways, so it repeats none.
MAX_REPEATED_LENGTH = 50.0
# A stop further than this from its link is not served by the route (e.g. the route takes a parallel street):
# a short detour of a long route keeps its share inside high, so the share alone does not reveal it.
MAX_STOP_LINK_DISTANCE = 30.0
# The corridor around a reference route also holds the parallel lanes (e.g. the car lanes next to a bus lane or a tram
# track shared with buses) and the other carriageway of a boulevard. A link costs 1 + d / LANE_SCALE times its length,
# d its distance to the reference route (_link_lane_distances), so that the lane closest to it is taken. A GTFS shape
# follows the lane the bus drives: in its corridor, a link further than LANE_DISTANCE is off the lane, and a route
# that drives more than MAX_OFF_LANE_LENGTH metres off the lane deviates.
LANE_SCALE = 2.0
LANE_DISTANCE = 8.0
MAX_OFF_LANE_LENGTH = 50.0
# Short links of these modes only (e.g. a few metres of bus lane mapped as a cycleway in OSM) can close a gap in the
# lane of a GTFS shape: they are used if they are on the lane (both ends within LANE_DISTANCE of the shape) and get
# the route mode in the patched network, like the links of the fallback modes.
GAP_MODES = {"bike"}
ALONG_MIN_RATIO = 0.7
GAP_MAX_LENGTH = 50.0
STOP_DISTANCE_COST = 1.0     # cost per metre between a stop and the link it is put on
OUTSIDE_STOP_COST  = 2000.0  # cost of putting a stop on a link outside the corridor
MAX_PATH_COST      = 60000.0
BOUNDS_MARGIN      = 1000.0
REVERSE_PREFIX     = "reverse:"  # routing id of a one-way link driven against its direction

# The stops of a route are within this distance (median) of the official route they follow
DIRECTION_FIT_DISTANCE = 20.0
# A GTFS shape passes a stop within this distance; it is the reference of a route if it passes its stops in order,
# all but at most SHAPE_MAX_MISSED_SHARE of them (e.g. a stop at the centroid of a station)
SHAPE_FIT_DISTANCE = 30.0
SHAPE_MAX_MISSED_SHARE = 0.2
SHAPE_KEY = "gtfs_shape"  # first item of the corridor key of a GTFS shape

PLATFORM_SWITCH_DISTANCE    = 100.0  # extra "distance" for using another platform of the same station
STATION_SWITCH_MIN_DISTANCE = 50.0   # only switch platform if the stop is this far outside the corridor

NEW_LINK_RATIO         = 1.5     # a leg gets new links if its path is longer than this x the official distance ...
NEW_LINK_SLACK         = 100.0   # ... plus this many metres
NEW_LINK_MAX_OUTSIDE   = 100.0   # ... or if it still has a U-turn or this many metres outside the corridor
NEW_LINK_SNAP_DISTANCE = 60.0    # stops further than this from the official route get no new links
NEW_LINK_MAX_LENGTH    = 3000.0
NEW_LINK_SPEED         = 11.1    # m/s
NEW_LINK_MIN_SPACING   = 3.0     # metres between the nodes of a new link
NEW_LINK_SIMPLIFY      = 2.0     # metres: tolerance when simplifying the official line into new links
# A stop with no link within MAX_STOP_LINK_DISTANCE (e.g. on a station forecourt or a bus loop missing in the network)
# is connected with new links along the GTFS shape, from the last node on the shape before it (within
# STOP_ACCESS_MAX_LENGTH) to a new node next to it, and on to the first node on the shape after it
STOP_ACCESS_MAX_LENGTH = 400.0


def read_line_agencies(routes_path):
    """GTFS route_id -> agency_id. The line ids of the mapped schedule are the GTFS route ids.
    The suffix added by the GTFS merge to duplicate ids (e.g. 881_m11) is removed from the agency_id."""
    with open(routes_path, newline = "", encoding = "utf-8-sig") as f:
        return {row["route_id"]: re.sub(r"_m\d+$", "", row.get("agency_id", "")) for row in csv.DictReader(f)}


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

# GTFS line name -> line of the official routes, by GTFS agency, where the operator numbers its lines differently in its
# shapefile (TL: the metro and night lines have numbers)
SHAPEFILE_LINE_ALIASES = {
    "151": {"m1": "70", "m2": "72", "N1": "91", "N2": "92", "N3": "93", "N4": "94", "N5": "95", "N6": "96"},
}


# Official line routes by GTFS agency_id (paths relative to data_path): the default of pt_line_routes_path, shared by
# matsim.scenario.network.corrected and analysis.pt.transit_schedule.visualize
DEFAULT_LINE_ROUTES = {
    "881": "PT_line_routes/Geneva/TPG_LIGNES.shp",       # TPG
    "151": "PT_line_routes/Lausanne/Shape_lignes.shp",   # TL
}


def resolve_line_routes(value, data_path, log = print):
    """pt_line_routes_path with absolute paths (relative ones are in data_path), without the files that do not exist:
    {agency_id: path}, one path, or None."""
    absolute = lambda path: path if os.path.isabs(path) else os.path.join(data_path, path)
    sources = {agency or "": absolute(path) for agency, path in line_route_sources(value)}
    for agency, path in list(sources.items()):
        if not os.path.exists(path):
            log(f"WARNING: official line routes not found at {path}: not used")
            del sources[agency]
    if not sources:
        return None
    if list(sources) == [""]:
        return sources[""]
    return sources


def official_line_key(keys, line_name, agency = None):
    """Key (agency or None, normalized line) of the official routes of a line of a GTFS agency among keys: the
    shapefile of the agency (with its line numbers, SHAPEFILE_LINE_ALIASES), else a shapefile without agency."""
    for source in (agency, None):
        name = SHAPEFILE_LINE_ALIASES.get(source, {}).get(str(line_name).strip(), line_name) if source else line_name
        key = (source, normalize_line(name))
        if key in keys:
            return key
    return None


def line_route_sources(shapefile_path):
    """[(GTFS agency_id or None, shapefile path)] of the official line routes. shapefile_path: one path (its lines are
    used for the lines of any agency), {agency_id: path} or a list of paths / (agency_id, path). The lines of a
    shapefile with an agency are only the reference of the routes of that agency: other networks have lines with the
    same names (e.g. lines 1 to 9 in Geneva and Lausanne)."""
    if not shapefile_path:
        return []
    if isinstance(shapefile_path, str):
        return [(None, shapefile_path)]
    if isinstance(shapefile_path, dict):
        return [(None if agency in (None, "") else str(agency), path) for agency, path in shapefile_path.items() if path]
    return [(None, p) if isinstance(p, str) else (None if p[0] in (None, "") else str(p[0]), p[1]) for p in shapefile_path]


def read_line_columns(gdf):
    """(line names, directions) of a shapefile of official line routes. SITG TPG_LIGNES has
    LIGNE and DIRECTION; TL (Lausanne) has CODE_LIGNE and SENS_TXT = "origin -> destination"."""
    if "LIGNE" in gdf.columns:
        return gdf["LIGNE"], gdf["DIRECTION"].astype(str)
    return gdf["CODE_LIGNE"], gdf["SENS_TXT"].astype(str).str.split("->").str[-1]


def select_directions(directions, centerlines, destination_name, stop_points = None):
    """(indices of the official routes of a line that a route follows, whether one direction was found).

    Routes are matched on their terminus, but the stops decide when given: some features are labelled with the
    terminus of the other direction (TPG lines 21, 31 and 70), and a route whose terminus no feature has (e.g. a
    short-turn service) follows one direction only. Several features can have the same terminus (e.g. TPG line 2:
    the line and a "service limité" piece along another road): the one that passes all the stops is used alone, as
    the others would let the route leave its road. Without a match, all the routes of the line are used."""
    destination = normalize_name(destination_name)
    named = [i for i, d in enumerate(directions) if destination in d]

    if stop_points is not None and len(stop_points):
        distances = [shapely.distance(stop_points, c) for c in centerlines]
        fit = [float(np.median(d)) for d in distances]
        best = min(range(len(fit)), key = lambda i: (fit[i], i not in named))
        named_fit = min((fit[i] for i in named), default = math.inf)
        if fit[best] <= DIRECTION_FIT_DISTANCE and named_fit > 3.0 * fit[best] + 3.0:
            return [best], True
        if len(named) > 1:
            best_named = min(named, key = lambda i: fit[i])
            if float(distances[best_named].max()) <= 2.0 * DIRECTION_FIT_DISTANCE:
                return [best_named], True

    return (named, True) if named else (list(range(len(directions))), False)


def _line_key(route):
    """(agency, short name) of a GTFS route: the same line in several merged feeds (agency 881 and 881_m11)."""
    return re.sub(r"_m\d+$", "", route.get("agency_id", "")), route.get("route_short_name", "")


def read_gtfs_shapes(routes_path, trips_path, shapes_path, trip_ids, line_ids):
    """GTFS shapes of the given trips (transit route ids of a schedule converted by pt2matsim are trip ids) and
    lines (route ids). Returns ({trip id: shape id}, {line id: [shape ids of the line]}, {shape id: LineString}).
    The shapes of a line are those of all the trips of the routes with the same agency and short name: the
    merged GTFS can have a line twice (e.g. from the national feed without shapes and from the operator's feed
    with shapes), and the trips of the schedule may come from the copy without shapes."""
    trip_ids, line_ids = set(trip_ids), set(line_ids)
    with open(routes_path, newline = "", encoding = "utf-8-sig") as f:
        route_keys = {row["route_id"]: _line_key(row) for row in csv.DictReader(f)}
    wanted_keys = {route_keys[l] for l in line_ids if l in route_keys}

    trip_shapes, key_shapes = {}, defaultdict(set)
    with open(trips_path, newline = "", encoding = "utf-8-sig") as f:
        for row in csv.DictReader(f):
            shape_id = row.get("shape_id")
            if not shape_id:
                continue
            if row["trip_id"] in trip_ids:
                trip_shapes[row["trip_id"]] = shape_id
            key = route_keys.get(row["route_id"])
            if key in wanted_keys:
                key_shapes[key].add(shape_id)

    points = defaultdict(list)
    wanted = set(trip_shapes.values()).union(*key_shapes.values())
    if wanted:
        with open(shapes_path, newline = "", encoding = "utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["shape_id"] in wanted:
                    points[row["shape_id"]].append(
                        (int(float(row["shape_pt_sequence"])), float(row["shape_pt_lon"]), float(row["shape_pt_lat"])))

    to_crs = Transformer.from_crs("EPSG:4326", CRS, always_xy = True)
    shapes = {}
    for shape_id, shape_points in points.items():
        shape_points.sort()
        x, y = to_crs.transform([p[1] for p in shape_points], [p[2] for p in shape_points])
        if len(shape_points) >= 2:
            shapes[shape_id] = LineString(list(zip(x, y)))
    line_shapes = {
        line_id: sorted(s for s in key_shapes.get(route_keys.get(line_id), ()) if s in shapes) for line_id in line_ids
    }
    return ({t: s for t, s in trip_shapes.items() if s in shapes},
            {l: v for l, v in line_shapes.items() if v}, shapes)


def shape_fit(shape, stop_points, tolerance = None):
    """(number of stops the shape does not pass, length) if the shape passes the stops in their order, else None.
    The stops are located one after the other along the rest of the shape (a stop served twice is located
    twice). A stop further than `tolerance` from the rest of the shape is not passed; more than
    SHAPE_MAX_MISSED_SHARE of the stops not passed and the shape does not fit."""
    tolerance = SHAPE_FIT_DISTANCE if tolerance is None else tolerance
    position, missed = 0.0, 0
    for point in stop_points:
        rest = shapely.ops.substring(shape, position, shape.length) if position > 0 else shape
        if rest.geom_type != "LineString":  # the end of the shape
            missed += rest.distance(point) > tolerance
            continue
        # the first stretch of the rest of the shape that passes the stop: the closest point of the whole rest
        # can be much further along (e.g. the first stop of a loop is also at its end)
        near = rest.intersection(point.buffer(tolerance))
        parts = [p for p in getattr(near, "geoms", [near]) if p.geom_type == "LineString" and not p.is_empty]
        if not parts:
            missed += 1
            continue
        first = min(parts, key = lambda p: rest.project(Point(p.coords[0])))
        position += rest.project(Point(first.coords[0])) + first.project(point)
    if missed > SHAPE_MAX_MISSED_SHARE * len(stop_points):
        return None
    return missed, shape.length


class Corridors:
    """The reference routes the PT routes are compared with: the GTFS shape of a route's trip if there is one,
    else the official line routes of a shapefile (selected by line and direction)."""

    def __init__(self, shapefile_path, buffer_m, trip_shapes = None, shapes = None, line_shapes = None):
        """shapefile_path: the official line routes, see line_route_sources (one path, {agency_id: path} or a list)."""
        self.buffer_m = buffer_m
        self.features = defaultdict(list)  # (agency or None, normalized line) -> [(directions, buffered geometry)]
        self.centerlines = defaultdict(list)  # (agency or None, normalized line) -> [unbuffered geometry]
        self.sources = line_route_sources(shapefile_path)
        bounds = []
        for agency, path in self.sources:
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs(CRS) if gdf.crs is not None else gdf.set_crs(CRS)
            for line, direction, geometry in zip(*read_line_columns(gdf), gdf.geometry):
                if geometry is None or geometry.is_empty:
                    continue
                geometry = shapely.force_2d(geometry)
                directions = {normalize_name(d) for d in str(direction).split("/")}
                self.features[(agency, normalize_line(line))].append((directions, shapely.buffer(geometry, buffer_m)))
                self.centerlines[(agency, normalize_line(line))].append(geometry)
            bounds.append(gdf.total_bounds)

        self.trip_shapes = trip_shapes or {}  # trip (= transit route) id -> shape id
        self.line_shapes = line_shapes or {}  # line (GTFS route) id -> shape ids of the line
        self.shapes = shapes or {}  # shape id -> LineString
        self._shape_choices = {}  # (line id, stop coordinates) -> shape id or None
        if self.shapes:
            bounds.append(shapely.total_bounds(list(self.shapes.values())))

        if bounds:
            minx, miny = min(b[0] for b in bounds), min(b[1] for b in bounds)
            maxx, maxy = max(b[2] for b in bounds), max(b[3] for b in bounds)
            self.bounds = (minx - buffer_m - BOUNDS_MARGIN, miny - buffer_m - BOUNDS_MARGIN,
                           maxx + buffer_m + BOUNDS_MARGIN, maxy + buffer_m + BOUNDS_MARGIN)
        else:
            self.bounds = (0.0, 0.0, 0.0, 0.0)
        self._geometries = {}
        self._graphs = {}


    def reference_line(self, key):
        """The reference route of a corridor key: its GTFS shape, or the official line route(s)."""
        if key[0] == SHAPE_KEY:
            return self.shapes[key[1]]
        line, selected = key
        return shapely.union_all([self.centerlines[line][i] for i in selected])


    def centerline(self, key):
        """The GTFS shape of a corridor key (None for an official line route, which does not follow the lanes)."""
        return self.shapes[key[1]] if key[0] == SHAPE_KEY else None


    def centerline_graph(self, key):
        """Graph of the reference route selected by a lookup key (None if not a single route)."""
        if key not in self._graphs:
            if key[0] == SHAPE_KEY:
                self._graphs[key] = CenterlineGraph(self.shapes[key[1]])
            else:
                line, selected = key
                self._graphs[key] = CenterlineGraph(self.centerlines[line][selected[0]]) if len(selected) == 1 else None
        return self._graphs[key]


    def shape_id(self, route_id, stop_points, line_id = None):
        """Shape id of the reference of a route: the shape of its GTFS trip if it passes its stops (shape_fit),
        else the shape of its line that passes its stops in order, missing the fewest, the shortest one (e.g.
        the shape of a short-turn service rather than of the whole line). None if no shape fits."""
        shape_id = self.trip_shapes.get(route_id)
        if shape_id is not None and shape_fit(self.shapes[shape_id], stop_points) is not None:
            return shape_id
        if line_id not in self.line_shapes:
            return None
        choice_key = (line_id, tuple(shapely.get_coordinates(stop_points).round(1).ravel()))
        if choice_key not in self._shape_choices:
            fits = [(shape_fit(self.shapes[s], stop_points), s) for s in self.line_shapes[line_id]]
            fits = [(fit, s) for fit, s in fits if fit is not None]
            self._shape_choices[choice_key] = min(fits)[1] if fits else None
        return self._shape_choices[choice_key]


    def line_key(self, line_name, agency = None):
        """Key of the official routes of a line of an agency in features / centerlines (None if there are none): the
        shapefile of the agency, else a shapefile without agency."""
        return official_line_key(self.features, line_name, agency)


    def lookup(self, line_name, destination_name, stop_points = None, route_id = None, line_id = None, agency = None):
        """Returns (key, geometry, matched_direction) or None if there is no reference for the route.
        The GTFS shape of the route is used if there is one (see shape_id), else the official route of the
        line in the shapefile of its agency (or one without agency), whose direction is selected by select_directions
        (from the destination and the stops)."""
        shape_id = self.shape_id(route_id, stop_points, line_id) if stop_points is not None else None
        if shape_id is not None:
            key = (SHAPE_KEY, shape_id)
            if key not in self._geometries:
                geometry = shapely.buffer(self.shapes[shape_id], self.buffer_m)
                shapely.prepare(geometry)
                self._geometries[key] = geometry
            return key, self._geometries[key], True

        line_key = self.line_key(line_name, agency)
        if line_key is None:
            return None
        features = self.features[line_key]

        selected, matched = select_directions(
            [directions for directions, _ in features], self.centerlines[line_key], destination_name, stop_points,
        )

        key = (line_key, tuple(selected))
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
        # Links that can only close a gap in the lane of a GTFS shape (see GAP_MODES); only when correcting
        self.gap_ids = {
            link_id for link_id, link in self.links.items()
            if fallback_modes and not link[3] & (self.routing_modes | frozenset(fallback_modes))
            and link[3] & GAP_MODES and not link[3] & ARTIFICIAL_MODES and self.length[link_id] <= GAP_MAX_LENGTH
        }
        self.routing_ids += sorted(self.gap_ids)
        self.routing_ids_set = set(self.routing_ids)
        self.adjacency = defaultdict(list)  # node -> [link ids leaving it]
        self.incoming = defaultdict(list)  # node -> [link ids arriving there]
        for link_id in self.routing_ids:
            self.adjacency[self.links[link_id][0]].append(link_id)
            self.incoming[self.links[link_id][1]].append(link_id)
        self._routing_pairs = {self.links[link_id][:2] for link_id in self.routing_ids}
        self._reverse_links = {}  # one-way link id -> id of the link added against it
        self._reverse_of = {}  # id of a link added against a one-way link -> id of the one-way link
        self.access_shapes = []  # GTFS shapes of all the lines, to reach the stops their own shape does not pass
        self._stop_nodes = {}  # rounded coordinates -> node added next to a stop (add_stop_access)
        self._expansions = {}  # node -> {id(inside): [(link id, step cost, head node)]}, see _expand
        self._tree = STRtree([self.segments[link_id] for link_id in self.routing_ids])
        self._inside = {}
        self._corridors = {}  # corridor key -> geometry
        self._centerlines = {}  # corridor key -> GTFS shape (the corridors that follow the lanes)
        self._reference_lines = {}  # corridor key -> reference route (GTFS shape or official line route)
        self._lane_distances = {}  # id(inside lengths of a corridor) -> {link id: distance to its reference route}
        self._shape_corridors = set()  # id(inside lengths) of the corridors of a GTFS shape


    def direction_cosine(self, link_id, vector):
        """Cosine between the direction of a link and a vector (0 if undefined)."""
        if link_id.startswith(REVERSE_PREFIX):
            return -self.direction_cosine(link_id[len(REVERSE_PREFIX):], vector)
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


    def stops_off_links(self, schedule, stop_ids):
        """Number of stops further than MAX_STOP_LINK_DISTANCE from the link they are on."""
        n = 0
        for stop_id in stop_ids:
            facility = schedule.facilities[stop_id]
            segment = self.segments.get(facility.attrib.get("linkRefId"))
            if segment is not None and segment.distance(
                    Point(float(facility.attrib["x"]), float(facility.attrib["y"]))) > MAX_STOP_LINK_DISTANCE:
                n += 1
        return n


    def repeated_length(self, link_ids):
        """Metres driven on links that the route has already driven."""
        seen, repeated = set(), 0.0
        for link_id in link_ids:
            if link_id in seen:
                repeated += self.length.get(link_id, 0.0)
            seen.add(link_id)
        return repeated


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
                if link_id not in inside:
                    inside[link_id] = self.length[link_id]
                    self._expansions.pop(self.links[link_id][0], None)
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
            self._expansions.pop(a, None)
            self.routing_ids_set.add(link_id)
            inside[link_id] = segment.length
            link_ids.append(link_id)

        self._add_to_corridors(link_ids)
        self._new_links_cache[(source_node, target_node)] = link_ids
        return link_ids


    def add_stop_access(self, x, y, centerline, inside):
        """Links along a GTFS shape that reach a stop at (x, y) without a link near it (STOP_ACCESS_MAX_LENGTH).
        Returns the new links that end or (at the start of the shape) start at the node next to the stop. If the shape
        does not pass the stop (e.g. it ends before it), the closest shape of another line that does is used."""
        point = Point(x, y)
        if centerline.distance(point) > NEW_LINK_SNAP_DISTANCE:
            others = [s for s in self.access_shapes if s.distance(point) <= NEW_LINK_SNAP_DISTANCE]
            if not others:
                return []
            centerline = min(others, key = point.distance)
        position = centerline.project(point)
        # The new links start at the end of a link that follows the shape (in its direction) before the stop, and end
        # at the start of one that follows it after the stop: a node merely close to the shape can be on a cross
        # street that the route does not come from
        stretch = shapely.ops.substring(centerline, max(position - STOP_ACCESS_MAX_LENGTH, 0.0),
                                        min(position + STOP_ACCESS_MAX_LENGTH, centerline.length))
        nearby = [self.routing_ids[int(i)] for i in self._tree.query(stretch, predicate = "dwithin", distance = LANE_DISTANCE)]
        nearby = [l for l in nearby if l not in self.gap_ids]
        along = [l for l, d in self._link_lane_distances(nearby, centerline).items() if d <= LANE_DISTANCE]
        before, after = None, None  # (position on the shape, node)
        for link_id in along:
            from_node, to_node = self.links[link_id][:2]
            start, end = (centerline.project(Point(self.nodes[n])) for n in (from_node, to_node))
            if end <= start:  # against the direction of the shape
                continue
            if end < position - NEW_LINK_MIN_SPACING and (before is None or end > before[0]):
                before = (end, to_node)
            if start > position + NEW_LINK_MIN_SPACING and (after is None or start < after[0]):
                after = (start, from_node)
        if before is None and after is None:
            return []

        stop_xy = tuple(centerline.interpolate(position).coords[0])
        stop_node = self._stop_nodes.get((round(stop_xy[0]), round(stop_xy[1])))
        if stop_node is None:
            stop_node = "pt_bus_node_%d" % (len(self.new_nodes) + 1)
            z = self.node_z.get((before or after)[1], 0.0)
            self.nodes[stop_node], self.node_z[stop_node] = stop_xy, z
            self.new_nodes[stop_node] = (stop_xy[0], stop_xy[1], z)
            self._stop_nodes[(round(stop_xy[0]), round(stop_xy[1]))] = stop_node

        stop_links = []
        if before is not None:
            coords = list(shapely.ops.substring(centerline, before[0], position).coords)[1:-1]
            stop_links.append(self.add_bus_links(before[1], stop_node, coords, inside)[-1])
        if after is not None:
            coords = list(shapely.ops.substring(centerline, position, after[0]).coords)[1:-1]
            links = self.add_bus_links(stop_node, after[1], coords, inside)
            if before is None:
                stop_links.append(links[0])
        return stop_links


    def add_reverse_link(self, link_id, inside):
        """Link that only allows the routing modes against a one-way link of the corridor (the network
        lacks the contraflow bus lane). Returns its id."""
        if link_id not in self._reverse_links:
            from_node, to_node, freespeed, _ = self.links[link_id]
            new_id = "pt_bus_link_%d" % (len(self.new_links) + 1)
            segment = LineString([self.nodes[to_node], self.nodes[from_node]])
            self.links[new_id] = (to_node, from_node, NEW_LINK_SPEED, self.routing_modes)
            self.segments[new_id] = segment
            self.length[new_id] = segment.length
            self.new_links[new_id] = (to_node, from_node, segment.length)
            self.adjacency[to_node].append(new_id)
            self.routing_ids_set.add(new_id)
            self._routing_pairs.add((to_node, from_node))
            self._reverse_links[link_id] = new_id
            self._reverse_of[new_id] = link_id
            self._add_to_corridors([new_id])
        new_id = self._reverse_links[link_id]
        if new_id not in inside:
            inside[new_id] = inside.get(link_id, 0.0)
        # the links leaving the head of the reverse link changed (or their cost in this corridor)
        self._expansions.pop(self.links[new_id][0], None)
        return new_id


    def _add_to_corridors(self, link_ids):
        """Adds new links to the inside lengths of every corridor they lie in. Without this, a link added for
        one line would be outside the corridors of the other lines (and of the other direction of the same
        line) although it lies on their official route, and the one-way link it is the reverse of could no
        longer be driven against its direction: the result would depend on the order of the routes."""
        segments = [self.segments[link_id] for link_id in link_ids]
        for key, geometry in self._corridors.items():
            inside = self._inside[key]
            for link_id, length in zip(link_ids, shapely.length(shapely.intersection(segments, geometry))):
                if length > 0 and link_id not in inside:
                    inside[link_id] = float(length)
            if key in self._reference_lines:
                self._lane_distances[id(inside)].update(self._link_lane_distances(link_ids, self._reference_lines[key]))
        for link_id in link_ids:
            self._expansions.pop(self.links[link_id][0], None)


    def _link_lane_distances(self, link_ids, centerline):
        """{link id: distance to a GTFS shape}: the median of the distances of the start, middle and end of the link.
        The middle of a long link can be far from the shape on a curve (the nodes are on the road, not the straight
        line between them), and a link that joins the lane of the shape has one end on it."""
        link_ids = list(link_ids)
        if not link_ids:
            return {}
        segments = [self.segments[l] for l in link_ids]
        distances = np.median([
            shapely.distance(shapely.line_interpolate_point(segments, f, normalized = True), centerline)
            for f in (0.0, 0.5, 1.0)
        ], axis = 0)
        return dict(zip(link_ids, distances.tolist()))


    def _off_lane_links(self, link_ids, centerline):
        """The links further than LANE_DISTANCE from a GTFS shape (_link_lane_distances)."""
        return {l for l, d in self._link_lane_distances(link_ids, centerline).items() if d > LANE_DISTANCE}


    def off_lane_length(self, link_ids, centerline):
        """Metres of a route on links that are off the lane of a GTFS shape (0 without a shape)."""
        if centerline is None:
            return 0.0
        known = [l for l in link_ids if l in self.segments and not self.is_artificial(l)]
        off = self._off_lane_links(set(known), centerline)
        return sum(self.length[l] for l in known if l in off)


    def link_is_off_lane(self, link_id, inside):
        return id(inside) in self._shape_corridors and self._lane_distances[id(inside)].get(link_id, 0.0) > LANE_DISTANCE


    def _lane_factor(self, link_id, inside):
        """Cost factor of a link of the corridor for its distance to the lane of the GTFS shape (1 without one)."""
        distances = self._lane_distances.get(id(inside))
        return 1.0 + distances.get(link_id, 0.0) / LANE_SCALE if distances else 1.0


    def inside_lengths(self, key, geometry, centerline = None, reference_line = None):
        """Length of each routing link that lies inside the corridor (only links with > 0), the links added
        so far (along the official routes or against one-way links) included. With the GTFS shape of the
        corridor (centerline), the links of the corridor that are off its lane are noted (link_is_off_lane)."""
        if key not in self._inside:
            candidates = self._tree.query(geometry, predicate = "intersects")
            segments = [self._tree.geometries[i] for i in candidates]
            lengths = shapely.length(shapely.intersection(segments, geometry)) if segments else []
            self._inside[key] = {
                self.routing_ids[i]: float(length) for i, length in zip(candidates, lengths) if length > 0
            }
            self._corridors[key] = geometry
            if self.new_links:
                new_ids = list(self.new_links)
                lengths = shapely.length(shapely.intersection([self.segments[l] for l in new_ids], geometry))
                self._inside[key].update((l, float(length)) for l, length in zip(new_ids, lengths) if length > 0)
            if centerline is not None:
                self._centerlines[key] = centerline
                self._inside[key].update(self._links_along(
                    [self.routing_ids[i] for i in candidates] + list(self.new_links), centerline))
                self._shape_corridors.add(id(self._inside[key]))
            reference_line = centerline if centerline is not None else reference_line
            if reference_line is not None:
                self._reference_lines[key] = reference_line
                self._lane_distances[id(self._inside[key])] = self._link_lane_distances(self._inside[key], reference_line)
        return self._inside[key]


    def _links_along(self, link_ids, centerline):
        """{link id: length} of the links that follow a GTFS shape although their straight line leaves the corridor
        (a long link on a curve): both ends within LANE_DISTANCE of the shape, and a length of at least
        ALONG_MIN_RATIO times the length of the shape between them (not a shortcut across a loop of the shape)."""
        link_ids = [l for l in link_ids if l in self.segments]
        if not link_ids:
            return {}
        starts = shapely.points([self.nodes[self.links[l][0]] for l in link_ids])
        ends = shapely.points([self.nodes[self.links[l][1]] for l in link_ids])
        near = (shapely.distance(starts, centerline) <= LANE_DISTANCE) & (shapely.distance(ends, centerline) <= LANE_DISTANCE)
        along = np.abs(shapely.line_locate_point(centerline, ends) - shapely.line_locate_point(centerline, starts))
        return {
            l: self.length[l] for l, n, a in zip(link_ids, near, along)
            if n and self.length[l] >= ALONG_MIN_RATIO * a
        }


    def link_is_inside(self, link_id, inside):
        if link_id.startswith(REVERSE_PREFIX):
            link_id = link_id[len(REVERSE_PREFIX):]
        return inside.get(link_id, 0.0) >= 0.5 * max(self.length[link_id], 1e-6)


    def route_metrics(self, link_ids, key, geometry):
        total = inside_total = artificial_total = 0.0
        known = [l for l in link_ids if l in self.segments]
        if not known:
            return 0.0, 0.0
        segments = [self.segments[l] for l in known]
        lengths = shapely.length(shapely.intersection(segments, geometry))
        inside = self._inside.get(key, {})  # includes the links along a GTFS shape (_links_along)
        for link_id, inside_length in zip(known, lengths):
            length = self.length[link_id]
            total += length
            inside_total += min(max(float(inside_length), inside.get(link_id, 0.0)), length)
            if self.is_artificial(link_id):
                artificial_total += length
        if total == 0:
            return 0.0, 0.0
        return inside_total / total, artificial_total / total

    # -- routing ------------------------------------------------------------

    def _ends(self, link_id):
        """(from node, to node) of a link, or of a one-way link driven against its direction."""
        if link_id.startswith(REVERSE_PREFIX):
            from_node, to_node = self.links[link_id[len(REVERSE_PREFIX):]][:2]
            return to_node, from_node
        return self.links[link_id][:2]


    def _turns_back(self, link_a, link_b):
        (a_from, a_to), (b_from, b_to) = self._ends(link_a), self._ends(link_b)
        return a_from == b_to and a_to == b_from


    def _next_links(self, node, inside):
        """Links the search can take from a node: the routing links leaving it, and the one-way links of
        the corridor arriving there, driven against their direction."""
        for link_id in self.adjacency.get(node, ()):
            if link_id not in self.gap_ids or self.closes_gap(link_id, inside):
                yield link_id
        for link_id in self.incoming.get(node, ()):
            from_node, to_node = self.links[link_id][:2]
            if (to_node, from_node) not in self._routing_pairs and self.link_is_inside(link_id, inside) \
                    and link_id not in self.gap_ids:
                yield REVERSE_PREFIX + link_id


    def closes_gap(self, link_id, inside):
        """True if a link of GAP_MODES can be used in a corridor: it follows the lane of its GTFS shape."""
        return id(inside) in self._shape_corridors and self.link_is_inside(link_id, inside) \
            and not self.link_is_off_lane(link_id, inside)


    def _step_cost(self, link_id, inside):
        if link_id.startswith(REVERSE_PREFIX):
            link_id = link_id[len(REVERSE_PREFIX):]
            return self.length[link_id] * CONTRAFLOW_PENALTY \
                * self._lane_factor(link_id, inside)
        if link_id in self._reverse_of:
            # same cost as before the link was added, so that the routes do not depend on which came first
            one_way = self._reverse_of[link_id]
            step = self.length[one_way] * CONTRAFLOW_PENALTY
            if not self.link_is_inside(link_id, inside):
                return step * OUTSIDE_PENALTY
            return step * self._lane_factor(link_id, inside)
        step = self.length[link_id]
        if not self.link_is_inside(link_id, inside):
            step *= OUTSIDE_PENALTY
        else:
            step *= self._lane_factor(link_id, inside)
        if self.lacks_mode(link_id):
            step *= NO_MODE_PENALTY
        return step


    def _expand(self, node, inside):
        """(link id, step cost, head node) of the links the search can take from a node (_next_links).
        Cached per corridor; add_bus_links and add_reverse_link drop the nodes whose links change."""
        by_corridor = self._expansions.setdefault(node, {})
        expansion = by_corridor.get(id(inside))
        if expansion is None:
            expansion = by_corridor[id(inside)] = [
                (next_id, self._step_cost(next_id, inside), self._ends(next_id)[1])
                for next_id in self._next_links(node, inside)
            ]
        return expansion


    def shortest_path(self, from_link, to_link, inside, cache):
        return self.shortest_paths(from_link, [to_link], inside, cache)[to_link]


    def shortest_paths(self, from_link, to_links, inside, cache):
        """{to_link: (cheapest list of links between from_link and to_link, both excluded, cost) or (None, inf)},
        from a single search for all the to_links. Turning back onto the reverse of the link just driven costs
        U_TURN_COST, also when entering a to_link. Where a path drives a one-way link against its direction,
        the link is added (add_reverse_link)."""
        results, targets = {}, defaultdict(list)  # from node of a to_link -> [(to_link, its head node)]
        pending = []
        for to_link in set(to_links):
            cache_key = (id(inside), from_link, to_link)
            if cache_key in cache:
                results[to_link] = cache[cache_key]
            else:
                pending.append(to_link)
                to_from, to_head = self._ends(to_link)
                targets[to_from].append((to_link, to_head))
        if not pending:
            return results

        # A state is a link (with its tail and head node), or ("__end__", to_link) once a to_link is entered.
        # Since the next link always leaves the head of the current one, it turns back if it leads to its tail.
        tail, head = self._ends(from_link)
        best = {from_link: 0.0}
        previous = {}
        found = {}
        order = itertools.count()
        queue = [(0.0, next(order), from_link, tail, head)]
        while queue and len(found) < len(pending):
            cost, _, state, tail, head = heapq.heappop(queue)
            if cost > best.get(state, math.inf):
                continue
            if tail is None:
                found[state[1]] = cost
                continue
            if cost > MAX_PATH_COST:
                break

            for to_link, to_head in targets.get(head, ()):
                end = ("__end__", to_link)
                final = cost + (U_TURN_COST if to_head == tail else 0.0)
                if final < best.get(end, math.inf):
                    best[end] = final
                    previous[end] = state
                    heapq.heappush(queue, (final, next(order), end, None, None))

            for next_id, step, next_head in self._expand(head, inside):
                new_cost = cost + step + (U_TURN_COST if next_head == tail else 0.0)
                if new_cost < best.get(next_id, math.inf):
                    best[next_id] = new_cost
                    previous[next_id] = state
                    heapq.heappush(queue, (new_cost, next(order), next_id, head, next_head))

        # The reverse links are only added once the search is over, as they change the graph
        for to_link, cost in found.items():
            path, state = [], previous[("__end__", to_link)]
            while state != from_link:
                path.append(state)
                state = previous[state]
            results[to_link] = ([
                self.add_reverse_link(l[len(REVERSE_PREFIX):], inside) if l.startswith(REVERSE_PREFIX) else l
                for l in reversed(path)
            ], cost)
        for to_link in pending:
            results.setdefault(to_link, (None, math.inf))
            cache[(id(inside), from_link, to_link)] = results[to_link]
        return results

    def nearby_routing_links(self, x, y, radius, inside):
        """Routing links within `radius` of a point that lie in the corridor."""
        point = Point(x, y)
        found = self._tree.query(point.buffer(radius), predicate = "intersects")
        return [
            (self.routing_ids[i], self._tree.geometries[i].distance(point))
            for i in found if self.link_is_inside(self.routing_ids[i], inside) and self.routing_ids[i] not in self.gap_ids
        ]


    def against(self, link_id, inside):
        """Id to drive a one-way link of the corridor against its direction: the link added for it if there is
        one, else REVERSE_PREFIX + link id (the link is added by add_reverse_link if a route uses it). None if the
        link is not one-way or lies outside the corridor."""
        if link_id in self._reverse_links:
            return self._reverse_links[link_id]
        from_node, to_node = self.links[link_id][:2]
        if (to_node, from_node) in self._routing_pairs or not self.link_is_inside(link_id, inside):
            return None
        return REVERSE_PREFIX + link_id


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
    "reference", "reference_matched", "share_inside", "share_artificial", "status", "new_share_inside",
    "n_stops_moved", "n_new_links", "n_uturns", "new_n_uturns", "repeated_m", "new_repeated_m",
    "n_stops_off", "new_n_stops_off", "off_lane_m", "new_off_lane_m",
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
    evaluated = {}  # (corridor key, stop refIds, link ids) -> (report row, (new stop refIds, new link ids) or None)
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

        stop_points = shapely.points([
            (float(schedule.facilities[s].attrib["x"]), float(schedule.facilities[s].attrib["y"]))
            for s in stop_ids
        ])
        reference = corridors.lookup(line_name, destination, stop_points, route_id = route.attrib["id"], line_id = line.attrib["id"],
                                     agency = line_agencies.get(line.attrib["id"]) if line_agencies else None)
        if reference is None:
            counts["no_reference"] += 1
            continue
        key, geometry, matched = reference
        centerline = corridors.centerline(key)
        network.inside_lengths(key, geometry, centerline, corridors.reference_line(key))

        # Lines with the same name exist in other cities: only routes serving the area of the
        # official route are compared.
        if shapely.dwithin(stop_points, geometry, stop_radius).mean() < 0.5:
            counts["other_area"] += 1
            continue

        # Routes of the same pattern (most routes: one per departure) are evaluated once
        pattern = (key, tuple(stop_ids), tuple(link_ids))
        if pattern in evaluated:
            fields, applied = evaluated[pattern]
            if applied is not None:
                _apply(route_element, stop_elements, *applied)
            row = dict(fields, line_name = line_name, line_id = line.attrib["id"], route_id = route.attrib["id"])
            counts[row["status"]] += 1
            rows.append(row)
            continue

        share_inside, share_artificial = network.route_metrics(link_ids, key, geometry)
        n_uturns = network.count_uturns(link_ids)
        repeated = network.repeated_length(link_ids)
        n_off = network.stops_off_links(schedule, stop_ids)
        applied = None
        off_lane = network.off_lane_length(link_ids, centerline)
        length = sum(network.length.get(l, 0.0) for l in link_ids)
        row = dict(
            line_name = line_name, line_id = line.attrib["id"], route_id = route.attrib["id"],
            destination = destination, mode = mode, n_stops = len(stop_ids), n_links = len(link_ids),
            length_m = round(length), reference = key[0] if key[0] == SHAPE_KEY else "line_routes",
            reference_matched = matched,
            share_inside = round(share_inside, 3), share_artificial = round(share_artificial, 3),
            status = "ok" if share_inside >= min_share and n_uturns == 0 and repeated <= MAX_REPEATED_LENGTH
                     and n_off == 0 and off_lane <= MAX_OFF_LANE_LENGTH else "deviates",
            new_share_inside = "", n_stops_moved = "", n_new_links = "",
            n_uturns = n_uturns, new_n_uturns = "", repeated_m = round(repeated), new_repeated_m = "",
            n_stops_off = n_off, new_n_stops_off = "", off_lane_m = round(off_lane), new_off_lane_m = "",
        )

        if row["status"] == "deviates" and correct:
            cache_key = pattern[:2]
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
                new_repeated = network.repeated_length(new_links)
                new_off = network.stops_off_links(schedule, new_stop_ids)
                new_off_lane = network.off_lane_length(new_links, centerline)
                # better: closer to the reference route, or fewer U-turns / loops / stops not served / metres off the
                # lane without leaving it
                fewer_detours = new_uturns <= n_uturns and new_repeated <= repeated + 1.0 and new_off <= n_off \
                    and new_off_lane <= off_lane + 1.0 and (new_uturns < n_uturns or new_repeated < repeated - 1.0
                                                           or new_off < n_off or new_off_lane < off_lane - 1.0)
                if new_share > share_inside + 1e-6 or (fewer_detours and new_share >= share_inside - U_TURN_MAX_SHARE_LOSS):
                    _apply(route_element, stop_elements, new_stop_ids, new_links)
                    patched_links.update(l for l in new_links if network.lacks_mode(l))
                    added = [l for l in new_links if l in network.new_links]
                    used_new_links.update(added)
                    row.update(
                        status = "corrected" if new_share >= min_share and new_uturns == 0
                                 and new_repeated <= MAX_REPEATED_LENGTH and new_off == 0
                                 and new_off_lane <= MAX_OFF_LANE_LENGTH else "improved",
                        new_share_inside = round(new_share, 3), new_n_uturns = new_uturns, n_stops_moved = moved,
                        new_repeated_m = round(new_repeated), new_n_stops_off = new_off, new_off_lane_m = round(new_off_lane),
                        n_new_links = len(set(added)),
                    )
                    applied = (new_stop_ids, new_links)
                else:
                    row["status"] = "unchanged"

        evaluated[pattern] = (row, applied)
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


def _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache, access_fallback = False):
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
            # the stop can be on a one-way road that the bus drives against its direction (contraflow bus lane
            # missing in the network): otherwise the route would have to drive the road the right way to reach it
            against = network.against(link_id, inside)
            if against is not None:
                options.setdefault(against, (None, base_id, distance))

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

        # No link near the stop (e.g. a forecourt or a bus loop missing in the network): it is reached by new links
        # along the GTFS shape
        # along the GTFS shape (only them: the other links would leave the platform aside, unless access_fallback)
        centerline = network._centerlines.get(key)
        near = {l: o for l, o in options.items() if o[2] <= MAX_STOP_LINK_DISTANCE}
        if centerline is not None and not near:
            access = network.add_stop_access(x, y, centerline, inside)
            if access:
                path_cache.clear()
                access = {l: (None, base_id, network.segments[l].distance(point)) for l in access}
                options = {**options, **access} if access_fallback else access

        # The links within MAX_STOP_LINK_DISTANCE serve the stop: the others only if there are none, as the cheaper
        # path they may allow is a route that does not reach the platform (e.g. on a parallel street)
        elif near:
            options = near

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
        # one search from each reachable candidate of the previous stop to all the candidates of this one
        target_links = [c[2] for c in candidates[i]]
        paths_from = {
            k: network.shortest_paths(link_k, [l for l in target_links if l != link_k], inside, path_cache)
            for k, (_, _, link_k, _) in enumerate(candidates[i - 1]) if best[i - 1][k] != math.inf
        }
        costs, pointers = [], []
        for j, (_, _, link_j, cost_j) in enumerate(candidates[i]):
            best_cost, best_k = math.inf, None
            for k, (_, _, link_k, _) in enumerate(candidates[i - 1]):
                if best[i - 1][k] == math.inf:
                    continue
                if link_k == link_j:
                    path, leg_cost = [], 0.0
                else:
                    path, leg_cost = paths_from[k][link_j]
                    if path is None:
                        continue
                legs[(i, k, j)] = path
                total = best[i - 1][k] + leg_cost + cost_j
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

    # a stop on a one-way link driven against its direction is put on the link added for it
    chosen_links = [candidates[i][j][2] for i, j in enumerate(chosen)]
    chosen_links = [
        network.add_reverse_link(l[len(REVERSE_PREFIX):], inside) if l.startswith(REVERSE_PREFIX) else l
        for l in chosen_links
    ]

    new_stop_ids, new_links, moved = [], [], 0
    stop_links, leg_lengths, legs_off_route = [], [], []
    for i, j in enumerate(chosen):
        copy_id, base_id, _, _ = candidates[i][j]
        link_id = chosen_links[i]
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
            legs_off_route.append(_is_off_route(network, inside, stop_links[-2], leg, link_id))
            new_links.extend(leg)
            if link_id != chosen_links[i - 1]:
                new_links.append(link_id)
    return new_stop_ids, new_links, moved, stop_links, leg_lengths, legs_off_route


def _is_off_route(network, inside, from_link, leg, to_link):
    """True if the path between two stop links turns back, or keeps more than NEW_LINK_MAX_OUTSIDE metres
    outside the corridor: the network lacks the official route there (e.g. a bus-only passage missing in OSM)."""
    links = [from_link] + leg + [to_link]
    if any(network._turns_back(a, b) for a, b in zip(links, links[1:])):
        return True
    return sum(network.length[l] for l in leg if not network.link_is_inside(l, inside)) > NEW_LINK_MAX_OUTSIDE


def _add_bus_links(schedule, network, graph, key, geometry, stop_ids, stop_links, leg_lengths, legs_off_route, path_cache):
    """Adds new bus links along the official route for legs without a sensible connection in the
    network. `leg_lengths[i]` is the length of the path between stop i and i + 1 (inf if none),
    `legs_off_route[i]` whether it turns back or leaves the corridor (_is_off_route).
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
        if leg_length <= NEW_LINK_RATIO * official_length + NEW_LINK_SLACK and not legs_off_route[i]:
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
    if result is None:  # e.g. the links added to reach a stop cannot be reached themselves
        result = _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache, access_fallback = True)
    graph = corridors.centerline_graph(key) if add_links else None
    if graph is None:
        return result

    if result is not None:
        stop_links, leg_lengths, legs_off_route = result[3], result[4], result[5]
    else:
        # No connection at all: look for the legs without a path between the current stop links
        stop_links = [schedule.facilities[s].attrib.get("linkRefId") for s in stop_ids]
        if any(l not in network.segments for l in stop_links):
            return None
        inside = network.inside_lengths(key, geometry)
        leg_lengths, legs_off_route = [], []
        for a, b in zip(stop_links, stop_links[1:]):
            if a == b:
                leg_lengths.append(0.0)
                legs_off_route.append(False)
                continue
            path, _ = network.shortest_path(a, b, inside, path_cache)
            leg_lengths.append(math.inf if path is None else sum(network.length[l] for l in path))
            legs_off_route.append(path is not None and _is_off_route(network, inside, a, path, b))

    chosen_stop_ids = result[0] if result is not None else stop_ids
    if _add_bus_links(schedule, network, graph, key, geometry, chosen_stop_ids, stop_links, leg_lengths, legs_off_route, path_cache):
        result = _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache) \
            or _reroute(schedule, network, key, geometry, stop_ids, stop_radius, path_cache, access_fallback = True)
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


def read_route_shapes(schedule, routes_path, trips_path, shapes_path, modes, line_agencies = None, agencies = ()):
    """read_gtfs_shapes for the routes of a schedule that are checked (modes, agencies)."""
    checked = [
        (line.attrib["id"], route.attrib["id"]) for line, route in schedule.routes()
        if (route.findtext("transportMode") or "").strip().lower() in modes
        and (not agencies or line_agencies.get(line.attrib["id"]) in agencies)
    ]
    return read_gtfs_shapes(routes_path, trips_path, shapes_path, [r for _, r in checked], [l for l, _ in checked])


def run(schedule_path, network_path, shapefile_path, report_path, corrected_schedule_path = None,
        patched_network_path = None, modes = ("bus",), fallback_modes = ("car",), buffer_m = 30.0,
        min_share = 0.9, stop_radius = 80.0, add_links = True, log = print,
        gtfs_routes_path = None, agencies = (), gtfs_trips_path = None, gtfs_shapes_path = None,
        input_geometry_path = None, output_geometry_path = None):
    """Checks (and with corrected_schedule_path also corrects) the mapped routes. If routes had
    to use links without the route mode, or links added along the reference route, a network with
    the mode added / the new links is written to patched_network_path.
    Returns (rows, number of network changes: links patched + links added).
    agencies (with gtfs_routes_path, the GTFS routes.txt) restricts the check to the lines of these
    agencies. With gtfs_trips_path and gtfs_shapes_path (the GTFS trips.txt and shapes.txt), a route is
    compared with the shape of its trip; the official line routes of shapefile_path (optional; one path, {agency_id:
    path} or a list, see line_route_sources) are used for the routes without one."""
    agencies = set(map(str, agencies or ()))
    line_agencies = None
    if agencies or any(agency for agency, _ in line_route_sources(shapefile_path)):
        assert gtfs_routes_path, "gtfs_routes_path is needed to filter the lines by agency"
        line_agencies = read_line_agencies(gtfs_routes_path)

    log("Reading schedule ...")
    schedule = Schedule(schedule_path)

    trip_shapes, line_shapes, shapes = {}, {}, {}
    if gtfs_trips_path and gtfs_shapes_path:
        assert gtfs_routes_path, "gtfs_routes_path is needed to find the GTFS shapes of the lines"
        log("Reading the GTFS shapes of the routes ...")
        trip_shapes, line_shapes, shapes = read_route_shapes(
            schedule, gtfs_routes_path, gtfs_trips_path, gtfs_shapes_path, set(modes), line_agencies, agencies)
        log("  -> %d shape(s): of the trip of %d route(s), of %d line(s)" % (len(shapes), len(trip_shapes), len(line_shapes)))
    assert shapefile_path or shapes, "PT route check: no reference routes (neither a shapefile nor GTFS shapes)"
    corridors = Corridors(shapefile_path, buffer_m, trip_shapes, shapes, line_shapes)

    log("Reading network inside the area of the reference routes ...")
    network = Network(
        network_path, corridors.bounds, set(modes),
        fallback_modes = set(fallback_modes) if corrected_schedule_path else set(),
    )
    log("  -> %d links (%d usable for re-routing)" % (len(network.links), len(network.routing_ids)))
    network.access_shapes = list(shapes.values())

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
            if input_geometry_path and output_geometry_path:
                from matsim.scenario.network.utils.link_geometry import copy_link_geometries_with_additions
                copy_link_geometries_with_additions(
                    input_geometry_path, output_geometry_path,
                    {link_id: network.segments[link_id] for link_id in new_links},
                    network._reverse_links,
                )
    return rows, n_patched


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--schedule", required = True)
    parser.add_argument("--network", required = True)
    parser.add_argument("--shapefile", action = "append", default = [],
                        help = "Official line routes (for the routes without a GTFS shape): PATH, or AGENCY_ID=PATH to use "
                               "them for the lines of one GTFS agency only (repeatable)")
    parser.add_argument("--report", required = True)
    parser.add_argument("--corrected-schedule", default = None,
                         help = "If given, deviating routes are re-routed and the schedule written here")
    parser.add_argument("--patched-network", default = None,
                         help = "Where to write the network with the bus mode added to links used by corrected routes")
    parser.add_argument("--no-new-links", action = "store_true",
                         help = "Do not add links along the official route where the network has no connection")
    parser.add_argument("--gtfs-routes", default = None, help = "GTFS routes.txt (to filter by agency)")
    parser.add_argument("--gtfs-trips", default = None, help = "GTFS trips.txt (with --gtfs-shapes: reference = trip shapes)")
    parser.add_argument("--gtfs-shapes", default = None, help = "GTFS shapes.txt")
    parser.add_argument("--agency", action = "append", default = [],
                         help = "Only check the lines of this GTFS agency_id (repeatable)")
    parser.add_argument("--buffer", type = float, default = 30.0)
    parser.add_argument("--min-share", type = float, default = 0.9)
    args = parser.parse_args()

    shapefiles = [tuple(s.split("=", 1)) if "=" in s else (None, s) for s in args.shapefile]
    run(args.schedule, args.network, shapefiles, args.report, args.corrected_schedule,
        patched_network_path = args.patched_network, buffer_m = args.buffer, min_share = args.min_share,
        add_links = not args.no_new_links, gtfs_routes_path = args.gtfs_routes, agencies = args.agency,
        gtfs_trips_path = args.gtfs_trips, gtfs_shapes_path = args.gtfs_shapes)
