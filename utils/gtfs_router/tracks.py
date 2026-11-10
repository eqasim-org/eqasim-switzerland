"""The real path of the vehicles between consecutive stops, and the fare zones these paths cross.

The GTFS timetable only gives the stops, so a ride would be drawn as straight lines between them. Here each pair of
consecutive stops gets a track:

- trains: the shortest path on the SBB rail lines (linie-mit-polygon: lines between "operating points", the stations
  and junctions of the railway), between the operating points near the two stations. If that path is much longer than
  the straight line, it is not trusted and a straight line is kept;
- buses: the road links of the MATSim route with the same stops (corrected_schedule.xml.gz and corrected_network.xml.gz
  of eqasim-switzerland);
- everything else (tram, boat...) and the pairs not found: a straight line.

A "segment" is the track of one pair of stops, shared by all the trips going from the first to the second. For each
segment, the zones it crosses are found by cutting it with the zone polygons (data.pt_pricing.zones), which is what the pricing uses
to know the zones of a ride, and its length is the distance of the ride.

This is computed once for the whole feed (the same for all its dates) and kept in the cache. All the geometry is done
in the Swiss projection (LV95, EPSG:2056, in metres); the drawn tracks are given back in latitude and longitude.
"""
import gzip
import hashlib
import logging
import os
import pickle
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


logger = logging.getLogger("synpp")
TO_LV95 = Transformer.from_crs(4326, 2056, always_xy = True)    # (lon, lat) -> Swiss coordinates in metres
TO_WGS84 = Transformer.from_crs(2056, 4326, always_xy = True)   # and back
# how far a station looks for its operating points: 400 m, or 2 km if the station is within 150 m of a rail line (a
# long station whose operating point is at one end)
STATION_RADIUS_M = (400, 150, 2000)
VERSION = 9                 # change it when the tracks computed change, so that the old cache files are not used
DRAW_TOLERANCE_M = 3        # the tracks drawn on the map are simplified to 3 m (not those used for the zones)
CANDIDATES = 6              # the operating points tried for each station (the nearest can be on another line)
RAIL_DETOUR = (1.5, 2000)   # a rail path is kept if shorter than 1.5 × the straight line + 2 km
BUS_DETOUR = (3.0, 1000)    # a MATSim bus path is kept if shorter than 3 × the straight line + 1 km


@dataclass
class Tracks:
    """The tracks of all the segments, and the pieces of zones each one crosses.

    The arrays are flat: the points of segment s are at seg_start[s] to seg_start[s + 1] in lat and lon, and the zone
    pieces it crosses at x_start[s] to x_start[s + 1] in the x_ arrays."""
    t_seg: np.ndarray           # the segment from each stop of each trip to the next one (aligned with network.t_stop; -1 at the last stop)
    seg_start: np.ndarray       # where the points of each segment begin in lat, lon
    lat: np.ndarray
    lon: np.ndarray
    seg_m: np.ndarray           # the length of each segment, in metres
    seg_source: np.ndarray      # where each segment comes from: 0 a straight line, 1 the rail lines, 2 a MATSim bus route
    x_start: np.ndarray         # where the zone pieces of each segment begin in the x_ arrays, in order along the segment
    x_network: np.ndarray       # the fare network and zone of each piece, its length in metres, and where it starts
    x_zone: np.ndarray
    x_m: np.ndarray
    x_from: np.ndarray          # (metres from the start of its segment)
    zones: gpd.GeoDataFrame     # the zone polygons (EPSG:2056), those of data.pt_pricing.zones

    def segments(self, net, ride):
        """The segments of a ride of the router, in order."""
        begin = net.t_start[ride.trip] + ride.pos
        return self.t_seg[begin: begin + len(ride.stops) - 1]

    def length_m(self, net, ride):
        """The length of a ride along its track, in metres."""
        return float(self.seg_m[self.segments(net, ride)].sum())

    def crossed(self, net, ride):
        """The zones a ride goes through and how far: {network: {zone: metres}}."""
        result = {}
        for s in self.segments(net, ride):
            for k in range(self.x_start[s], self.x_start[s + 1]):
                zones = result.setdefault(self.x_network[k], {})
                zones[self.x_zone[k]] = zones.get(self.x_zone[k], 0.0) + float(self.x_m[k])
        return result

    def pieces(self, net, ride, network):
        """The zones of one network along a ride, in order: [(zone, from, to)], in metres from the start of the ride."""
        result, offset = [], 0.0
        for s in self.segments(net, ride):
            for k in range(self.x_start[s], self.x_start[s + 1]):
                if self.x_network[k] == network:
                    result.append((self.x_zone[k], offset + float(self.x_from[k]), offset + float(self.x_from[k] + self.x_m[k])))
            offset += float(self.seg_m[s])
        return result

    def path(self, net, ride):
        """The track of a ride to draw: [[lat, lon], ...]."""
        points = []
        for s in self.segments(net, ride):
            part = np.column_stack([self.lat[self.seg_start[s]:self.seg_start[s + 1]], self.lon[self.seg_start[s]:self.seg_start[s + 1]]])
            points += part[1:].tolist() if points else part.tolist()   # the first point is the last of the segment before
        return points


def build_tracks(net, lines_path, zones, zones_path, schedule_path = None, matsim_network_path = None, cache_dir = None):
    """The Tracks of a network (the whole feed).

    lines_path: the SBB rail lines (linie-mit-polygon.geojson). zones: the zone polygons of data.pt_pricing.zones
    (network, zone, geometry, municipalities, name; EPSG:2056), and zones_path the pt_zones.gpkg it wrote, which tells
    the cache when they change. schedule_path, matsim_network_path: the MATSim schedule and network, for the bus routes
    (optional). cache_dir: where the result is kept, and found again as long as the network and all these files are the
    same."""
    inputs = [p for p in (lines_path, zones_path, schedule_path, matsim_network_path) if p]
    if cache_dir:
        stamp = hashlib.md5(net.stop_ids.tobytes() + net.t_start.tobytes() + net.t_stop.tobytes()).hexdigest()
        stamp += f"|{VERSION}|" + "|".join(f"{os.path.abspath(p)}:{os.path.getmtime(p)}" for p in inputs)
        cache = os.path.join(cache_dir, f"tracks_{net.date or 'feed'}_{hashlib.md5(stamp.encode()).hexdigest()[:10]}.pkl")
        if os.path.exists(cache):
            logger.info(f"Tracks from the cache {cache}")
            with open(cache, "rb") as f:
                return pickle.load(f)

    logger.info("Tracks of the vehicles")
    x, y = TO_LV95.transform(net.stop_lon, net.stop_lat)
    xy = np.column_stack([x, y])

    # the pairs of consecutive stops (a, b) of all the trips, each once, and the segment of each stop of each trip;
    # a pair is "rail" if a train runs between its stops
    nxt = np.arange(len(net.t_stop)) + 1
    last = np.zeros(len(net.t_stop), bool)
    last[net.t_start[1:] - 1] = True
    trip_of = np.repeat(np.arange(len(net.trip_ids)), np.diff(net.t_start))
    keys = np.where(last, -1, net.t_stop.astype(np.int64) * len(net.stop_ids) + net.t_stop[np.minimum(nxt, len(nxt) - 1)])
    pairs, t_seg = np.unique(keys[~last], return_inverse = True)
    seg_of = np.full(len(net.t_stop), -1, np.int64)
    seg_of[~last] = t_seg
    a, b = pairs // len(net.stop_ids), pairs % len(net.stop_ids)
    mode = net.routes["mode"].to_numpy()[net.trip_route[trip_of[~last]]]
    rail = np.zeros(len(pairs), bool)
    rail[np.unique(t_seg[mode == "rail"])] = True
    logger.info(f"  {len(pairs)} pairs of stops, {rail.sum()} by rail")

    geometry = [None] * len(pairs)   # the track of each pair (coordinates), None while not found
    source = np.zeros(len(pairs), np.int8)
    beeline = np.hypot(*(xy[a] - xy[b]).T)

    # trains: the rail lines as a graph whose nodes are the operating points and whose edges are the lines between them
    lines = gpd.read_file(lines_path).to_crs(2056)
    lines = lines[lines["bp_anfang"] != lines["bp_ende"]]
    points = pd.Index(pd.unique(pd.concat([lines["bp_anfang"], lines["bp_ende"]])))
    u, v = points.get_indexer(lines["bp_anfang"]), points.get_indexer(lines["bp_ende"])
    length = lines.geometry.length.to_numpy()
    links = pd.DataFrame({"u": np.concatenate([u, v]), "v": np.concatenate([v, u]), "m": np.concatenate([length, length])}).groupby(["u", "v"])["m"].min()
    graph = coo_matrix((links.to_numpy(), (links.index.get_level_values(0), links.index.get_level_values(1))), shape = (len(points),) * 2).tocsr()
    edge = {}   # (from, to) -> the coordinates of the shortest line between them, in that direction
    for i, j, geom, d in zip(u, v, lines.geometry, length):
        coords = np.asarray(geom.coords)[:, :2]
        if (i, j) not in edge or d < _line_m(edge[(i, j)]):
            edge[(i, j)], edge[(j, i)] = coords, coords[::-1]
    # the position of each operating point: the average of the ends of its lines
    point_xy = np.zeros((len(points), 2))
    np.add.at(point_xy, u, np.asarray([g.coords[0][:2] for g in lines.geometry]))
    np.add.at(point_xy, v, np.asarray([g.coords[-1][:2] for g in lines.geometry]))
    point_xy /= np.bincount(np.concatenate([u, v]), minlength = len(points))[:, None]
    # the candidate operating points of each station: several, because the nearest one can be on another line
    # (Dietikon: the BDWM; Bern: the RBS)
    near, line, far = STATION_RADIUS_M
    near_line = lines.sindex.nearest(gpd.points_from_xy(xy[:, 0], xy[:, 1]), return_all = False, return_distance = True)[1]
    radius = np.where(near_line <= line, far, near)
    distance, nearest = cKDTree(point_xy).query(xy, k = CANDIDATES, distance_upper_bound = far)
    ok = (distance <= radius[:, None]) & (nearest < len(points))
    has_op = ok.any(axis = 1)

    # for each rail pair, the shortest path between any candidate of a and any candidate of b (plus the distances from
    # the stations to these points); the shortest paths from all the candidates are computed at once (Dijkstra)
    todo = np.flatnonzero(rail & has_op[a] & has_op[b])
    sources = np.unique(nearest[a[todo]][ok[a[todo]]])
    dist, pred = dijkstra(graph, indices = sources, return_predecessors = True)
    row = pd.Series(np.arange(len(sources)), index = sources)
    found, too_long, same = 0, 0, 0
    for k in todo:
        best = (np.inf, -1, -1)
        for da, ca in zip(distance[a[k]][ok[a[k]]], nearest[a[k]][ok[a[k]]]):
            for db, cb in zip(distance[b[k]][ok[b[k]]], nearest[b[k]][ok[b[k]]]):
                cost = dist[row[ca], cb] + da + db
                if ca != cb and cost < best[0]:
                    best = (cost, ca, cb)
        cost, ca, cb = best
        if ca < 0:   # both stations at the same operating point
            same += 1
            continue
        if cost > beeline[k] * RAIL_DETOUR[0] + RAIL_DETOUR[1]:
            too_long += 1
            continue
        chain, r = [cb], row[ca]   # the operating points of the path, followed back from b
        while chain[-1] != ca:
            chain.append(pred[r, chain[-1]])
        chain.reverse()
        coords = _join([xy[a[k]]] + [edge[(p, q)] for p, q in zip(chain, chain[1:])] + [xy[b[k]][None]])
        geometry[k], source[k] = np.asarray(coords), 1
        found += 1
    logger.info(f"  rail: {found} of {rail.sum()} pairs on the lines; straight: {(rail & ~(has_op[a] & has_op[b])).sum()} with a stop far from "
                f"them, {same} at the same operating point, {too_long} without a path or with a detour")

    # buses: the MATSim routes
    if schedule_path and matsim_network_path:
        found = _matsim(net, schedule_path, matsim_network_path, pairs, a, b, xy, beeline, geometry, source)
        logger.info(f"  MATSim: {found} pairs")

    for k in range(len(pairs)):   # the rest: straight lines
        if geometry[k] is None:
            geometry[k] = np.array([xy[a[k]], xy[b[k]]])
    tracks = _assemble(net, zones, seg_of, geometry, source)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok = True)
        with open(cache, "wb") as f:
            pickle.dump(tracks, f, protocol = pickle.HIGHEST_PROTOCOL)
    return tracks


def _join(parts):
    """The coordinates of consecutive lines joined into one, each line turned round if needed to start where the
    previous one ends (some lines are drawn from their end to their start)."""
    coords = [np.atleast_2d(parts[0])]
    for part in parts[1:]:
        end = coords[-1][-1]
        if len(part) > 1 and np.hypot(*(part[-1] - end)) < np.hypot(*(part[0] - end)):
            part = part[::-1]
        coords.append(part)
    return np.concatenate(coords)


def _line_m(coords):
    """The length of a line given by its coordinates, in metres."""
    return float(np.hypot(*np.diff(coords, axis = 0).T).sum())


def _matsim(net, schedule_path, network_path, pairs, a, b, xy, beeline, geometry, source):
    """The tracks of the MATSim bus routes, for the pairs of stops that have none yet. Fills geometry and source, and
    returns the number of pairs found.

    A MATSim route is a list of stops and the list of road links it drives along. Between two of its consecutive stops,
    the track goes through the end nodes of the links between the link of the first stop and that of the second."""
    stop_index = pd.Series(np.arange(len(net.stop_ids)), index = net.stop_ids)
    pair_index = pd.Series(np.arange(len(pairs)), index = pairs)
    # from the schedule: the stop facilities (their link and position) and the bus routes (their stops and links)
    facilities, routes = {}, []
    for _, el in ET.iterparse(gzip.open(schedule_path), events = ("end",)):
        if el.tag == "stopFacility":
            facilities[el.get("id")] = (el.get("linkRefId"), float(el.get("x")), float(el.get("y")))
        elif el.tag == "transitRoute":
            route = el.find("route")
            if el.findtext("transportMode") == "bus" and route is not None:
                routes.append(([s.get("refId") for s in el.find("routeProfile")], [l.get("refId") for l in route]))
            el.clear()
    # from the network: the end node of each link used, and the positions of the nodes
    needed = {l for _, links in routes for l in links}
    link_to, nodes = {}, {}
    for _, el in ET.iterparse(gzip.open(network_path), events = ("end",)):
        if el.tag == "node":
            nodes[el.get("id")] = (float(el.get("x")), float(el.get("y")))
            el.clear()
        elif el.tag == "link":
            if el.get("id") in needed:
                link_to[el.get("id")] = el.get("to")
            el.clear()
    found = 0
    for stops, links in routes:
        ids = [f.split(".link:")[0] for f in stops]
        if not all(i in stop_index.index for i in ids):   # a route with a stop the timetable does not have
            continue
        idx = stop_index[ids].to_numpy()
        keys = idx[:-1].astype(np.int64) * len(net.stop_ids) + idx[1:]
        at, position = 0, []
        for f in stops:   # the position of the link of each stop in the route
            link = facilities[f][0]
            while at < len(links) and links[at] != link:
                at += 1
            position.append(at)
        for k, key in enumerate(keys):
            if key not in pair_index.index:
                continue
            p = pair_index[key]
            if geometry[p] is not None or position[k + 1] >= len(links):
                continue
            inner = [nodes[link_to[l]] for l in links[position[k] + 1: position[k + 1]] if l in link_to]
            coords = np.array([facilities[stops[k]][1:]] + inner + [facilities[stops[k + 1]][1:]])
            if _line_m(coords) <= beeline[p] * BUS_DETOUR[0] + BUS_DETOUR[1]:
                geometry[p], source[p] = coords, 2
                found += 1
    return found


def _assemble(net, zones, seg_of, geometry, source):
    """The Tracks of the segments: their lengths, the zones they cross (cut with the zone polygons), and their
    simplified coordinates for the map."""
    zones = zones[["network", "zone", "geometry", "municipalities", "name"]].reset_index(drop = True)
    lines = gpd.GeoDataFrame(geometry = [shapely.LineString(g) for g in geometry], crs = 2056)
    hits = gpd.sjoin(lines, zones[["network", "zone", "geometry"]], predicate = "intersects")
    inside = shapely.intersection(lines.geometry.values[hits.index.to_numpy()], zones.geometry.values[hits["index_right"].to_numpy()])
    # each continuous piece of a segment in a zone, with where it starts along the segment (a segment can leave a zone
    # and come back: two pieces)
    parts, which = shapely.get_parts(shapely.line_merge(inside), return_index = True)
    keep = shapely.length(parts) > 0
    parts, which = parts[keep], which[keep]
    seg = hits.index.to_numpy()[which]
    starts = shapely.line_locate_point(lines.geometry.values[seg], shapely.get_point(parts, 0))
    ends = shapely.line_locate_point(lines.geometry.values[seg], shapely.get_point(parts, -1))
    hits = pd.DataFrame({"seg": seg, "network": hits["network"].to_numpy()[which], "zone": hits["zone"].to_numpy()[which],
                         "start": np.minimum(starts, ends), "m": shapely.length(parts)}).sort_values(["seg", "start"])
    x_start = np.searchsorted(hits["seg"].to_numpy(), np.arange(len(geometry) + 1))
    drawn = [shapely.get_coordinates(shapely.simplify(g, DRAW_TOLERANCE_M)) for g in lines.geometry.values]   # for the map
    sizes = np.array([len(g) for g in drawn])
    seg_start = np.concatenate([[0], np.cumsum(sizes)])
    allxy = np.concatenate(drawn)
    lon, lat = TO_WGS84.transform(allxy[:, 0], allxy[:, 1])
    logger.info(f"  {len(geometry)} segments: {(source == 1).sum()} rail lines, {(source == 2).sum()} MATSim, {(source == 0).sum()} straight; "
                f"{len(hits)} zone crossings")
    return Tracks(t_seg = seg_of, seg_start = seg_start, lat = lat.astype(np.float32), lon = lon.astype(np.float32),
                  seg_m = lines.geometry.length.to_numpy(), seg_source = source, x_start = x_start,
                  x_network = hits["network"].to_numpy(str), x_zone = hits["zone"].to_numpy(str), x_m = hits["m"].to_numpy(),
                  x_from = hits["start"].to_numpy(), zones = zones)
