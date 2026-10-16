import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import geopandas as gpd
import shapely
import xopen
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

import matsim.scenario.network.utils.pt_route_check as pt_route_check
from analysis.pt.transit_schedule import page

NETWORK_CRS = "epsg:2056"

TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

MODE_COLORS = {
    "rail": "#2374b6",
    "train": "#2374b6",
    "tram": "#983572",
    "bus": "#25A4AB",
    "subway": "#984ea3",
    "metro": "#984ea3",
    "ferry": "#eb6834",
    "boat": "#eb6834",
    "funicular": "#4daf4a",
    "gondola": "#f2a71b",
    "cablecar": "#f2a71b",
}

DEFAULT_MODE_COLOR    = "#888888"
BUS_LANE_COLOR        = "#190387"
DEFAULT_VISIBLE_MODES = {"bus"}
LINE_ROUTE_COLOR      = "#ff8c00"
CHECK_OK_COLOR        = "#2a9d3a"
CHECK_DEVIATES_COLOR  = "#c0392b"
DEFAULT_LINE_ROUTES   = "PT_line_routes/Geneva/TPG_LIGNES.shp"
SELECTED_COLOR        = "#e34948"

TO_WGS84 = Transformer.from_crs(NETWORK_CRS, "epsg:4326", always_xy = True)


def configure(context):
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("analysis.pt.transit_schedule_modes", default = [])
    context.config("analysis.pt.transit_schedule_show_schedules", default = True)
    context.config("extent_path", default = "")
    context.config("data_path")
    context.config("analysis.pt.transit_schedule_line_routes_path", default = "PT_line_routes/Geneva/TPG_LIGNES.shp")
    context.config("analysis.pt.transit_schedule_source", default = "simulation")

    # Compares the routes with the official line routes (report in route_check.csv)
    context.config("analysis.pt.transit_schedule_route_check", default = True)
    context.config("pt_route_modes", ["bus"])
    context.config("pt_route_buffer", 30.0)
    context.config("pt_route_min_share", 0.9)
    context.config("pt_route_agencies", ["881"])
    if context.config("analysis.pt.transit_schedule_route_check") and context.config("pt_route_agencies"):
        context.stage("data.gtfs.cleaned")

    source = context.config("analysis.pt.transit_schedule_source")
    if source in ("mapped", "corrected"):
        context.stage("matsim.scenario.network.%s" % source)
    elif source != "simulation":
        raise RuntimeError(
            "analysis.pt.transit_schedule_source must be 'simulation', 'mapped' or 'corrected', got: %s" % source
        )


def execute(context):
    modes = set(context.config("analysis.pt.transit_schedule_modes")) or None
    show_schedules = context.config("analysis.pt.transit_schedule_show_schedules")
    source = context.config("analysis.pt.transit_schedule_source")

    if source in ("mapped", "corrected"):
        mapped = context.stage("matsim.scenario.network.%s" % source)
        schedule_path = mapped["schedule"]
        network_path = mapped["network"]
        output_dir = context.path()
    else:
        matsim_output_folder = os.path.join(
            context.config("output_path"),
            context.config("output_id"),
            context.config("simulation_directory"),
        )

        if not os.path.isdir(matsim_output_folder):
            raise FileNotFoundError(
                f"MATSim output not found at {matsim_output_folder} - "
                "has the simulation been run for this output_id?"
            )

        schedule_path = _find_output_file(matsim_output_folder, "output_transitSchedule.xml.gz")
        network_path = _find_output_file(matsim_output_folder, "output_network.xml.gz")
        output_dir = os.path.join(matsim_output_folder, "pt_transit_schedule")

    # Absolute, or relative to data_path
    line_routes_path = os.path.join(
        context.config("data_path"),
        context.config("analysis.pt.transit_schedule_line_routes_path") or DEFAULT_LINE_ROUTES,
    )
    if not os.path.exists(line_routes_path):
        print(f"WARNING: official line routes not found at {line_routes_path}: "
              "they are not shown on the map and the routes are not checked")
        line_routes_path = None

    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "transit_schedule_map.html")

    route_check_rows = None
    report_path = None
    if line_routes_path and context.config("analysis.pt.transit_schedule_route_check"):
        agencies = [str(a) for a in context.config("pt_route_agencies")]
        report_path = os.path.join(output_dir, "route_check.csv")
        route_check_rows, _ = pt_route_check.run(
            schedule_path = schedule_path, network_path = network_path,
            shapefile_path = line_routes_path, report_path = report_path,
            modes = tuple(context.config("pt_route_modes")),
            buffer_m = context.config("pt_route_buffer"), min_share = context.config("pt_route_min_share"),
            agencies = agencies,
            gtfs_routes_path = "%s/output/routes.txt" % context.path("data.gtfs.cleaned") if agencies else None,
        )

    plot_transit_schedule(
        schedule_path = schedule_path, network_path = network_path,
        output_path = output_file, modes = modes, show_schedules = show_schedules,
        extent_path = context.config("extent_path"), line_routes_path = line_routes_path,
        route_check_rows = route_check_rows,
    )

    return dict(done = True, path = output_file, route_check = report_path)


def _find_output_file(simulation_path, filename):
    direct_path = os.path.join(simulation_path, filename)
    if os.path.exists(direct_path):
        return direct_path

    stem, _, suffix = filename.partition(".")
    candidates = glob.glob(os.path.join(simulation_path, "ITERS", "*", f"*.{suffix}"))
    if not candidates:
        raise FileNotFoundError(f"Could not find {filename} in {simulation_path} (or its ITERS folder)")

    candidate = max(candidates, key=os.path.getmtime)
    print(f"{filename} not found at {direct_path}, using latest iteration snapshot {candidate} instead")
    return candidate


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def load_network_geometry(network_path):
    node_coords = {}
    link_node_ids = {}
    link_modes = {}

    with xopen.xopen(network_path, "r") as f:
        for _, elem in ET.iterparse(f, events=["end"]):
            if elem.tag == "node":
                node_coords[elem.attrib["id"]] = (float(elem.attrib["x"]), float(elem.attrib["y"]))
            elif elem.tag == "link":
                link_id = elem.attrib["id"]
                link_node_ids[link_id] = (elem.attrib["from"], elem.attrib["to"])
                link_modes[link_id] = elem.attrib.get("modes", "")
            elem.clear()

    link_geometry = {
        link_id: (node_coords[from_node], node_coords[to_node])
        for link_id, (from_node, to_node) in link_node_ids.items()
        if from_node in node_coords and to_node in node_coords
    }
    return link_geometry, link_modes


def is_bus_reserved_link(modes_string):
    modes = set(modes_string.split(","))
    # pt2matsim "artificial" links only connect stops that could not be mapped to the road
    # network (they can be kilometres long); they are not real bus lanes.
    if modes & {"artificial", "stopFacilityLink"}:
        return False
    return bool(modes & {"bus", "pt"}) and "car" not in modes


def load_extent(extent_path):
    if not extent_path:
        return None
    if not os.path.exists(extent_path):
        raise FileNotFoundError(f"extent_path {extent_path} does not exist")

    boundary = gpd.read_file(extent_path)
    if boundary.crs is not None and boundary.crs != NETWORK_CRS:
        boundary = boundary.to_crs(NETWORK_CRS)
    return shapely.force_2d(unary_union(boundary.geometry))


def load_line_routes(line_routes_path, region = None):
    """Official line routes by normalized line name: list of dicts with the terminus names
    the route runs towards ("DIRECTION", possibly several separated by "/") and its paths."""
    gdf = gpd.read_file(line_routes_path)
    gdf = gdf.to_crs(epsg = 4326) if gdf.crs is not None else gdf.set_crs(epsg = 4326)

    if region is not None:
        region_4326 = gpd.GeoSeries([region], crs = NETWORK_CRS).to_crs(epsg = 4326).iloc[0]
        gdf = gdf[gdf.intersects(region_4326)]

    line_routes = defaultdict(list)
    geometries_2056 = gdf.geometry.to_crs(NETWORK_CRS)
    for line, direction, geometry, geometry_2056 in zip(*pt_route_check.read_line_columns(gdf), gdf.geometry, geometries_2056):
        if geometry is None or geometry.is_empty:
            continue
        parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
        line_routes[pt_route_check.normalize_line(line)].append({
            "directions": {pt_route_check.normalize_name(d) for d in str(direction).split("/")},
            "centerline": shapely.force_2d(geometry_2056),
            "paths": [[[c[1], c[0]] for c in part.coords] for part in parts],
        })
    return line_routes


def _parse_hms(value):
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_transit_schedule(schedule_path):
    stops = {}
    routes = []

    current_line_id = None
    current_line_name = None
    current_route_id = None
    current_mode = None
    current_stop_ids = []
    current_stop_offsets = []
    current_link_ids = []
    current_departure_times = []
    in_route_profile = False
    in_route = False
    in_departures = False

    with xopen.xopen(schedule_path, "r") as f:
        for event, elem in ET.iterparse(f, events=["start", "end"]):
            tag = elem.tag

            if event == "start":
                if tag == "transitLine":
                    current_line_id = elem.attrib["id"]
                    current_line_name = elem.attrib.get("name") or current_line_id
                elif tag == "transitRoute":
                    current_route_id = elem.attrib["id"]
                    current_mode = None
                    current_stop_ids = []
                    current_stop_offsets = []
                    current_link_ids = []
                    current_departure_times = []
                elif tag == "routeProfile":
                    in_route_profile = True
                elif tag == "route":
                    in_route = True
                elif tag == "departures":
                    in_departures = True
                continue

            # event == "end" from here on
            if tag == "stopFacility":
                stop_id = elem.attrib["id"]
                stops[stop_id] = {
                    "x": float(elem.attrib["x"]),
                    "y": float(elem.attrib["y"]),
                    "name": elem.attrib.get("name") or stop_id,
                    "lines": stops.get(stop_id, {}).get("lines", set()),
                }
            elif tag == "stop" and in_route_profile:
                ref_id = elem.attrib["refId"]
                current_stop_ids.append(ref_id)
                arrival = elem.attrib.get("arrivalOffset")
                departure = elem.attrib.get("departureOffset")
                current_stop_offsets.append((
                    _parse_hms(arrival) if arrival is not None else None,
                    _parse_hms(departure) if departure is not None else None,
                ))
                stops.setdefault(ref_id, {"x": None, "y": None, "name": None, "lines": set()})
                if current_line_name:
                    stops[ref_id]["lines"].add(current_line_name)
            elif tag == "transportMode" and current_route_id is not None:
                current_mode = (elem.text or "").strip().lower()
            elif tag == "link" and in_route:
                current_link_ids.append(elem.attrib["refId"])
            elif tag == "departure" and in_departures:
                current_departure_times.append(_parse_hms(elem.attrib["departureTime"]))
            elif tag == "routeProfile":
                in_route_profile = False
            elif tag == "route":
                in_route = False
            elif tag == "departures":
                in_departures = False
            elif tag == "transitRoute":
                routes.append({
                    "line_id": current_line_id, "line_name": current_line_name,
                    "route_id": current_route_id, "mode": current_mode or "unknown",
                    "stop_ids": list(current_stop_ids),
                    "stop_offsets": list(current_stop_offsets),
                    "link_ids": list(current_link_ids),
                    "departure_times": list(current_departure_times),
                })
                current_route_id = None
            elif tag == "transitLine":
                current_line_id = None
                current_line_name = None

            elem.clear()

    return stops, routes


def _extract_station_code(stop_id):
    # pt2matsim appends ".link:<linkId>" to stop ids; the link id can itself be a 7-8 digit
    # number and must not be mistaken for the station code (it would merge unrelated stops).
    matches = re.findall(r"\d{7,8}", str(stop_id).split(".link:")[0])
    if not matches:
        return None
    code = matches[-1]
    if len(code) == 8:
        code = code[:7]  # drop UIC check digit
    return code



# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _route_path(route, link_geometry, stops):
    points = []
    for link_id in route["link_ids"]:
        endpoints = link_geometry.get(link_id)
        if endpoints is None:
            continue
        from_xy, to_xy = endpoints
        if not points:
            points.append(from_xy)
        points.append(to_xy)

    if len(points) >= 2:
        return points

    fallback = [
        (stops[stop_id]["x"], stops[stop_id]["y"])
        for stop_id in route["stop_ids"]
        if stop_id in stops and stops[stop_id]["x"] is not None
    ]
    return fallback if len(fallback) >= 2 else None


MIN_VARIANT_SHARE = 0.25  # keep a branch if it has at least this share of the busiest branch's departures


def select_representative_routes(routes, stops):
    """One route per line + direction + branch.

    Routes of a line are grouped by (line, origin, destination); inside a group, routes are
    split into branches by their sequence of stations. A branch is kept when it carries at
    least MIN_VARIANT_SHARE of the departures of the busiest branch, so that real branches
    (e.g. TPG line 1 via Belle-Terre or via Belle-Idée) are all selectable while rare
    variants (night or one-off services) are ignored. Kept branches beyond the first get a
    "via" hint naming a stop the other branches do not serve.
    """
    groups = defaultdict(list)
    for route in routes:
        if len(route["stop_ids"]) < 2:
            continue
        origin_name = stops.get(route["stop_ids"][0], {}).get("name") or route["stop_ids"][0]
        dest_name   = stops.get(route["stop_ids"][-1], {}).get("name") or route["stop_ids"][-1]
        key = (route["line_name"] or route["line_id"], origin_name, dest_name)
        groups[key].append(route)

    def stop_name(stop_id):
        return stops.get(stop_id, {}).get("name") or stop_id

    representatives = []
    for group_routes in groups.values():
        branches = defaultdict(list)
        for route in group_routes:
            branches[tuple(_extract_station_code(sid) or sid for sid in route["stop_ids"])].append(route)

        ranked = sorted(
            branches.items(),
            key = lambda item: -sum(len(r["departure_times"]) for r in item[1]),
        )
        top_departures = sum(len(r["departure_times"]) for r in ranked[0][1])
        kept = [
            item for item in ranked
            if sum(len(r["departure_times"]) for r in item[1]) >= MIN_VARIANT_SHARE * top_departures
        ]

        for signature, branch_routes in kept:
            signature_freq = Counter(tuple(r["stop_ids"]) for r in branch_routes)
            best = dict(max(
                branch_routes,
                key = lambda r: (len(r["departure_times"]), signature_freq[tuple(r["stop_ids"])]),
            ))

            if len(kept) > 1:
                others = set()
                for other_signature, other_routes in kept:
                    if other_signature != signature:
                        others.update(_extract_station_code(sid) or sid for sid in other_routes[0]["stop_ids"])
                own_stops = [sid for sid in best["stop_ids"] if (_extract_station_code(sid) or sid) not in others]
                best["via"] = stop_name(own_stops[len(own_stops) // 2]) if own_stops else "fewer stops"
            representatives.append(best)
    return representatives


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def plot_transit_schedule(
    simulation_path = None,
    schedule_path = None,
    network_path = None,
    output_path = "transit_schedule_map.html",
    modes = None,
    show_schedules = True,
    extent_path = None,
    line_routes_path = None,
    route_check_rows = None,
):
    """route_check_rows: rows of pt_route_check.run (comparison with the official line routes),
    shown next to the routes in the search and the line details."""
    if schedule_path is None or network_path is None:
        if simulation_path is None:
            raise ValueError(
                "Either simulation_path, or both schedule_path and network_path, must be provided."
            )

    if schedule_path is None:
        schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    if network_path is None:
        network_path = _find_output_file(simulation_path, "output_network.xml.gz")

    print(f"Reading transit schedule from {schedule_path} ...")
    stops, routes = load_transit_schedule(schedule_path)
    print(f"  -> {len(stops)} stops, {len(routes)} transit routes")

    region = load_extent(extent_path)
    if region is not None:
        shapely.prepare(region)
        print(f"Restricting to extent {extent_path} ...")
        # Stops outside the extent lose their coordinates (so they are not plotted) but keep
        # their names, which are still needed for the origin/destination labels of schedules.
        for info in stops.values():
            if info["x"] is not None and not region.contains(Point(info["x"], info["y"])):
                info["x"] = info["y"] = None
        routes = [
            route for route in routes
            if any(stops.get(sid, {}).get("x") is not None for sid in route["stop_ids"])
        ]
        print(f"  -> {sum(info['x'] is not None for info in stops.values())} stops and {len(routes)} routes within the extent")

    if modes is not None:
        routes = [route for route in routes if route["mode"] in modes]
        print(f"  -> {len(routes)} routes after restricting to modes {sorted(modes)}")

    mode_counts = defaultdict(int)
    for route in routes:
        mode_counts[route["mode"]] += 1
    for mode, count in sorted(mode_counts.items(), key = lambda kv: -kv[1]):
        print(f"    {mode}: {count} route(s)")

    print(f"Reading network geometry from {network_path} ...")
    link_geometry, link_modes = load_network_geometry(network_path)
    if region is not None:
        link_geometry = {
            link_id: endpoints for link_id, endpoints in link_geometry.items()
            if region.intersects(LineString(endpoints))
        }
    print(f"  -> {len(link_geometry)} links")

    bus_lane_link_ids = [
        link_id for link_id, modes in link_modes.items()
        if link_id in link_geometry and is_bus_reserved_link(modes)
    ]
    print(f"  -> {len(bus_lane_link_ids)} link(s) reserved for bus/PT (excluded from car)")

    # Only plot stops served by the retained routes (drops unused pt2matsim stop copies, which
    # would otherwise skew the station position, and stops of modes excluded by `modes`).
    used_stop_ids = {sid for route in routes for sid in route["stop_ids"]}
    stop_ids_with_coords = [
        sid for sid, info in stops.items() if info["x"] is not None and sid in used_stop_ids
    ]
    if not stop_ids_with_coords:
        raise RuntimeError("No stop coordinates found in the transit schedule (within the extent); nothing to plot.")

    print("Grouping stops across sources by station code...")
    group_key_of = {sid: (_extract_station_code(sid) or sid) for sid in stop_ids_with_coords}
    group_members = defaultdict(set)
    for sid in stop_ids_with_coords:
        group_members[group_key_of[sid]].add(sid)
    group_keys = list(group_members)
    group_index = {key: k for k, key in enumerate(group_keys)}
    print(f"  -> {len(group_keys)} station(s) after grouping ({len(stop_ids_with_coords)} platform(s)/stop id(s) before)")

    print("Selecting one representative route per line + direction + branch...")
    line_routes = select_representative_routes(routes, stops)
    print(f"  -> {len(line_routes)} representative route(s) (from {len(routes)} total patterns)")

    official_routes = {}
    if line_routes_path:
        print(f"Reading official line routes from {line_routes_path} ...")
        official_routes = load_line_routes(line_routes_path, region)
        print(f"  -> routes for {len(official_routes)} line(s)")

    print("Building the map data...")
    data = build_page_data(
        stops, routes, link_geometry, bus_lane_link_ids, group_keys, group_members, group_index, group_key_of,
        line_routes, official_routes, route_check_rows, show_schedules, region,
    )
    if show_schedules:
        print(f"  -> {sum(len(route[4]) for route in data['routes'])} departure(s) of {len(data['routes'])} route(s) "
              "(pass a smaller modes=... set, an extent or show_schedules=False if the output gets too large)")

    print("Saving map...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok = True)
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(page.page("Transit schedule", data))
    print(f"Done - saved to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def build_page_data(stops, routes, link_geometry, bus_lane_link_ids, group_keys, group_members, group_index,
                    group_key_of, line_routes, official_routes, route_check_rows, show_schedules, region):
    """The data of the page (see page.py), coordinates in WGS84:
    - stops: one per station [name, lat, lon, modes]
    - routes: every route [line, mode, origin, destination, departure times (s), [[stop, offset (s)], ...], representative]
      (stop -1: outside the extent), from which the page computes the timetable of every stop
    - reps: the representative routes (one per line, direction and branch) with their path on the network
    - links: the links used by the routes [lat1, lon1, lat2, lon2, mode, lines], busLanes: [lat1, lon1, lat2, lon2]
    Names (stops, lines, termini) are indexes into `names`."""
    names, name_index = [], {}

    def name_id(name):
        name = str(name)
        if name not in name_index:
            name_index[name] = len(names)
            names.append(name)
        return name_index[name]

    def stop_name(stop_id):
        return stops.get(stop_id, {}).get("name") or stop_id

    def to_latlon(xy):
        lon, lat = TO_WGS84.transform(*zip(*xy)) if xy else ((), ())
        return [[round(a, 6), round(b, 6)] for a, b in zip(lat, lon)]

    mode_names = sorted({route["mode"] for route in routes}, key = lambda m: (m not in MODE_COLORS, m))
    mode_index = {mode: k for k, mode in enumerate(mode_names)}

    # Stations: at the mean position of their platforms
    station_modes = defaultdict(set)
    for route in routes:
        for sid in route["stop_ids"]:
            if sid in group_key_of:
                station_modes[group_index[group_key_of[sid]]].add(mode_index[route["mode"]])
    positions = to_latlon([
        (sum(stops[m]["x"] for m in group_members[key]) / len(group_members[key]),
         sum(stops[m]["y"] for m in group_members[key]) / len(group_members[key]))
        for key in group_keys
    ])
    stop_rows = []
    for k, key in enumerate(group_keys):
        member_names = [stops[m]["name"] for m in group_members[key] if stops[m]["name"]]
        label = Counter(member_names).most_common(1)[0][0] if member_names else key
        stop_rows.append([name_id(label), *positions[k], sorted(station_modes[k]) or [0]])

    # Representative routes, with their path on the network, the platforms they serve and their official routes
    route_checks = {
        (row["line_id"], row["route_id"]): {
            "status": row["status"], "share_inside": row["share_inside"],
            "n_uturns": row["n_uturns"], "reference_matched": row["reference_matched"],
        }
        for row in route_check_rows or []
    }
    references, reps, rep_of_key = {}, [], {}
    for rep in line_routes:
        path_2056 = _route_path(rep, link_geometry, stops)
        if path_2056 is None:
            continue
        line_label = rep["line_name"] or rep["line_id"]
        from_name, to_name = stop_name(rep["stop_ids"][0]), stop_name(rep["stop_ids"][-1])
        platforms = [(stops[s]["x"], stops[s]["y"]) for s in rep["stop_ids"] if stops.get(s, {}).get("x") is not None]

        # Same official route as the route check: matched on the terminus and the stops; if none
        # matches, every route of the line is shown rather than none.
        features = official_routes.get(pt_route_check.normalize_line(line_label), [])
        selected, _ = pt_route_check.select_directions(
            [f["directions"] for f in features], [f["centerline"] for f in features], to_name, shapely.points(platforms),
        ) if features and platforms else ([], False)
        reference_ids = []
        for feature in (features[i] for i in selected):
            references[str(id(feature))] = [[[round(a, 6), round(b, 6)] for a, b in path] for path in feature["paths"]]
            reference_ids.append(str(id(feature)))

        rep_of_key[_route_key(rep, stops)] = len(reps)
        rep_of_key.setdefault(_route_key(rep, stops)[:3], len(reps))
        reps.append({
            "label": f"{line_label} ({rep['mode']}): {from_name} → {to_name}" + (f" [via {rep['via']}]" if rep.get("via") else ""),
            "line": name_id(line_label), "mode": mode_index[rep["mode"]],
            "path": to_latlon(path_2056), "platforms": to_latlon(platforms),
            "check": route_checks.get((rep["line_id"], rep["route_id"])),
            "references": reference_ids, "route": None, "key": (rep["line_id"], rep["route_id"]),
        })

    # Every route: departures and offsets at its stops, and the representative it is drawn with
    route_rows = []
    rep_by_route_id = {rep["key"]: k for k, rep in enumerate(reps)}
    for route in routes:
        if len(route["stop_ids"]) < 2:
            continue
        calls = []
        for position, (sid, (arrival, departure)) in enumerate(zip(route["stop_ids"], route["stop_offsets"])):
            offset = arrival if position == len(route["stop_ids"]) - 1 else departure
            if offset is None:
                offset = arrival if arrival is not None else departure
            group = group_index[group_key_of[sid]] if sid in group_key_of else -1
            calls.append([group, int(round(offset or 0))])
        key = _route_key(route, stops)
        rep = rep_of_key.get(key, rep_of_key.get(key[:3], -1))
        if (route["line_id"], route["route_id"]) in rep_by_route_id:
            reps[rep_by_route_id[(route["line_id"], route["route_id"])]]["route"] = len(route_rows)
        route_rows.append([
            name_id(route["line_name"] or route["line_id"]), mode_index[route["mode"]],
            name_id(stop_name(route["stop_ids"][0])), name_id(stop_name(route["stop_ids"][-1])),
            sorted(int(round(t)) for t in route["departure_times"]) if show_schedules else [],
            calls, rep,
        ])
    for rep in reps:
        del rep["key"]

    # Links used by the routes: one segment per link, with the lines using it
    link_lines = defaultdict(set)
    link_mode = {}
    for route in routes:
        for link_id in route["link_ids"]:
            if link_id in link_geometry:
                link_lines[link_id].add(route["line_name"] or route["line_id"])
                link_mode.setdefault(link_id, mode_index[route["mode"]])
    link_rows = []
    for link_id, lines in link_lines.items():
        (a, b) = to_latlon(link_geometry[link_id])
        sorted_lines = sorted(lines)
        shown = ", ".join(sorted_lines[:12]) + (f" (+{len(sorted_lines) - 12} more)" if len(sorted_lines) > 12 else "")
        link_rows.append([*a, *b, link_mode[link_id], name_id(shown)])
    bus_lanes = [[*a, *b] for a, b in (to_latlon(link_geometry[link_id]) for link_id in bus_lane_link_ids)]

    perimeter = None
    if region is not None:
        outline = gpd.GeoSeries([region.simplify(20)], crs = NETWORK_CRS).to_crs(epsg = 4326)
        perimeter = json.loads(outline.to_json())

    present = {mode_names[link[4]] for link in link_rows}
    default_modes = [mode_index[m] for m in DEFAULT_VISIBLE_MODES if m in present] or list(range(len(mode_names)))

    return {
        "names": names, "stops": stop_rows, "routes": route_rows, "reps": reps, "references": references,
        "links": link_rows, "busLanes": bus_lanes, "schedules": bool(show_schedules),
        "modes": mode_names, "modeColors": {m: MODE_COLORS.get(m, DEFAULT_MODE_COLOR) for m in mode_names},
        "defaultModes": default_modes, "perimeter": perimeter,
        "routeCheck": {"deviating": sum(row["status"] != "ok" for row in route_check_rows),
                       "total": len(route_check_rows)} if route_check_rows else None,
        "colors": {"busLane": BUS_LANE_COLOR, "reference": LINE_ROUTE_COLOR, "selected": SELECTED_COLOR,
                   "ok": CHECK_OK_COLOR, "deviates": CHECK_DEVIATES_COLOR},
    }


def _route_key(route, stops):
    """(line, origin, destination, stations): the key of the branch of select_representative_routes."""
    origin = stops.get(route["stop_ids"][0], {}).get("name") or route["stop_ids"][0]
    destination = stops.get(route["stop_ids"][-1], {}).get("name") or route["stop_ids"][-1]
    return (route["line_name"] or route["line_id"], origin, destination,
            tuple(_extract_station_code(sid) or sid for sid in route["stop_ids"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("simulation_path", nargs = "?", default = None,
                         help = "MATSim simulation_output directory (omit if using --schedule/--network)")
    parser.add_argument("--schedule", default = None,
                         help = "Explicit path to a transitSchedule.xml.gz (e.g. the pt2matsim-mapped schedule), "
                                "instead of reading it from simulation_path")
    parser.add_argument("--network", default = None,
                         help = "Explicit path to a network.xml.gz (e.g. the pt2matsim-mapped network), "
                                "instead of reading it from simulation_path")
    parser.add_argument("--output", default = "transit_schedule_map.html", help = "Output HTML path")
    parser.add_argument("--modes", nargs = "*", default = None,
                         help = "Restrict to these transportMode values (default: all modes)")
    parser.add_argument("--extent", default = None,
                         help = "Path to a polygon file; only the PT schedule within it is plotted")
    parser.add_argument("--line-routes", default = None,
                         help = "Shapefile of official line routes (columns LIGNE, DIRECTION) shown when a line is selected")
    parser.add_argument("--no-schedules", action = "store_true",
                         help = "Skip building per-stop departure-time popups (faster, smaller output)")
    parser.add_argument("--no-route-check", action = "store_true",
                         help = "Do not compare the routes with the official line routes (--line-routes)")
    parser.add_argument("--gtfs-routes", default = None, help = "GTFS routes.txt (to filter the route check by agency)")
    parser.add_argument("--agency", action = "append", default = [],
                         help = "Only check the lines of this GTFS agency_id (repeatable)")
    args = parser.parse_args()

    route_check_rows = None
    if args.line_routes and not args.no_route_check:
        if args.schedule is None:
            args.schedule = _find_output_file(args.simulation_path, "output_transitSchedule.xml.gz")
        if args.network is None:
            args.network = _find_output_file(args.simulation_path, "output_network.xml.gz")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok = True)
        route_check_rows, _ = pt_route_check.run(
            args.schedule, args.network, args.line_routes,
            os.path.join(os.path.dirname(args.output) or ".", "route_check.csv"),
            gtfs_routes_path = args.gtfs_routes, agencies = args.agency,
        )

    plot_transit_schedule(
        simulation_path = args.simulation_path,
        schedule_path = args.schedule, network_path = args.network,
        output_path = args.output,
        modes = set(args.modes) if args.modes else None,
        show_schedules = not args.no_schedules,
        extent_path = args.extent, line_routes_path = args.line_routes,
        route_check_rows = route_check_rows,
    )
