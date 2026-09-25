import glob
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import folium
import geopandas as gpd
import xopen
from shapely.geometry import LineString, Point

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


def configure(context):
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("analysis.pt.transit_schedule_modes", default = [])
    context.config("analysis.pt.transit_schedule_show_schedules", default = True)


def execute(context):
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

    modes = set(context.config("analysis.pt.transit_schedule_modes")) or None
    show_schedules = context.config("analysis.pt.transit_schedule_show_schedules")

    output_dir = os.path.join(matsim_output_folder, "pt_transit_schedule")
    os.makedirs(output_dir, exist_ok = True)
    output_file = os.path.join(output_dir, "transit_schedule_map.html")

    plot_transit_schedule(matsim_output_folder, output_path = output_file, modes = modes, show_schedules = show_schedules)

    return dict(done = True, path = output_file)


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
    return bool(modes & {"bus", "pt"}) and "car" not in modes


def _parse_hms(value):
    h, m, s = value.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _format_hms(seconds):
    seconds = int(round(seconds)) % (24 * 3600)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


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


# ---------------------------------------------------------------------------
# Per-stop schedules (departure times of every service calling there)
# ---------------------------------------------------------------------------

def build_stop_schedules(stops, routes, group_key_of = None):
    schedules = defaultdict(dict)

    for route in routes:
        stop_ids = route["stop_ids"]
        offsets = route["stop_offsets"]
        departure_times = route["departure_times"]
        if len(stop_ids) < 2 or not offsets or not departure_times:
            continue

        origin_id, dest_id = stop_ids[0], stop_ids[-1]
        origin_name = stops.get(origin_id, {}).get("name") or origin_id
        dest_name   = stops.get(dest_id, {}).get("name") or dest_id
        origin_offset = offsets[0][1] if offsets[0][1] is not None else (offsets[0][0] or 0.0)
        dest_offset   = offsets[-1][0] if offsets[-1][0] is not None else (offsets[-1][1] or 0.0)

        for position, (stop_id, (arrival_offset, departure_offset)) in enumerate(zip(stop_ids, offsets)):
            is_terminal = position == len(stop_ids) - 1
            offset = arrival_offset if is_terminal else departure_offset
            if offset is None:
                offset = arrival_offset if arrival_offset is not None else departure_offset
            if offset is None:
                continue

            here_name = stops.get(stop_id, {}).get("name") or stop_id
            label_place = origin_name if is_terminal else dest_name
            line_key = (route["line_name"] or route["line_id"], label_place)
            group_key = group_key_of.get(stop_id, stop_id) if group_key_of else stop_id

            times = []
            for base_time in departure_times:
                here_time = base_time + offset
                parts = [f"{origin_name} {_format_hms(base_time + origin_offset)}"]
                if here_name not in (origin_name, dest_name):
                    parts.append(f"{here_name} {_format_hms(here_time)}")
                parts.append(f"{dest_name} {_format_hms(base_time + dest_offset)}")
                times.append((_format_hms(here_time), " &rarr; ".join(parts)))

            existing = schedules[group_key].get(line_key)
            if existing is None:
                schedules[group_key][line_key] = (is_terminal, label_place, origin_name, times)
            else:
                existing[3].extend(times)

    for groups in schedules.values():
        for is_terminal, label_place, origin_name, times in groups.values():
            times.sort(key = lambda t: t[0])

    return schedules


def _extract_station_code(stop_id):
    matches = re.findall(r"\d{7,8}", str(stop_id))
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


def _bearing_deg(from_latlon, to_latlon):
    lat1, lon1 = math.radians(from_latlon[0]), math.radians(from_latlon[1])
    lat2, lon2 = math.radians(to_latlon[0]), math.radians(to_latlon[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def select_representative_routes(routes, stops):
    groups = defaultdict(list)
    for route in routes:
        if len(route["stop_ids"]) < 2:
            continue
        origin_name = stops.get(route["stop_ids"][0], {}).get("name") or route["stop_ids"][0]
        dest_name   = stops.get(route["stop_ids"][-1], {}).get("name") or route["stop_ids"][-1]
        key = (route["line_name"] or route["line_id"], origin_name, dest_name)
        groups[key].append(route)

    representatives = []
    for group_routes in groups.values():
        signature_freq = Counter(tuple(r["stop_ids"]) for r in group_routes)
        best = max(
            group_routes,
            key = lambda r: (len(set(r["stop_ids"])), signature_freq[tuple(r["stop_ids"])]),
        )
        representatives.append(best)
    return representatives


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def plot_transit_schedule(
    simulation_path,
    output_path = "transit_schedule_map.html",
    modes = None,
    show_schedules = True,
):
    
    schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    network_path  = _find_output_file(simulation_path, "output_network.xml.gz")

    print(f"Reading transit schedule from {schedule_path} ...")
    stops, routes = load_transit_schedule(schedule_path)
    print(f"  -> {len(stops)} stops, {len(routes)} transit routes")

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
    print(f"  -> {len(link_geometry)} links")

    bus_lane_link_ids = [
        link_id for link_id, modes in link_modes.items()
        if link_id in link_geometry and is_bus_reserved_link(modes)
    ]
    print(f"  -> {len(bus_lane_link_ids)} link(s) reserved for bus/PT (excluded from car)")

    stop_ids_with_coords = [sid for sid, info in stops.items() if info["x"] is not None]
    if not stop_ids_with_coords:
        raise RuntimeError("No stop coordinates found in the transit schedule; nothing to plot.")

    print("Grouping stops across sources by station code...")
    group_key_of = {sid: (_extract_station_code(sid) or sid) for sid in stop_ids_with_coords}
    group_members = defaultdict(set)
    for sid in stop_ids_with_coords:
        group_members[group_key_of[sid]].add(sid)

    group_keys = list(group_members)
    group_points_2056 = gpd.GeoSeries(
        [
            Point(
                sum(stops[m]["x"] for m in group_members[key]) / len(group_members[key]),
                sum(stops[m]["y"] for m in group_members[key]) / len(group_members[key]),
            )
            for key in group_keys
        ],
        crs = NETWORK_CRS,
    ).to_crs(epsg = 4326)

    group_name = {}
    group_lines = {}
    for key in group_keys:
        members = group_members[key]
        names = [stops[m]["name"] for m in members if stops[m]["name"]]
        group_name[key] = Counter(names).most_common(1)[0][0] if names else key
        lines = set()
        for m in members:
            lines |= stops[m]["lines"]
        group_lines[key] = lines
    print(f"  -> {len(group_keys)} station(s) after grouping ({len(stop_ids_with_coords)} platform(s)/stop id(s) before)")

    group_latlon = {
        key: [float(point.y), float(point.x)]
        for key, point in zip(group_keys, group_points_2056)
    }

    print("Aggregating links used by transit routes (one segment per link, not per line)...")
    mode_link_lines = defaultdict(lambda: defaultdict(set))
    for route in routes:
        line_label = route["line_name"] or route["line_id"]
        for link_id in route["link_ids"]:
            if link_id in link_geometry:
                mode_link_lines[route["mode"]][link_id].add(line_label)

    mode_features = defaultdict(list)
    for mode, link_lines in mode_link_lines.items():
        for link_id, lines in link_lines.items():
            from_xy, to_xy = link_geometry[link_id]
            sorted_lines = sorted(lines)
            shown_lines = ", ".join(sorted_lines[:8]) + (f" (+{len(sorted_lines) - 8} more)" if len(sorted_lines) > 8 else "")
            mode_features[mode].append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [from_xy, to_xy]},
                "properties": {"mode": mode, "n_lines": len(sorted_lines), "lines": shown_lines},
            })

    print("Reprojecting aggregated links to WGS84...")
    for mode, features in mode_features.items():
        lines_2056 = gpd.GeoSeries(
            [LineString(f["geometry"]["coordinates"]) for f in features], crs = NETWORK_CRS,
        ).to_crs(epsg = 4326)
        for feature, geometry in zip(features, lines_2056):
            feature["geometry"]["coordinates"] = list(geometry.coords)

    print("Selecting one representative route per line + direction (for the line-selector menu)...")
    line_routes = select_representative_routes(routes, stops)
    print(f"  -> {len(line_routes)} representative route(s) selectable (from {len(routes)} total patterns)")

    print("Building per-line paths for the selector menu...")
    route_index = []
    skipped = 0
    for route in line_routes:
        path_2056 = _route_path(route, link_geometry, stops)
        if path_2056 is None:
            skipped += 1
            continue
        path_4326 = list(
            gpd.GeoSeries([LineString(path_2056)], crs = NETWORK_CRS).to_crs(epsg = 4326).iloc[0].coords
        )

        from_name = stops.get(route["stop_ids"][0], {}).get("name", "-") if route["stop_ids"] else "-"
        to_name   = stops.get(route["stop_ids"][-1], {}).get("name", "-") if route["stop_ids"] else "-"

        stop_group_keys = []
        for stop_id in route["stop_ids"]:
            key = group_key_of.get(stop_id)
            if key is not None and (not stop_group_keys or stop_group_keys[-1] != key):
                stop_group_keys.append(key)

        line_label = route["line_name"] or route["line_id"]
        route_index.append({
            "route_id": route["route_id"],
            "mode": route["mode"],
            "label": f"{line_label} ({route['mode']}): {from_name} → {to_name}",
            "path": [[lat, lon] for lon, lat in path_4326],
            "stop_group_keys": stop_group_keys,
        })

    if skipped:
        print(f"  -> skipped {skipped} route(s) with no resolvable geometry")
    route_index.sort(key = lambda r: r["label"])

    center = [group_points_2056.y.mean(), group_points_2056.x.mean()]

    print("Building map...")
    m = folium.Map(location = center, zoom_start = 12, tiles = None, prefer_canvas = True)
    folium.TileLayer(
        tiles   = TILE_URL,
        attr    = "© swisstopo",
        name    = "swisstopo (grayscale)",
        opacity = 0.8,
        control = False,
    ).add_to(m)

    mode_layers = {}
    for mode, features in sorted(mode_features.items()):
        color = MODE_COLORS.get(mode, DEFAULT_MODE_COLOR)
        n_links = len(features)
        layer = folium.FeatureGroup(name = f"{mode} ({n_links} link(s))", show = mode in DEFAULT_VISIBLE_MODES)
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            style_function     = lambda _f, color = color: {"color": color, "weight": 2, "opacity": 0.6},
            highlight_function = lambda _f: {"weight": 6, "opacity": 1, "color": "#222222"},
            tooltip = folium.GeoJsonTooltip(
                fields  = ["mode", "n_lines", "lines"],
                aliases = ["Mode:", "Lines using this link:", ""],
                sticky  = True,
            ),
        ).add_to(layer)
        layer.add_to(m)
        mode_layers[mode] = layer

    if bus_lane_link_ids:
        print("Adding reserved bus-lane layer...")
        bus_lane_coords_2056 = [link_geometry[link_id] for link_id in bus_lane_link_ids]
        bus_lane_lines_2056 = gpd.GeoSeries(
            [LineString([from_xy, to_xy]) for from_xy, to_xy in bus_lane_coords_2056], crs = NETWORK_CRS,
        ).to_crs(epsg = 4326)
        bus_lane_layer = folium.FeatureGroup(name = f"Reserved bus lanes ({len(bus_lane_link_ids)})", show = True)
        for geometry in bus_lane_lines_2056:
            latlon_coords = [(lat, lon) for lon, lat in geometry.coords]
            folium.PolyLine(
                latlon_coords,
                color = BUS_LANE_COLOR, weight = 3, opacity = 0.85, dash_array = "6,4",
                tooltip = "Reserved bus/PT lane",
            ).add_to(bus_lane_layer)

            mid_index = len(latlon_coords) // 2
            midpoint  = latlon_coords[mid_index]
            bearing   = _bearing_deg(latlon_coords[max(mid_index - 1, 0)], latlon_coords[min(mid_index + 1, len(latlon_coords) - 1)])
            folium.Marker(
                midpoint,
                icon = folium.DivIcon(html = (
                    f'<div style="transform: rotate({bearing - 90:.1f}deg); '
                    f'transform-origin: center; color:{BUS_LANE_COLOR}; font-size:16px; '
                    'font-weight:bold; line-height:1;">&#9658;</div>'
                )),
                tooltip = "Reserved bus/PT lane",
            ).add_to(bus_lane_layer)
        bus_lane_layer.add_to(m)

    line_to_mode = {route["line_name"] or route["line_id"]: route["mode"] for route in routes}

    stop_schedules = {}
    if show_schedules:
        print("Building per-stop schedules (departure times for every service)...")
        stop_schedules = build_stop_schedules(stops, routes, group_key_of = group_key_of)
        n_departure_rows = sum(
            len(times) for groups in stop_schedules.values() for *_, times in groups.values()
        )
        print(
            f"  -> {len(stop_schedules)} station(s) with a schedule, {n_departure_rows} departure row(s) total "
            "(pass a smaller modes=... set or show_schedules=False if the output gets too large)"
        )

    print("Adding stops layer...")
    stops_layer = folium.FeatureGroup(name = f"Stops ({len(group_keys)})", show = True)
    for key, point in zip(group_keys, group_points_2056):
        stop_name = group_name[key]
        groups = stop_schedules.get(key)

        if groups:
            popup_html = (
                '<div style="max-height:350px; overflow-y:auto; font-family:sans-serif; '
                'font-size:12px; min-width:280px;">'
                f"<b>{stop_name}</b><br><br>"
            )
            for (line_name, label_place), (is_terminal, _, _, times) in sorted(groups.items()):
                color = MODE_COLORS.get(line_to_mode.get(line_name), DEFAULT_MODE_COLOR)
                label = "Arrivals" if is_terminal else "Departures"
                time_header = "Arrival" if is_terminal else "Departure"
                direction = f"&larr; from {label_place}" if is_terminal else f"&rarr; {label_place}"
                rows_html = "".join(
                    f"<tr><td>{time}</td><td>{summary}</td></tr>" for time, summary in times
                )
                popup_html += (
                    f'<div style="margin-bottom:8px;">'
                    f'<b style="color:{color};">{line_name}</b> {direction}<br>'
                    f'<button onclick="var t=this.nextElementSibling;'
                    f"t.style.display=(t.style.display==='none'?'table':'none');\""
                    f'style="font-size:11px; margin:2px 0; cursor:pointer;">'
                    f"{label} ({len(times)} trips)</button>"
                    f'<table style="display:none; width:100%; font-size:11px; border-collapse:collapse;">'
                    f'<tr><th style="text-align:left;">{time_header}</th><th style="text-align:left;">Itinerary</th></tr>'
                    f"{rows_html}</table>"
                    f"</div>"
                )
            popup_html += "</div>"
            popup   = folium.Popup(popup_html, max_width = 420)
            tooltip = f"{stop_name} (click for schedule)"
        else:
            popup   = None
            lines   = ", ".join(sorted(group_lines[key])) or "-"
            tooltip = f"{stop_name} - {lines}"

        folium.CircleMarker(
            location     = [point.y, point.x],
            radius       = 3,
            color        = "#333333",
            weight       = 1,
            fill         = True,
            fill_color   = "#ffffff",
            fill_opacity = 0.9,
            tooltip      = tooltip,
            popup        = popup,
        ).add_to(stops_layer)
    stops_layer.add_to(m)

    folium.LayerControl(position = "topleft", collapsed = False).add_to(m)

    print("Adding line-selector menu...")
   
    map_var_name = m.get_name()
    menu_html = """
<div id="line-menu" style="position:fixed; top:80px; right:10px; z-index:1000;
     background:white; border:1px solid #ccc; border-radius:6px; padding:8px;
     font-family:sans-serif; font-size:12px; width:280px;">
  <b>Select a line to highlight</b>
  <input id="line-menu-filter" type="text" placeholder="Filter lines..."
         style="width:100%; box-sizing:border-box; margin:6px 0; padding:4px;">
  <div id="line-menu-list" style="max-height:320px; overflow-y:auto; border-top:1px solid #eee;"></div>
</div>
"""
    m.get_root().html.add_child(folium.Element(menu_html))

    highlight_js = f"""
<script>
(function () {{
  var routeIndex = {json.dumps(route_index)};
  var mapVarName = {json.dumps(map_var_name)};
  var groupLatLon = {json.dumps(group_latlon)};
  var highlightLine = null;
  var highlightStopMarkers = [];
  var selectedRouteId = null;
  var listEl = document.getElementById("line-menu-list");
  var filterEl = document.getElementById("line-menu-filter");

  function clearHighlight(map) {{
    if (highlightLine) {{
      map.removeLayer(highlightLine);
      highlightLine = null;
    }}
    highlightStopMarkers.forEach(function (marker) {{ map.removeLayer(marker); }});
    highlightStopMarkers = [];
  }}

  function selectRoute(route, rowEl) {{
    var map = window[mapVarName];
    if (!map) {{ return; }}

    clearHighlight(map);
    var previouslySelected = listEl.querySelector(".line-menu-item.selected");
    if (previouslySelected) {{ previouslySelected.classList.remove("selected"); }}

    if (selectedRouteId !== null && selectedRouteId === route.route_id) {{
      selectedRouteId = null;
      return;
    }}

    highlightLine = L.polyline(route.path, {{color: "#e34948", weight: 6, opacity: 1}}).addTo(map);
    highlightLine.bringToFront();
    map.fitBounds(highlightLine.getBounds(), {{maxZoom: 16, padding: [40, 40]}});

    // Also mark every station this route serves. interactive: false keeps
    // this ring from capturing clicks itself - it sits exactly on top of
    // the real stop marker (which still has its own popup/tooltip), and
    // without this a click here would hit the ring instead and do
    // nothing, making the stop appear to have "lost" its popup while
    // highlighted.
    (route.stop_group_keys || []).forEach(function (key) {{
      var latlon = groupLatLon[key];
      if (!latlon) {{ return; }}
      var marker = L.circleMarker(latlon, {{
        radius: 7, color: "#e34948", weight: 2, fill: true,
        fillColor: "#ffffff", fillOpacity: 1, interactive: false,
      }}).addTo(map);
      marker.bringToFront();
      highlightStopMarkers.push(marker);
    }});

    rowEl.classList.add("selected");
    selectedRouteId = route.route_id;
  }}

  function renderList(filterText) {{
    var needle = (filterText || "").toLowerCase();
    listEl.innerHTML = "";
    routeIndex.forEach(function (route) {{
      if (needle && route.label.toLowerCase().indexOf(needle) === -1) {{ return; }}
      var row = document.createElement("div");
      row.className = "line-menu-item" + (route.route_id === selectedRouteId ? " selected" : "");
      row.textContent = route.label;
      row.style.cssText = "padding:3px 4px; cursor:pointer; border-bottom:1px solid #f2f2f2;";
      row.addEventListener("mouseenter", function () {{ row.style.background = "#f0f0f0"; }});
      row.addEventListener("mouseleave", function () {{
        row.style.background = row.classList.contains("selected") ? "#ffe8e8" : "";
      }});
      row.addEventListener("click", function () {{ selectRoute(route, row); }});
      if (row.classList.contains("selected")) {{ row.style.background = "#ffe8e8"; }}
      listEl.appendChild(row);
    }});
  }}

  filterEl.addEventListener("input", function () {{ renderList(filterEl.value); }});
  renderList("");
}})();
</script>
"""
    m.get_root().html.add_child(folium.Element(highlight_js))

    legend_html = (
        '<div style="position:fixed; bottom:30px; left:30px; z-index:1000;'
        'background:white; padding:10px; border-radius:6px;'
        'border:1px solid #ccc; font-size:12px;"><b>Transport mode</b><br>'
    )
    for mode in sorted(mode_features):
        color = MODE_COLORS.get(mode, DEFAULT_MODE_COLOR)
        legend_html += f'<span style="color:{color};">&#9644;</span> {mode}<br>'
    if bus_lane_link_ids:
        legend_html += f'<span style="color:{BUS_LANE_COLOR};">&#9644;&#9658;</span> reserved bus/PT lane (arrow = direction)<br>'
    legend_html += '<div style="margin-top:6px;color:#666;">Use the line menu (top right) to highlight a line and its stops</div>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    print("Saving map...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok = True)
    m.save(output_path)
    print(f"Done - saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("simulation_path", help = "MATSim simulation_output directory")
    parser.add_argument("--output", default = "transit_schedule_map.html", help = "Output HTML path")
    parser.add_argument("--modes", nargs = "*", default = None,
                         help = "Restrict to these transportMode values (default: all modes)")
    parser.add_argument("--no-schedules", action = "store_true",
                         help = "Skip building per-stop departure-time popups (faster, smaller output)")
    args = parser.parse_args()

    plot_transit_schedule(
        args.simulation_path, output_path = args.output,
        modes = set(args.modes) if args.modes else None,
        show_schedules = not args.no_schedules,
    )
