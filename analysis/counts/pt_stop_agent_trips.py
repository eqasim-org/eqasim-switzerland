"""
Stage: analysis.counts.pt_stop_agent_trips

Visualizes, link-by-link, the individual trips of every agent who boards or
alights a specific PT stop within a specific time window (default: "Meyrin,
CERN", 6:00-9:00). Unlike cross_border_flow_pt.py's stop-to-stop flow map
(aggregated straight lines between stops), this traces each matched agent's
actual trip geometry - the real chain of network links their car and/or PT
vehicle(s) traversed between the activity before and the activity after the
boarding/alighting event, including any transfers - and draws it as a thin
polyline: blue for agents who boarded at the stop, red for agents who
alighted there.

Network geometry is parsed directly from output_network.xml.gz (straight
node-to-node lines) rather than reusing analysis.counts.matching.network's
RoadNetwork, which filters to car-only links (filter_car_links()) and would
silently drop the bus/tram/rail links PT vehicles actually traverse.

Writes pt_stop_agent_trips_map.html into the shared get_analysis_output_path
folder (same as every other analysis.counts stage).
"""

import logging
import math
import os
from collections import defaultdict
from html import escape
from pathlib import Path

import geopandas as gpd
import pydeck as pdk
import shapely
import xml.etree.ElementTree as ET
import xopen
from pydeck.data_utils.viewport_helpers import compute_view
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .cross_border_flow_cars import (
    _find_output_file, _polygon_outline_layer,
    load_cross_border_person_ids, load_french_resident_cross_border_person_ids,
    load_swiss_resident_cross_border_person_ids,
)
from .cross_border_flow_pt import load_transit_schedule
from .matching.plots import Plotter
from .paths import configure_simulation_path, get_analysis_output_path, get_simulation_path

logger = logging.getLogger("synpp")

PERSON_ENTERS_VEHICLE_EVENT = "PersonEntersVehicle"
PERSON_LEAVES_VEHICLE_EVENT = "PersonLeavesVehicle"
VEHICLE_DEPARTS_EVENT = "VehicleDepartsAtFacility"
VEHICLE_ARRIVES_EVENT = "VehicleArrivesAtFacility"
LINK_ENTRY_EVENT = "entered link"
ACT_END_EVENT = "actend"
ACT_START_EVENT = "actstart"
CAR_VEHICLE_SUFFIX = ":car"
PROGRESS_EVERY = 2_000_000

STOP_COORDINATE_CRS = "epsg:2056"
BOARD_COLOR = [20, 90, 220]
ALIGHT_COLOR = [210, 30, 30]

# Lateral separation (in the projected CRS' meters) applied between trip
# lines that would otherwise overlap, so individual agents' paths stay
# distinguishable - see _offset_path.
LANE_OFFSET_METERS = 8.0
MAX_LANE = 5

PERSON_CATEGORY_LABELS = {
    "swiss_crossborder": "Swiss resident crossing the border",
    "french_resident": "French resident",
    "other_crossborder": "Cross-border resident (other)",
    "swiss_resident": "Swiss resident (non cross-border)",
}


def configure(context):
    configure_simulation_path(context)
    context.config("extent_path", default="")
    context.config("pt_stop_agent_trips_stop_name", default="Meyrin, CERN")
    context.config("pt_stop_agent_trips_start_time", default=6 * 3600)
    context.config("pt_stop_agent_trips_end_time", default=9 * 3600)
    context.config("only_weekday", default = True)


def execute(context):
    simulation_path = get_simulation_path(context)
    output_path = get_analysis_output_path(context)
    os.makedirs(output_path, exist_ok=True)

    events_path = _find_output_file(simulation_path, "output_events.xml.gz")
    schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    network_path = _find_output_file(simulation_path, "output_network.xml.gz")
    persons_path = _find_output_file(simulation_path, "output_persons.csv.gz")

    stop_name = context.config("pt_stop_agent_trips_stop_name")
    start_time = context.config("pt_stop_agent_trips_start_time")
    end_time = context.config("pt_stop_agent_trips_end_time")

    stop_coords, stop_names, _ = load_transit_schedule(schedule_path)
    target_facilities = {fid for fid, name in stop_names.items() if name == stop_name}
    if not target_facilities:
        logger.warning("No stop facility named %r found in %s; skipping agent-trip map.", stop_name, schedule_path)
        return dict(done=True, path=output_path)
    logger.info("Matched %d stop facility id(s) for %r", len(target_facilities), stop_name)

    boarding_events, alighting_events = find_stop_events(
        events_path, target_facilities, start_time, end_time,
    )
    logger.info(
        "Found %d boarding and %d alighting agent(s) at %r between %ds and %ds",
        len(boarding_events), len(alighting_events), stop_name, start_time, end_time,
    )
    if not boarding_events and not alighting_events:
        logger.info("No matching boarding/alighting events; skipping agent-trip map.")
        return dict(done=True, path=output_path)

    target_persons = set(boarding_events) | set(alighting_events)
    person_categories = classify_persons(persons_path, target_persons)

    link_endpoints = load_network_geometry(network_path)
    logger.info("Loaded straight-line geometry for %d network links from %s", len(link_endpoints), network_path)

    trips = trace_agent_trips(events_path, boarding_events, alighting_events, link_endpoints)
    logger.info("Reconstructed %d agent trip path(s)", len(trips))
    if not trips:
        logger.info("No traceable trip geometry for matched agents; skipping agent-trip map.")
        return dict(done=True, path=output_path)

    stop_points = [stop_coords[facility] for facility in target_facilities if facility in stop_coords]

    plot_agent_trips_map(
        trips, person_categories, stop_points, stop_name, context.config("extent_path"), output_path,
    )

    return dict(done=True, path=output_path)


# ---------------------------------------------------------------------------
# Agent classification (cross-border / Swiss / French resident)
# ---------------------------------------------------------------------------

def classify_persons(persons_path, person_ids):
    """{person_id: category_label} for exactly the requested person_ids -
    same cross-border/Swiss-resident/French-resident split as
    cross_border_flow_pt.py/cross_border_flow_cars.py, so an agent shown
    here can be read the same way as on those maps."""
    cross_border_ids = load_cross_border_person_ids(persons_path)
    swiss_crossborder_ids = load_swiss_resident_cross_border_person_ids(persons_path)
    french_resident_ids = load_french_resident_cross_border_person_ids(persons_path)

    def category(person_id):
        if person_id in swiss_crossborder_ids:
            return PERSON_CATEGORY_LABELS["swiss_crossborder"]
        if person_id in french_resident_ids:
            return PERSON_CATEGORY_LABELS["french_resident"]
        if person_id in cross_border_ids:
            return PERSON_CATEGORY_LABELS["other_crossborder"]
        return PERSON_CATEGORY_LABELS["swiss_resident"]

    return {person_id: category(person_id) for person_id in person_ids}


# ---------------------------------------------------------------------------
# Network parsing (mode-agnostic, unlike analysis.counts.matching.network)
# ---------------------------------------------------------------------------

def load_network_geometry(network_path):
    """{link_id: (from_xy, to_xy)} for every link in output_network.xml.gz,
    straight from node coordinates - independent of RoadNetwork's car-only
    filtering, so bus/tram/rail links used by PT vehicles are included."""
    node_coords = {}
    link_node_ids = {}

    with xopen.xopen(network_path, "r") as f:
        for _, elem in ET.iterparse(f, events=["end"]):
            if elem.tag == "node":
                node_coords[elem.attrib["id"]] = (float(elem.attrib["x"]), float(elem.attrib["y"]))
            elif elem.tag == "link":
                link_node_ids[elem.attrib["id"]] = (elem.attrib["from"], elem.attrib["to"])
            elem.clear()

    return {
        link_id: (node_coords[from_node], node_coords[to_node])
        for link_id, (from_node, to_node) in link_node_ids.items()
        if from_node in node_coords and to_node in node_coords
    }


def _to_node_point(link_endpoints, link_id):
    endpoints = link_endpoints.get(link_id)
    return endpoints[1] if endpoints else None


def _link_midpoint(link_endpoints, link_id):
    endpoints = link_endpoints.get(link_id)
    if not endpoints:
        return None
    (x0, y0), (x1, y1) = endpoints
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _activity_point(elem, link_endpoints):
    """The real location of an actend/actstart event: MATSim writes the
    activity's own x/y on these events, which is the true trip endpoint
    (e.g. the door of a building reached by a walk access/egress leg) - not
    just a point on its nearest network link. Falls back to that link's
    midpoint only if x/y are missing (older/differently configured events)."""
    x, y = elem.attrib.get("x"), elem.attrib.get("y")
    if x is not None and y is not None:
        return (float(x), float(y))
    return _link_midpoint(link_endpoints, elem.attrib.get("link", ""))


# ---------------------------------------------------------------------------
# Pass 1: find agents boarding/alighting the target stop within the window
# ---------------------------------------------------------------------------

def find_stop_events(events_path, target_facilities, start_time, end_time):
    """Returns (boarding_events, alighting_events), each {person_id:
    [event_time, ...]} - a person can appear more than once if they pass
    through the stop several times within the window."""
    boarding_events = defaultdict(list)
    alighting_events = defaultdict(list)
    vehicle_at_facility = {}

    processed = 0
    with xopen.xopen(events_path, "r") as f:
        parser = ET.iterparse(f, events=["start", "end"])
        _, root = next(parser)  # <events> start tag, kept to periodically release finished events

        for event, elem in parser:
            if event != "end" or elem.tag != "event":
                continue

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                logger.info("Processed %d events (stop-matching pass)...", processed)

            etype = elem.attrib.get("type")

            if etype == VEHICLE_ARRIVES_EVENT:
                vehicle_at_facility[elem.attrib["vehicle"]] = elem.attrib["facility"]
            elif etype == VEHICLE_DEPARTS_EVENT:
                vehicle_at_facility.pop(elem.attrib["vehicle"], None)
            elif etype in (PERSON_ENTERS_VEHICLE_EVENT, PERSON_LEAVES_VEHICLE_EVENT):
                facility = vehicle_at_facility.get(elem.attrib["vehicle"])
                if facility in target_facilities:
                    time = float(elem.attrib["time"])
                    if start_time <= time <= end_time:
                        target = boarding_events if etype == PERSON_ENTERS_VEHICLE_EVENT else alighting_events
                        target[elem.attrib["person"]].append(time)

            elem.clear()
            if processed % PROGRESS_EVERY == 0:
                root.clear()

    logger.info("Finished stop-matching pass over %d events from %s", processed, events_path)
    return dict(boarding_events), dict(alighting_events)


# ---------------------------------------------------------------------------
# Pass 2: reconstruct the matched agents' trip geometry, link by link
# ---------------------------------------------------------------------------

def trace_agent_trips(events_path, boarding_events, alighting_events, link_endpoints):
    """A "trip" here is one agent's chain of legs (with transfers) between
    two consecutive activities (actend -> ... -> actstart), matching MATSim's
    own trip concept. Only trips that contain one of the matched
    boarding/alighting events are kept - one row per (trip, matched event
    type), so a trip could produce both a "board" and an "alight" row in the
    rare case it does both at the target stop."""
    target_persons = set(boarding_events) | set(alighting_events)
    vehicle_onboard_targets = defaultdict(set)
    active_trips = {}
    completed_trips = []

    processed = 0
    with xopen.xopen(events_path, "r") as f:
        parser = ET.iterparse(f, events=["start", "end"])
        _, root = next(parser)  # <events> start tag, kept to periodically release finished events

        for event, elem in parser:
            if event != "end" or elem.tag != "event":
                continue

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                logger.info("Processed %d events (trip-tracing pass)...", processed)

            etype = elem.attrib.get("type")

            if etype == ACT_END_EVENT:
                person = elem.attrib["person"]
                if person in target_persons:
                    point = _activity_point(elem, link_endpoints)
                    active_trips[person] = dict(
                        start_time=float(elem.attrib["time"]),
                        points=[point] if point else [],
                    )
            elif etype == ACT_START_EVENT:
                person = elem.attrib["person"]
                trip = active_trips.pop(person, None)
                if trip is not None:
                    end_time = float(elem.attrib["time"])
                    end_point = _activity_point(elem, link_endpoints)
                    if end_point:
                        _append_point(trip["points"], end_point)
                    completed_trips.extend(_finalize_trip(
                        person, trip["start_time"], end_time, trip["points"],
                        boarding_events, alighting_events,
                    ))
            elif etype == PERSON_ENTERS_VEHICLE_EVENT:
                person, vehicle = elem.attrib["person"], elem.attrib["vehicle"]
                if person in target_persons:
                    vehicle_onboard_targets[vehicle].add(person)
            elif etype == PERSON_LEAVES_VEHICLE_EVENT:
                person, vehicle = elem.attrib["person"], elem.attrib["vehicle"]
                if person in target_persons and vehicle in vehicle_onboard_targets:
                    vehicle_onboard_targets[vehicle].discard(person)
                    if not vehicle_onboard_targets[vehicle]:
                        del vehicle_onboard_targets[vehicle]
            elif etype == LINK_ENTRY_EVENT:
                vehicle = elem.attrib.get("vehicle", "")
                riders = ()
                if vehicle.endswith(CAR_VEHICLE_SUFFIX):
                    person = vehicle[: -len(CAR_VEHICLE_SUFFIX)]
                    if person in active_trips:
                        riders = (person,)
                elif vehicle in vehicle_onboard_targets:
                    riders = [person for person in vehicle_onboard_targets[vehicle] if person in active_trips]

                if riders:
                    point = _to_node_point(link_endpoints, elem.attrib["link"])
                    if point:
                        for person in riders:
                            _append_point(active_trips[person]["points"], point)

            elem.clear()
            if processed % PROGRESS_EVERY == 0:
                root.clear()

    logger.info("Finished trip-tracing pass over %d events from %s", processed, events_path)
    return completed_trips


def _append_point(points, point):
    if not points or points[-1] != point:
        points.append(point)


def _finalize_trip(person, start_time, end_time, points, boarding_events, alighting_events):
    if len(points) < 2:
        return []
    trips = []
    if any(start_time <= t <= end_time for t in boarding_events.get(person, ())):
        trips.append(dict(
            person_id=person, event_type="board",
            start_time=start_time, end_time=end_time, points=list(points),
        ))
    if any(start_time <= t <= end_time for t in alighting_events.get(person, ())):
        trips.append(dict(
            person_id=person, event_type="alight",
            start_time=start_time, end_time=end_time, points=list(points),
        ))
    return trips


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _offset_path(points, offset):
    """Shifts every point of a path sideways by `offset` meters,
    perpendicular to the path's local direction there (average of its
    incoming/outgoing segment directions) - keeps the shifted line running
    parallel to the original route rather than a straight-line shear."""
    if offset == 0 or len(points) < 2:
        return points

    n = len(points)
    offset_points = []
    for i, (x, y) in enumerate(points):
        if i == 0:
            dx, dy = points[1][0] - x, points[1][1] - y
        elif i == n - 1:
            dx, dy = x - points[i - 1][0], y - points[i - 1][1]
        else:
            dx = points[i + 1][0] - points[i - 1][0]
            dy = points[i + 1][1] - points[i - 1][1]
        length = math.hypot(dx, dy)
        if length == 0:
            offset_points.append((x, y))
            continue
        nx, ny = -dy / length, dx / length  # unit vector perpendicular to travel direction
        offset_points.append((x + nx * offset, y + ny * offset))
    return offset_points


def plot_agent_trips_map(trips, person_categories, stop_points, stop_name, extent_path, output_path):
    lane_count = 2 * MAX_LANE + 1
    rows = []
    for i, trip in enumerate(trips):
        if len(trip["points"]) < 2:
            continue
        lane = (i % lane_count) - MAX_LANE
        offset_points = _offset_path(trip["points"], lane * LANE_OFFSET_METERS)
        rows.append(dict(
            person_id=trip["person_id"],
            person_category=person_categories.get(trip["person_id"], "Unknown"),
            event_type="Boarded here" if trip["event_type"] == "board" else "Alighted here",
            start_time=_format_time(trip["start_time"]),
            end_time=_format_time(trip["end_time"]),
            color=BOARD_COLOR if trip["event_type"] == "board" else ALIGHT_COLOR,
            geometry=LineString(offset_points),
        ))
    if not rows:
        logger.info("No trip geometry to plot; skipping agent-trip map.")
        return

    trips_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=STOP_COORDINATE_CRS).to_crs(epsg=4326)

    tooltip_fields = ["person_id", "person_category", "event_type", "start_time", "end_time"]
    path_data = Plotter._prepare_path_data(trips_gdf, [*tooltip_fields, "color"])
    path_data["_tooltip_html"] = path_data.apply(
        lambda row: Plotter._build_tooltip_html(row, tooltip_fields), axis=1,
    )

    layers = [pdk.Layer(
        "PathLayer",
        path_data,
        pickable=True,
        auto_highlight=True,
        get_path="path",
        get_width=1,
        get_color="color",
        highlight_color=[0, 200, 255],
        width_min_pixels=1,
        width_scale=1,
    )]

    # Small origin (ring, unfilled) / destination (filled dot) markers per
    # trip, colored to match its board/alight line, so the direction of
    # travel is legible even where trips overlap.
    origin_gdf = gpd.GeoDataFrame(
        {"color": [row["color"] for row in rows]},
        geometry=[Point(row["geometry"].coords[0]) for row in rows],
        crs=STOP_COORDINATE_CRS,
    ).to_crs(epsg=4326)
    origin_gdf["coordinates"] = list(zip(origin_gdf.geometry.x, origin_gdf.geometry.y))

    destination_gdf = gpd.GeoDataFrame(
        {"color": [row["color"] for row in rows]},
        geometry=[Point(row["geometry"].coords[-1]) for row in rows],
        crs=STOP_COORDINATE_CRS,
    ).to_crs(epsg=4326)
    destination_gdf["coordinates"] = list(zip(destination_gdf.geometry.x, destination_gdf.geometry.y))

    layers.append(pdk.Layer(
        "ScatterplotLayer",
        destination_gdf,
        pickable=False,
        get_position="coordinates",
        get_fill_color="color",
        filled=True,
        stroked=False,
        radius_units=pdk.types.String("pixels"),
        get_radius=3,
        radius_min_pixels=3,
    ))
    layers.append(pdk.Layer(
        "ScatterplotLayer",
        origin_gdf,
        pickable=False,
        get_position="coordinates",
        filled=False,
        stroked=True,
        get_line_color="color",
        line_width_min_pixels=1.5,
        radius_units=pdk.types.String("pixels"),
        get_radius=3,
        radius_min_pixels=3,
    ))

    if stop_points:
        point_gdf = gpd.GeoDataFrame(
            {"stop_name": [stop_name] * len(stop_points)},
            geometry=gpd.points_from_xy([p[0] for p in stop_points], [p[1] for p in stop_points]),
            crs=STOP_COORDINATE_CRS,
        ).to_crs(epsg=4326)
        point_gdf["coordinates"] = list(zip(point_gdf.geometry.x, point_gdf.geometry.y))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            point_gdf,
            pickable=True,
            get_position="coordinates",
            get_fill_color=[20, 20, 20],
            get_line_color=[255, 255, 255],
            stroked=True,
            line_width_min_pixels=1,
            radius_units=pdk.types.String("pixels"),
            get_radius=6,
            radius_min_pixels=6,
        ))

    if extent_path and os.path.exists(extent_path):
        boundary = gpd.read_file(extent_path)
        if boundary.crs != trips_gdf.crs:
            boundary = boundary.to_crs(trips_gdf.crs)
        region = shapely.force_2d(unary_union(boundary.geometry))
        layers.insert(0, _polygon_outline_layer(region, trips_gdf.crs, color=[0, 0, 0], width=2))

    minx, miny, maxx, maxy = trips_gdf.total_bounds
    margin_x, margin_y = max((maxx - minx) * 0.1, 0.005), max((maxy - miny) * 0.1, 0.005)
    view_state = compute_view([
        [minx - margin_x, miny - margin_y],
        [maxx + margin_x, maxy + margin_y],
    ])
    view_state.pitch = 0

    tooltip = {
        "html": "{_tooltip_html}",
        "style": {"color": "white", "background-color": "black", "padding": "10px"},
    }
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )

    output_file = os.path.join(output_path, "pt_stop_agent_trips_map.html")
    deck.to_html(output_file, notebook_display=False)
    _inject_legend(output_file, stop_name, len(trips))
    logger.info("Saved PT-stop agent-trip map to %s", output_file)


def _format_time(seconds):
    seconds = int(seconds) % (24 * 3600)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}"


def _inject_legend(path_to_save, stop_name, n_trips):
    legend_html = f"""
<div style="position:absolute;z-index:20;bottom:12px;left:12px;padding:10px 12px;
            border-radius:7px;background:rgba(255,255,255,0.95);
            box-shadow:0 2px 10px rgba(0,0,0,0.28);color:#222;
            font:12px/1.3 Arial, sans-serif;">
  <div style="font-weight:700;margin-bottom:6px;">Agent trips through {escape(stop_name)} ({n_trips})</div>
  <div><span style="display:inline-block;width:14px;height:3px;background:rgb({BOARD_COLOR[0]},{BOARD_COLOR[1]},{BOARD_COLOR[2]});margin-right:6px;"></span>Boarded here</div>
  <div style="margin-top:3px;"><span style="display:inline-block;width:14px;height:3px;background:rgb({ALIGHT_COLOR[0]},{ALIGHT_COLOR[1]},{ALIGHT_COLOR[2]});margin-right:6px;"></span>Alighted here</div>
  <div style="margin-top:6px;color:#555;">
    <span style="display:inline-block;width:9px;height:9px;border-radius:50%;border:1.5px solid #555;margin-right:6px;"></span>Trip origin
  </div>
  <div style="margin-top:3px;color:#555;">
    <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:#555;margin-right:6px;"></span>Trip destination
  </div>
</div>
"""
    path = Path(path_to_save)
    html = path.read_text(encoding="utf-8")
    html = html.replace("</body>", legend_html + "\n</body>", 1)
    path.write_text(html, encoding="utf-8")
