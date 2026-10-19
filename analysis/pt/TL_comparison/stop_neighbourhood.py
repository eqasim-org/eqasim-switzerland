"""Everything the MATSim events say about one public transport stop and its surroundings (by default Lausanne,
Grancy): who boards and alights there, where they come from and go to, how they get to the stop, when, and which
activities take place around it.

The result is an interactive map centred on the stop, with a panel of charts, and the tables behind it:
  - the stops around, sized by their number of passengers,
  - the boarding passengers: the places where their trips started (colored by activity), the rides they take,
  - the alighting passengers: the places where their trips end and the rides they come from,
  - the activities that take place in the area (the density of homes, jobs ... around the stop),
  - the walking distance to and from the stop, the hourly profile, the transfers.
"""
import argparse
import base64
import bisect
import io
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "TPG_comparison"))
sys.path.insert(0, os.path.join(HERE, "..", "transit_schedule"))

import matsim_events
import visualize

EVENTS_FILE = matsim_events.EVENTS_FILE

# The places are added up in cells of this size (m)
CELL = 50.0

# categorical slots 1-8 of the reference palette, in order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
)
GRAY = "#8a8985"
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

ACTIVITY_COLORS = {
    "home": BLUE, "work": ORANGE, "education": AQUA, "shop": YELLOW, "leisure": MAGENTA, "outside": VIOLET,
}
OTHER_ACTIVITY_COLOR = GRAY


def activity_group(activity):
    """home_secondary -> home ... so that the legend stays short"""
    base = activity.split("_")[0]
    return base if base in ACTIVITY_COLORS else "other"


def configure(context):
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("extent_path", default = "")

    # The run whose events are read, when it is not the one of the other analyses
    context.config("analysis.pt.neighbourhood_output_id", default = None)
    context.config("analysis.pt.neighbourhood_stop",      default = "8592052")  # GTFS id of Lausanne, Grancy
    context.config("analysis.pt.neighbourhood_radius",    default = 600.0)


def execute(context):
    output_id = context.config("analysis.pt.neighbourhood_output_id") or context.config("output_id")
    simulation_path = os.path.join(context.config("output_path"), output_id, context.config("simulation_directory"))

    output_dir = run(
        simulation_path, input_downsampling = context.config("input_downsampling"),
        stop = str(context.config("analysis.pt.neighbourhood_stop")),
        radius = float(context.config("analysis.pt.neighbourhood_radius")),
        extent_path = context.config("extent_path") or None,
    )

    return dict(done = True, path = output_dir)


# ---------------------------------------------------------------------------
# Reading the events
# ---------------------------------------------------------------------------

def stop_code(facility):
    return str(facility).split(":")[0].split(".")[0]


def read_pt_events(events_path, stops, routes, near_codes, target):
    """One pass over the public transport events.

    Returns the passengers of every stop within the radius (stop code, line, hour, boardings and alightings), the
    arrivals of every vehicle at its stops, the service (line, route) of the vehicles and the agents that board or
    alight at the target stop."""
    line_of_route = {key: route["line_name"] for key, route in routes.items()}
    drivers, service, vehicle_stop = set(), {}, {}
    arrivals = defaultdict(lambda: ([], []))
    passengers = Counter()
    at_target = set()

    for event in matsim_events._pt_events(events_path):
        kind = event["type"]

        if kind == "TransitDriverStarts":
            drivers.add(event["driverId"])
            service[event["vehicleId"]] = (event["transitLineId"], event["transitRouteId"])
        elif kind == "VehicleArrivesAtFacility":
            vehicle = event["vehicle"]
            vehicle_stop[vehicle] = (event["facility"], float(event["time"]))
            times, facilities = arrivals[vehicle]
            times.append(float(event["time"]))
            facilities.append(event["facility"])
        elif event["person"] not in drivers and event["vehicle"] in vehicle_stop and event["vehicle"] in service:
            facility, arrival = vehicle_stop[event["vehicle"]]
            code = stop_code(facility)

            if code in near_codes:
                side = "boardings" if kind == "PersonEntersVehicle" else "alightings"
                passengers[(code, line_of_route.get(service[event["vehicle"]], "?"), int(arrival // 3600) % 24, side)] += 1
            if code == target:
                at_target.add(event["person"])

    return passengers, arrivals, service, at_target


def _stop_at(arrivals, vehicle, time):
    times, facilities = arrivals.get(vehicle, ([], []))
    index = bisect.bisect_right(times, time) - 1

    return facilities[index] if index >= 0 else None


def _is_interaction(event):
    return event.get("actType", "").endswith("interaction")


def follow_persons(events_path, persons):
    """The trips of the given agents: stages with the places where they start and end and the rides.

    A trip goes from the end of an activity to the start of the next one, the interactions (car, pt ...) excluded."""
    with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as patterns:
        for person in persons:
            patterns.write(f'person="{person}"\n')
        patterns_path = patterns.name

    trips, state, position, open_stage = [], {}, {}, {}

    try:
        for event in matsim_events_stream(events_path, ["-F", "-f", patterns_path]):
            kind, person = event["type"], event["person"]
            trip = state.get(person)

            if kind == "actend":
                position[person] = (float(event["x"]), float(event["y"]))
            elif kind == "actstart" and person in open_stage:
                open_stage.pop(person)["end_xy"] = (float(event["x"]), float(event["y"]))

            if kind == "actend" and not _is_interaction(event):
                state[person] = dict(
                    person = person, start_type = event["actType"], start_xy = position[person],
                    departure = float(event["time"]), stages = [], current = None, end_type = None, end_xy = None,
                )
            elif trip is None:
                continue
            elif kind == "actstart" and not _is_interaction(event):
                trip.update(end_type = event["actType"], end_xy = (float(event["x"]), float(event["y"])), arrival = float(event["time"]))
                if any(stage["rides"] for stage in trip["stages"]):
                    trips.append(trip)
                del state[person]
            elif kind == "departure":
                trip["current"] = dict(mode = event["legMode"], start_xy = position.get(person), end_xy = None, rides = [])
                trip["stages"].append(trip["current"])
            elif trip["current"] is None:
                continue
            elif kind == "PersonEntersVehicle" and event["vehicle"].startswith("veh_"):
                trip["current"]["rides"].append(dict(vehicle = event["vehicle"], board = float(event["time"]), leave = None))
            elif kind == "PersonLeavesVehicle" and event["vehicle"].startswith("veh_"):
                for ride in trip["current"]["rides"]:
                    if ride["vehicle"] == event["vehicle"] and ride["leave"] is None:
                        ride["leave"] = float(event["time"])
            elif kind == "arrival":
                open_stage[person] = trip["current"]
                trip["current"] = None
    finally:
        os.remove(patterns_path)

    return trips


def matsim_events_stream(events_path, grep_arguments):
    import subprocess

    zcat = subprocess.Popen(["zcat", events_path], stdout = subprocess.PIPE)
    grep = subprocess.Popen(["grep", "-a"] + grep_arguments, stdin = zcat.stdout, stdout = subprocess.PIPE, text = True)
    zcat.stdout.close()

    for line in grep.stdout:
        yield dict(matsim_events._ATTRIBUTES.findall(line))

    if grep.wait() not in (0, 1):
        raise RuntimeError(f"Could not read the events of {events_path}")
    zcat.wait()


def activities_in_area(events_path, center, radius):
    """{(activity type, cell): number of activities} and {activity type: number of distinct agents} in the radius."""
    cells, agents = Counter(), defaultdict(set)
    cx, cy = center

    for event in matsim_events_stream(events_path, ["-F", 'type="actstart"']):
        activity = event["actType"]
        if activity.endswith("interaction"):
            continue

        x, y = float(event["x"]), float(event["y"])
        if abs(x - cx) <= radius and abs(y - cy) <= radius and math.hypot(x - cx, y - cy) <= radius:
            cells[(activity, (round(x / CELL) * CELL, round(y / CELL) * CELL))] += 1
            agents[activity].add(event["person"])

    return cells, {activity: len(found) for activity, found in agents.items()}


# ---------------------------------------------------------------------------
# What happens at the stop
# ---------------------------------------------------------------------------

def resolve_rides(trip, service, arrivals, routes):
    rides = []

    for stage in trip["stages"]:
        for ride in stage["rides"]:
            if ride["leave"] is None or ride["vehicle"] not in service:
                continue

            route  = routes.get(service[ride["vehicle"]])
            board  = _stop_at(arrivals, ride["vehicle"], ride["board"])
            alight = _stop_at(arrivals, ride["vehicle"], ride["leave"])

            if route is None or board not in route["stop_ids"]:
                continue

            start = route["stop_ids"].index(board)
            end   = route["stop_ids"].index(alight, start) if alight in route["stop_ids"][start:] else start

            rides.append(dict(
                line = route["line_name"], mode = route["mode"], board = board, alight = route["stop_ids"][end],
                stops = route["stop_ids"][start:end + 1], board_arrival = _arrival(arrivals, ride["vehicle"], ride["board"]),
                alight_arrival = _arrival(arrivals, ride["vehicle"], ride["leave"]),
                direction = f"{route['stop_ids'][0]}|{route['stop_ids'][-1]}",
            ))

    return rides


def _arrival(arrivals, vehicle, time):
    times, _ = arrivals.get(vehicle, ([], []))
    index = bisect.bisect_right(times, time) - 1

    return times[index] if index >= 0 else None


def passengers_of_the_stop(trips, target, service, arrivals, routes, stops, center, direction_names):
    """One record per boarding and per alighting at the target stop."""
    records, trip_rows = [], []

    for trip in trips:
        rides = resolve_rides(trip, service, arrivals, routes)

        for i, ride in enumerate(rides):
            for role, stop, arrival in (("board", ride["board"], ride["board_arrival"]), ("alight", ride["alight"], ride["alight_arrival"])):
                if stop_code(stop) != target or arrival is None:
                    continue

                first = role == "board" and i == 0
                last  = role == "alight" and i == len(rides) - 1

                other_stop = other_mode = None

                if first:
                    other = ("start", trip["start_type"], trip["start_xy"])
                elif last:
                    other = ("end", trip["end_type"], trip["end_xy"])
                else:
                    # the other ride of the transfer: the one the agent comes from (boarding) or goes to (alighting)
                    neighbour = rides[i - 1] if role == "board" else rides[i + 1]
                    other = ("transfer", neighbour["line"], None)
                    other_stop = neighbour["alight"] if role == "board" else neighbour["board"]
                    other_mode = neighbour["mode"]

                walk = None
                if other[2] is not None:
                    walk = math.hypot(other[2][0] - center[0], other[2][1] - center[1])

                records.append(dict(
                    person = trip["person"], role = role, line = ride["line"], mode = ride["mode"], hour = int(arrival // 3600) % 24,
                    heading = direction_names.get(ride["direction"], ""), other_kind = other[0], other_type = other[1],
                    other_xy = other[2], walk_m = walk, ride_stops = ride["stops"], platform = stop,
                    other_stop = other_stop, other_mode = other_mode,
                ))

        if any(stop_code(r["board"]) == target or stop_code(r["alight"]) == target for r in rides):
            trip_rows.append(dict(
                person = trip["person"], start = trip["start_type"], end = trip["end_type"], departure = trip["departure"],
                chain = " > ".join(
                    f"{r['line']} ({stops[r['board']]['name']} → {stops[r['alight']]['name']})" for r in rides
                ),
            ))

    return pd.DataFrame(records), pd.DataFrame(trip_rows)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, title):
    ax.set_title(title, loc = "left", fontsize = 10, color = INK, fontweight = "bold")
    ax.set_facecolor(SURFACE)
    ax.grid(axis = "y", color = GRID, linewidth = 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors = INK_SECONDARY, labelsize = 8, length = 0)


def charts(records, passengers, target, stop_names, scale, radius):
    fig, axes = plt.subplots(2, 2, figsize = (9.2, 7.4), facecolor = SURFACE)
    boards, alights = records[records.role == "board"], records[records.role == "alight"]
    hours = np.arange(24)

    ax = axes[0, 0]
    _style(ax, "Passengers by hour")
    ax.bar(hours - 0.2, [(boards.hour == h).sum() * scale for h in hours], width = 0.4, color = BLUE, label = "boarding")
    ax.bar(hours + 0.2, [(alights.hour == h).sum() * scale for h in hours], width = 0.4, color = ORANGE, label = "alighting")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon = False, fontsize = 8)

    ax = axes[0, 1]
    _style(ax, "Where the trips start (boarding) and end (alighting)")
    ax.grid(axis = "x", color = GRID, linewidth = 0.8); ax.grid(axis = "y", visible = False)
    start = boards[boards.other_kind == "start"].other_type.map(activity_group).value_counts() * scale
    end   = alights[alights.other_kind == "end"].other_type.map(activity_group).value_counts() * scale
    transfer_in, transfer_out = (boards.other_kind == "transfer").sum() * scale, (alights.other_kind == "transfer").sum() * scale
    labels = list(dict.fromkeys(list(start.index) + list(end.index)))
    y = np.arange(len(labels))
    ax.barh(y - 0.2, [start.get(l, 0) for l in labels], height = 0.4, color = BLUE, label = "boarding: trip starts at")
    ax.barh(y + 0.2, [end.get(l, 0) for l in labels], height = 0.4, color = ORANGE, label = "alighting: trip ends at")
    ax.set_yticks(y, labels); ax.invert_yaxis()
    ax.legend(frameon = False, fontsize = 8, loc = "lower right")
    ax.text(0, -0.12, f"transfers: {transfer_in:.0f} boardings, {transfer_out:.0f} alightings", transform = ax.transAxes,
            fontsize = 8, color = INK_SECONDARY)

    ax = axes[1, 0]
    _style(ax, "Walking distance between the stop and the trip's start / end")
    edges = np.arange(0, radius + CELL * 2, CELL * 2)
    for data, color, label in ((boards.walk_m.dropna(), BLUE, "boarding"), (alights.walk_m.dropna(), ORANGE, "alighting")):
        counts, _ = np.histogram(data, bins = edges)
        ax.step(edges[:-1], counts * scale, where = "post", color = color, linewidth = 2, label = label)
    ax.set_xlabel("m (straight line)", fontsize = 8, color = INK_SECONDARY)
    ax.legend(frameon = False, fontsize = 8)

    ax = axes[1, 1]
    _style(ax, "Stops around: passengers per day")
    ax.grid(axis = "x", color = GRID, linewidth = 0.8); ax.grid(axis = "y", visible = False)
    total = passengers.groupby("code").n.sum().sort_values(ascending = False).head(10) * scale
    colors = [BLUE if code == target else GRAY for code in total.index]
    ax.barh(range(len(total)), total.values, color = colors, height = 0.6)
    ax.set_yticks(range(len(total)), [stop_names.get(code, code) for code in total.index]); ax.invert_yaxis()

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format = "png", dpi = 130, facecolor = SURFACE)
    plt.close(fig)

    return base64.b64encode(buffer.getvalue()).decode()


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def _weight(count, maximum):
    return 1.5 + 7.0 * math.sqrt(count / maximum)


def build_map(output_path, center, target, target_name, records, passengers, cells, agents, stops, stop_names, near_codes,
              scale, radius, chart_png, extent_path = None):
    to_wgs = Transformer.from_crs(visualize.NETWORK_CRS, "EPSG:4326", always_xy = True)

    def latlon(x, y):
        lon, lat = to_wgs.transform(x, y)
        return (lat, lon)

    m = folium.Map(location = latlon(*center), zoom_start = 16, tiles = None, prefer_canvas = True)
    folium.TileLayer(tiles = visualize.TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", control = False).add_to(m)

    for ring, dash in ((radius, "6,6"), (300, "2,6")):
        folium.Circle(latlon(*center), radius = ring, color = "#444444", weight = 1, fill = False, dash_array = dash,
                      tooltip = f"{ring:.0f} m").add_to(m)

    boards, alights = records[records.role == "board"], records[records.role == "alight"]

    # Where the trips of the passengers start and end
    for role, frame, title, show in (("board", boards, "boarding: where the trips start", True),
                                     ("alight", alights, "alighting: where the trips end", False)):
        placed = frame[frame.other_xy.notna()]
        layer  = folium.FeatureGroup(name = f"{title} ({len(placed) * scale:.0f} passengers)", show = show)
        grouped = defaultdict(Counter)
        for row in placed.itertuples():
            cell = (round(row.other_xy[0] / CELL) * CELL, round(row.other_xy[1] / CELL) * CELL)
            grouped[cell][activity_group(row.other_type)] += 1

        maximum = max((sum(c.values()) for c in grouped.values()), default = 1)
        for cell, counter in sorted(grouped.items(), key = lambda item: -sum(item[1].values())):
            total, activity = sum(counter.values()), counter.most_common(1)[0][0]
            color = ACTIVITY_COLORS.get(activity, OTHER_ACTIVITY_COLOR)
            detail = ", ".join(f"{a} {n * scale:.0f}" for a, n in counter.most_common())
            if total * scale >= 20:
                folium.PolyLine([latlon(*cell), latlon(*center)], color = color, weight = 1, opacity = 0.3,
                                dash_array = "3,5").add_to(layer)
            folium.CircleMarker(
                latlon(*cell), radius = 3 + 11 * math.sqrt(total / maximum), color = "#ffffff", weight = 1.5, fill = True,
                fill_color = color, fill_opacity = 0.85, tooltip = f"{total * scale:.0f} passengers ({detail})",
            ).add_to(layer)
        layer.add_to(m)

    # Transfers: the stop where the agents change from / to another line, joined to the platform of the target stop
    transfers = records[records.other_kind == "transfer"]
    for role, title, show in (("board", "transfers in: from another line to the stop", True),
                              ("alight", "transfers out: from the stop to another line", True)):
        frame   = transfers[transfers.role == role]
        grouped = frame.groupby(["other_stop", "other_type", "other_mode", "platform"]).size().reset_index(name = "n")
        layer   = folium.FeatureGroup(name = f"{title} ({grouped.n.sum() * scale:.0f} passengers)", show = show)
        maximum = max(grouped.n.max() if len(grouped) else 1, 1)

        for row in grouped.itertuples():
            if stops.get(row.other_stop, {}).get("x") is None or stops.get(row.platform, {}).get("x") is None:
                continue

            other, platform = stops[row.other_stop], stops[row.platform]
            walk  = math.hypot(other["x"] - platform["x"], other["y"] - platform["y"])
            color = visualize.MODE_COLORS.get(row.other_mode, visualize.DEFAULT_MODE_COLOR)
            label = (f"{row.other_type} ({row.other_mode}) at {other['name']} → {target_name}" if role == "board"
                     else f"{target_name} → {row.other_type} ({row.other_mode}) at {other['name']}")

            folium.PolyLine(
                [latlon(other["x"], other["y"]), latlon(platform["x"], platform["y"])], color = color,
                weight = _weight(row.n, maximum), opacity = 0.8, dash_array = "6,6",
                tooltip = f"{label}: {row.n * scale:.0f} passengers, {walk:.0f} m walk",
            ).add_to(layer)
            folium.CircleMarker(
                latlon(other["x"], other["y"]), radius = 4 + 8 * math.sqrt(row.n / maximum), color = color, weight = 2,
                fill = True, fill_color = "#ffffff", fill_opacity = 0.9,
                tooltip = f"{other['name']}, {row.other_type}: {row.n * scale:.0f} transfers",
            ).add_to(layer)
        layer.add_to(m)

    # Rides taken from / to the stop
    for role, frame, title, color, show in (("board", boards, "rides after boarding", BLUE, True),
                                            ("alight", alights, "rides before alighting", ORANGE, False)):
        segments = Counter()
        for row in frame.itertuples():
            for stop_a, stop_b in zip(row.ride_stops[:-1], row.ride_stops[1:]):
                segments[(stop_a, stop_b, row.line)] += 1

        layer   = folium.FeatureGroup(name = title, show = show)
        maximum = max(segments.values(), default = 1)
        for (stop_a, stop_b, line), count in segments.items():
            if stops.get(stop_a, {}).get("x") is not None and stops.get(stop_b, {}).get("x") is not None:
                folium.PolyLine(
                    [latlon(stops[stop_a]["x"], stops[stop_a]["y"]), latlon(stops[stop_b]["x"], stops[stop_b]["y"])],
                    color = color, weight = _weight(count, maximum), opacity = 0.75,
                    tooltip = f"{line}: {stops[stop_a]['name']} → {stops[stop_b]['name']}: {count * scale:.0f}",
                ).add_to(layer)
        layer.add_to(m)

    # The activities that take place around the stop
    layer = folium.FeatureGroup(name = "activities around the stop", show = False)
    maximum = max(cells.values(), default = 1)
    for (activity, cell), count in cells.items():
        color = ACTIVITY_COLORS.get(activity_group(activity), OTHER_ACTIVITY_COLOR)
        folium.CircleMarker(
            latlon(*cell), radius = 2 + 6 * math.sqrt(count / maximum), color = color, weight = 0, fill = True,
            fill_color = color, fill_opacity = 0.5, tooltip = f"{activity}: {count * scale:.0f} activities",
        ).add_to(layer)
    layer.add_to(m)

    # The stops around
    layer = folium.FeatureGroup(name = "stops around", show = True)
    per_stop = passengers.groupby("code").n.sum()
    maximum  = per_stop.max()
    for code in near_codes:
        members = [s for s in stops.values() if s.get("x") is not None and stop_code(s["id"]) == code]
        if not members or code not in per_stop.index:
            continue

        x, y = np.mean([s["x"] for s in members]), np.mean([s["y"] for s in members])
        lines = passengers[passengers.code == code].groupby(["line", "side"]).n.sum().unstack(fill_value = 0)
        rows = "".join(f"<tr><td>{line}</td><td align='right'>{r.get('boardings', 0) * scale:.0f}</td>"
                       f"<td align='right'>{r.get('alightings', 0) * scale:.0f}</td></tr>" for line, r in lines.iterrows())
        is_target = code == target
        folium.CircleMarker(
            latlon(x, y), radius = 4 + 18 * math.sqrt(per_stop[code] / maximum), color = INK if is_target else "#ffffff",
            weight = 3 if is_target else 1.5, fill = True, fill_color = BLUE if is_target else GRAY, fill_opacity = 0.5,
            popup = folium.Popup(f"<b>{stop_names.get(code, code)}</b>: {per_stop[code] * scale:.0f} passengers"
                                 f"<table><tr><th>line</th><th>board</th><th>alight</th></tr>{rows}</table>", max_width = 320),
            tooltip = f"{stop_names.get(code, code)}: {per_stop[code] * scale:.0f}",
        ).add_to(layer)
    layer.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)

    legend = "".join(
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:{color};margin:0 4px 0 10px'></span>{name}"
        for name, color in (*ACTIVITY_COLORS.items(), ("other", OTHER_ACTIVITY_COLOR))
    )
    transfer_rows = "".join(
        f"<tr><td>{other if role == 'board' else line}</td><td>→</td><td>{line if role == 'board' else other}</td>"
        f"<td>{'boarding here' if role == 'board' else 'alighting here'}</td><td align='right'>{n * scale:.0f}</td></tr>"
        for (role, other, line), n in transfers.groupby(["role", "other_type", "line"]).size().sort_values(ascending = False).items()
    )
    area = ", ".join(f"{a} {n * scale:.0f}" for a, n in sorted(agents.items(), key = lambda item: -item[1])[:6])
    panel = f"""
    <details open style="position:fixed;top:10px;right:10px;z-index:9999;width:460px;max-height:94vh;overflow:auto;
      background:{SURFACE};color:{INK};border:1px solid {GRID};border-radius:8px;padding:10px 14px;font:13px/1.4 system-ui,sans-serif">
      <summary style="cursor:pointer;font-weight:600">{target_name}: {len(boards) * scale:.0f} boardings, {len(alights) * scale:.0f} alightings</summary>
      <p style="color:{INK_SECONDARY};margin:6px 0">Counts are scaled to the full population. Circles are the places where the trips of the
      passengers start (boarding layer) or end (alighting layer), colored by activity:{legend}.</p>
      <img src="data:image/png;base64,{chart_png}" style="width:100%">
      <p style="color:{INK_SECONDARY};margin:6px 0 2px">Transfers ({len(transfers) * scale:.0f}, {len(transfers) / max(len(records), 1):.0%} of the
      passengers), all of them: the line they arrive on, the line they continue on, and whether the change happens by boarding or by alighting at this stop.</p>
      <div style="max-height:220px;overflow:auto"><table style="border-collapse:collapse;width:100%"><tr><th align="left">from line</th><th></th>
      <th align="left">to line</th><th align="left">where</th><th>passengers</th></tr>{transfer_rows}</table></div>
      <p style="color:{INK_SECONDARY};margin:6px 0">Agents with an activity within {radius:.0f} m: {area}.</p>
    </details>"""
    m.get_root().html.add_child(folium.Element(panel))

    m.save(output_path)


# ---------------------------------------------------------------------------

def run(simulation_path, input_downsampling = 1.0, stop = "8592052", radius = 600.0, output_dir = None, extent_path = None):
    events_path   = os.path.join(simulation_path, EVENTS_FILE)
    schedule_path = visualize._find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    output_dir    = output_dir or os.path.join(simulation_path, "pt_stop_neighbourhood", stop)
    os.makedirs(output_dir, exist_ok = True)

    scale = 1.0 / input_downsampling

    stops, routes_list = visualize.load_transit_schedule(schedule_path)
    for stop_id, info in stops.items():
        info["id"] = stop_id
    routes = {(route["line_id"], route["route_id"]): route for route in routes_list}

    members = [s for s in stops.values() if s.get("x") is not None and stop_code(s["id"]) == stop]
    if not members:
        raise RuntimeError(f"Stop {stop} not found in the transit schedule {schedule_path}")

    center = (float(np.mean([s["x"] for s in members])), float(np.mean([s["y"] for s in members])))
    target_name = members[0]["name"]

    stop_names = {}
    for s in stops.values():
        if s.get("x") is not None:
            stop_names.setdefault(stop_code(s["id"]), s["name"])

    near_codes = {
        stop_code(s["id"]) for s in stops.values()
        if s.get("x") is not None and math.hypot(s["x"] - center[0], s["y"] - center[1]) <= radius
    }
    print(f"{target_name} ({stop}): {len(members)} platforms, {len(near_codes)} stops within {radius:.0f} m")

    print("Reading the public transport events ...")
    counts, arrivals, service, at_target = read_pt_events(events_path, stops, routes, near_codes, stop)
    passengers = pd.DataFrame(
        [(code, line, hour, side, n) for (code, line, hour, side), n in counts.items()],
        columns = ["code", "line", "hour", "side", "n"],
    )
    print(f"  -> {len(at_target)} agents board or alight at {target_name}")

    print("Following these agents ...")
    trips = follow_persons(events_path, at_target)

    direction_names = {}
    for route in routes_list:
        ids = route["stop_ids"]
        if ids:
            direction_names[f"{ids[0]}|{ids[-1]}"] = f"{stops[ids[0]]['name']} → {stops[ids[-1]]['name']}"

    records, trip_table = passengers_of_the_stop(trips, stop, service, arrivals, routes, stops, center, direction_names)
    print(f"  -> {len(records)} boardings and alightings at the stop in {len(trip_table)} trips")

    print("Counting the activities around the stop ...")
    cells, agents = activities_in_area(events_path, center, radius)

    # The tables
    records.drop(columns = ["other_xy", "ride_stops"]).assign(passengers = scale).to_csv(os.path.join(output_dir, "stop_passengers.csv"), index = False)
    trip_table.to_csv(os.path.join(output_dir, "stop_trips.csv"), index = False)
    (records[records.other_kind == "transfer"]
        .groupby(["role", "line", "other_type", "other_mode", "other_stop", "platform"]).size().mul(scale).rename("passengers").reset_index()
        .sort_values("passengers", ascending = False).to_csv(os.path.join(output_dir, "stop_transfers.csv"), index = False))
    passengers.assign(passengers = passengers["n"] * scale).drop(columns = "n").to_csv(os.path.join(output_dir, "stops_around.csv"), index = False)

    boards, alights = records[records.role == "board"], records[records.role == "alight"]
    print("Passengers at the stop (scaled to the full population):")
    print(records.groupby(["role", "line", "heading"]).size().mul(scale).round(0).to_string())
    print("Boarding: the trips start at")
    print(boards[boards.other_kind == "start"].other_type.value_counts().mul(scale).round(0).head(8).to_string())
    print("Alighting: the trips end at")
    print(alights[alights.other_kind == "end"].other_type.value_counts().mul(scale).round(0).head(8).to_string())
    print(f"Transfers: {(boards.other_kind == 'transfer').sum() * scale:.0f} boardings, "
          f"{(alights.other_kind == 'transfer').sum() * scale:.0f} alightings")
    print("Median walking distance to / from the stop (m):",
          round(boards.walk_m.median()), "/", round(alights.walk_m.median()))
    print("Agents with an activity within the radius:", {a: round(n * scale) for a, n in sorted(agents.items(), key = lambda i: -i[1])[:8]})

    chart_png = charts(records, passengers, stop, stop_names, scale, radius)
    output_path = os.path.join(output_dir, "stop_neighbourhood_map.html")
    build_map(output_path, center, stop, target_name, records, passengers, cells, agents, stops, stop_names, near_codes,
              scale, radius, chart_png, extent_path)
    print(f"Map written to {output_path}")

    return output_dir


def main(argv = None):
    parser = argparse.ArgumentParser(description = "Map of everything that happens around a public transport stop")
    parser.add_argument("--simulation-path", required = True, help = "MATSim output folder (events and transit schedule)")
    parser.add_argument("--input-downsampling", type = float, required = True)
    parser.add_argument("--stop", default = "8592052", help = "GTFS stop code (default: Lausanne, Grancy)")
    parser.add_argument("--radius", type = float, default = 600.0, help = "Radius of the area around the stop (m)")
    parser.add_argument("--output-dir", default = None)
    args = parser.parse_args(argv)

    run(args.simulation_path, args.input_downsampling, args.stop, args.radius, args.output_dir)


if __name__ == "__main__":
    main()
