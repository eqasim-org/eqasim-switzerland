import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "transit_schedule"))
sys.path.insert(0, os.path.join(HERE, "..", "TPG_comparison"))

import visualize # pyright: ignore[reportMissingImports]
import passenger_profiles # pyright: ignore[reportMissingImports]
import line_asymmetry # pyright: ignore[reportMissingImports]
import gtfs_utils # pyright: ignore[reportMissingImports]
from comparison_passenger_counts_lausanne import TL_TO_MATSIM_LINE, SUBMODE_NAMES, SUBMODE_OF_LINE

# the MATSim line -> the TL line
MATSIM_TO_TL_LINE = {matsim: tl for tl, matsim in TL_TO_MATSIM_LINE.items()}

# Folder of the outputs of the TL comparisons in the simulation folder (comparison_passenger_counts_lausanne)
TL_OUTPUT_FOLDER = "pt_comparison_tl_2024"

MAX_SNAP_DISTANCE = 250.0

# Elevations of the network nodes (relative to data_path), for the uphill / downhill connections of the asymmetry map
ELEVATION_PATH = os.path.join("osm", "elevations.csv")


def configure(context):
    # TL stops are placed at the stops of the GTFS of the simulation
    context.stage("data.gtfs.cleaned")
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

    context.config("analysis.pt.tl_processed_counts_path", default = "Lausanne_passenger_counts/tl data 2024/processed")

    # Rebuild the processed TL counts from the raw ones first (analysis.pt.TL_comparison.tl_2024)
    context.config("analysis.pt.tl_rebuild_counts", default = False)
    if context.config("analysis.pt.tl_rebuild_counts"):
        context.stage("analysis.pt.TL_comparison.tl_2024")
    context.config("analysis.pt.night_hours",       default = 5)

    context.config("analysis.pt.line_comparison_output_id", default = None)  # the run of output_id by default
    context.config("analysis.pt.tl_line_comparison_lines",  default = [])    # MATSim line names; empty: all the lines of the TL data


def execute(context):
    data_path = context.config("data_path")
    output_id = context.config("analysis.pt.line_comparison_output_id") or context.config("output_id")
    simulation_path = os.path.join(context.config("output_path"), output_id, context.config("simulation_directory"))

    lines = context.config("analysis.pt.tl_line_comparison_lines")
    if isinstance(lines, str):
        lines = [line.strip() for line in lines.split(",") if line.strip()]

    output_dir = run(
        simulation_path, read_tl_table(os.path.join(data_path, context.config("analysis.pt.tl_processed_counts_path"))),
        "%s/output" % context.path("data.gtfs.cleaned"),
        input_downsampling = context.config("input_downsampling"), lines = lines,
        night_hours = context.config("analysis.pt.night_hours"),
        output_dir = os.path.join(simulation_path, TL_OUTPUT_FOLDER),
        reference_name = "TL", prefix = "tl", elevation_path = os.path.join(data_path, ELEVATION_PATH),
    )

    return dict(done = True, path = output_dir)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Stop codes of the platforms of a stop (French quays, which have their own ids) merged into one code: register_stops
STOP_ALIASES = {}


def base_code(stop_id):
    """The stop of a platform, a MATSim stop facility or a GTFS stop: the UIC number of the Swiss stops (8592910:0:W,
    Parent8592910, 8592910.link:123 -> 8592910; the SLOIDs ch:1:sloid:92693:0:1 -> 8592693), else the whole id without the
    MATSim link (FR:74085:ZE:58041:HAUTExSAVOIE, StopPoint:OCETrain:::TER-87745588)."""
    stop = str(stop_id).split(".link:")[0].split(".")[0]
    if stop.startswith("Parent"):
        stop = stop[len("Parent"):]
    parts = stop.split(":")
    if stop.startswith("ch:1:sloid:") and len(parts) > 3:
        return "85" + parts[3].zfill(5) if len(parts[3]) <= 5 else ":".join(parts[:4])
    return parts[0] if parts[0].isdigit() else stop


def code_of(stop_id):
    code = base_code(stop_id)
    return STOP_ALIASES.get(code, code)


def register_stops(stops):
    """The platforms that are not Swiss stops (no UIC number, e.g. the French quays) with the same name within
    MAX_SNAP_DISTANCE of each other get one code, so the two directions of a line share their stops."""
    STOP_ALIASES.clear()
    by_name = defaultdict(dict)
    for stop_id, info in stops.items():
        code = base_code(stop_id)
        if not code.isdigit() and info["x"] is not None:
            by_name[info["name"]].setdefault(code, (info["x"], info["y"]))

    for platforms in by_name.values():
        kept = []
        for code, (x, y) in sorted(platforms.items()):
            main = next((c for c, (cx, cy) in kept if math.hypot(x - cx, y - cy) <= MAX_SNAP_DISTANCE), None)
            if main is None:
                kept.append((code, (x, y)))
            else:
                STOP_ALIASES[code] = main


def submode_of(line, mode):
    return SUBMODE_OF_LINE.get(line) or SUBMODE_NAMES.get(mode, mode)


def stop_coordinates(gtfs_path, codes):
    with gtfs_utils.open_gtfs_file(gtfs_path, "stops.txt") as f:
        stops = pd.read_csv(f, dtype = {"stop_id": str}, usecols = ["stop_id", "stop_lat", "stop_lon"])

    stops["code"] = stops["stop_id"].map(code_of)
    stops = stops[stops["code"].isin(codes)].groupby("code")[["stop_lat", "stop_lon"]].mean()
    to_lv95 = Transformer.from_crs("EPSG:4326", visualize.NETWORK_CRS, always_xy = True)

    return {code: to_lv95.transform(row.stop_lon, row.stop_lat) for code, row in stops.iterrows()}


def read_tl_table(processed_dir):
    candidates = sorted(
        path for path in glob.glob(os.path.join(processed_dir, "tl2024_agg_workday_*.csv"))
        if re.fullmatch(r"tl2024_agg_workday_\d{8}\.csv", os.path.basename(path))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed TL stats in {processed_dir}: run analysis.pt.TL_comparison.tl_2024 first")

    print(f"TL counts: {candidates[-1]}")
    tl = pd.read_csv(candidates[-1])
    tl["tl_line"]   = tl["line_direction"].str.rsplit("_", n = 1).str[0]
    tl["line"]      = tl["tl_line"].replace(TL_TO_MATSIM_LINE)
    tl["direction"] = tl["line_direction"].str[-1]
    tl["code"]      = tl["gtfs_code"].astype(str).str.split(".").str[0]

    return tl


def choose_line(routes_of_name, tl_codes, tl_xy, xy):
    """Among the transit lines (line_id) of a name, the one that serves most of the TL stops (within 250 m)."""
    scores = {}
    for route in routes_of_name:
        scores.setdefault(route["line_id"], set()).update(code_of(s) for s in route["stop_ids"])

    best = None
    for line_id, stops in scores.items():
        points = [xy[c] for c in stops if c in xy]
        served = sum(
            any(math.hypot(tl_xy[code][0] - p[0], tl_xy[code][1] - p[1]) <= MAX_SNAP_DISTANCE for p in points)
            for code in tl_codes if code in tl_xy
        )
        if best is None or served > best[0]:
            best = (served, line_id)

    return best[1], best[0], len(scores)


def line_order(routes, line_id):
    """Stop codes of the longest route of the transit line."""
    candidates = [route for route in routes if route["line_id"] == line_id]
    order = []
    for stop_id in max(candidates, key = lambda route: len(route["stop_ids"]))["stop_ids"]:
        if not order or order[-1] != code_of(stop_id):
            order.append(code_of(stop_id))

    return order


def prepare_counts(simulation_path, scale, night_hours):
    counts = pd.read_csv(os.path.join(simulation_path, "pt_passenger_counts.csv.gz"), sep = ";",
                         usecols = ["stop_id", "line_id", "line_name", "route_direction", "time_bin", "boardings", "alightings"])
    counts["hour"] = counts["time_bin"].str.extract(r"^(\d{1,2})").astype(int)

    if night_hours > 0:
        counts = counts[(counts["hour"] >= night_hours) & (counts["hour"] < 30)].copy()
        counts["hour"] %= 24
    else:
        counts = counts[counts["hour"] < 24]

    counts["code"] = counts["stop_id"].map(code_of)
    counts[["boardings", "alightings"]] *= scale

    return counts


def read_matsim(counts, line, line_id, order, names):
    """{orientation: DataFrame(code, hour, boardings, alightings)} of the transit line; orientation is fwd when the route
    runs in the order of the longest route of the line, rev otherwise."""
    index = {names[code]: k for k, code in enumerate(order)}
    frame = counts[counts["line_id"] == line_id].copy()

    def orientation(direction):
        origin, _, destination = str(direction).partition("->")
        if origin in index and destination in index:
            return "fwd" if index[origin] < index[destination] else "rev"
        return None

    frame["orientation"] = frame["route_direction"].map(orientation)
    dropped = frame["orientation"].isna().sum()
    if dropped:
        print(f"  {line}: {dropped} count rows of routes with an unknown direction left out")

    return {
        name: group.groupby(["code", "hour"])[["boardings", "alightings"]].sum().reset_index()
        for name, group in frame.dropna(subset = ["orientation"]).groupby("orientation")
    }


def _cumulative_flow(table, order):
    """Passengers on board after each stop of the order (daily): the boardings minus the alightings so far."""
    daily = table.groupby("code")[["boardings", "alightings"]].sum()
    flow, total = [], 0.0
    for code in order[:-1]:
        if code in daily.index:
            total += daily.loc[code, "boardings"] - daily.loc[code, "alightings"]
        flow.append(total)

    return np.array(flow)


def _hourly_flow(table, order):
    """Passengers on board after each stop of the order for each hour (array stops - 1 x 24): the boardings minus the
    alightings of that hour so far. The passengers that board in an hour and alight in the next one are not followed,
    so this is an approximation of the flow of the hour, computed the same way for both sources."""
    hours = np.arange(24)
    per_hour = {code: group.set_index("hour")[["boardings", "alightings"]].reindex(hours, fill_value = 0.0)
                for code, group in table.groupby("code")}
    flow, total = [], np.zeros(24)
    for code in order[:-1]:
        if code in per_hour:
            total = total + per_hour[code]["boardings"].to_numpy() - per_hour[code]["alightings"].to_numpy()
        flow.append(np.clip(total, 0, None))

    return np.array(flow)


def read_tl(tl_line, line, order, xy, tl_xy):
    """{orientation: DataFrame(code, hour, boardings, alightings)}; the TL stops are moved to the closest stop of the line
    (within 250 m) and the orientation is the one that gives a consistent flow (never negative, back to 0)."""
    frame = tl_line

    mapping = {}
    for code in set(frame["code"]):
        if code in order:
            mapping[code] = code
        elif code in tl_xy:
            distances = [(math.hypot(tl_xy[code][0] - xy[c][0], tl_xy[code][1] - xy[c][1]), c) for c in order if c in xy]
            if distances and min(distances)[0] <= MAX_SNAP_DISTANCE:
                mapping[code] = min(distances)[1]

    unmatched = sorted(set(frame["code"]) - set(mapping))
    if unmatched:
        print(f"  {line}: counted stops without a stop of the line within {MAX_SNAP_DISTANCE:.0f} m, left out: {unmatched}")

    frame = frame.assign(code = frame["code"].map(mapping)).dropna(subset = ["code"])
    result = {}

    for direction, group in frame.groupby("direction"):
        table = (group.groupby(["code", "hour"])[["boardings_raw_mean", "alightings_raw_mean"]].sum()
                 .rename(columns = {"boardings_raw_mean": "boardings", "alightings_raw_mean": "alightings"}).reset_index())

        best = None
        for orientation, stops_order in (("fwd", order), ("rev", order[::-1])):
            flow  = _cumulative_flow(table, stops_order)
            score = abs(flow[-1]) + abs(min(flow.min(), 0))
            if best is None or score < best[0]:
                best = (score, orientation)

        if best[1] in result:  # two TL directions on the same side: add them up
            table = pd.concat([result[best[1]], table]).groupby(["code", "hour"], as_index = False)[["boardings", "alightings"]].sum()
        result[best[1]] = table

    return result


def _hours_by_stop(table):
    """{code: [boardings per hour, alightings per hour]}, rounded to keep the page small."""
    def clean(values):
        return [int(v) if float(v).is_integer() else round(float(v), 1) for v in values]

    result = {}
    for code, group in table.groupby("code"):
        boardings, alightings = np.zeros(24), np.zeros(24)
        boardings[group["hour"].to_numpy()]  = group["boardings"].to_numpy()
        alightings[group["hour"].to_numpy()] = group["alightings"].to_numpy()
        result[code] = [clean(boardings), clean(alightings)]

    return result


# ---------------------------------------------------------------------------

def run(simulation_path, reference, gtfs_path, input_downsampling = 1.0, lines = (), night_hours = 5, output_dir = None,
        reference_name = "TL", prefix = "tl", elevation_path = None):
    """Map comparing the passengers of the lines in MATSim and in the counts of an operator.
    reference: the counts of the operator (read_tl_table, or read_tpg_table of analysis.pt.TPG_comparison.line_comparison),
    columns line (MATSim line name), direction, code (stop code), hour, boardings_raw_mean and alightings_raw_mean.
    The connections between consecutive stops are drawn as straight lines.
    Writes <prefix>_line_stops_and_flows.csv, <prefix>_line_hourly_flows.csv and <prefix>_line_comparison_map.html, and the
    asymmetry map of the same lines (line_asymmetry: <prefix>_line_asymmetry_map.html and its CSVs); elevation_path: the
    elevation cache of the pipeline (osm/elevations.csv), for the uphill / downhill connections."""
    schedule_path = visualize._find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    output_dir    = output_dir or os.path.join(simulation_path, f"pt_line_comparison_{prefix}")
    os.makedirs(output_dir, exist_ok = True)
    scale = 1.0 / input_downsampling

    stops, routes = visualize.load_transit_schedule(schedule_path)
    register_stops(stops)

    names, coordinates = {}, defaultdict(list)
    for stop_id, info in stops.items():
        names.setdefault(code_of(stop_id), info["name"])
        if info["x"] is not None:
            coordinates[code_of(stop_id)].append((info["x"], info["y"]))
    xy = {code: tuple(np.mean(points, axis = 0)) for code, points in coordinates.items()}

    routes_by_name = defaultdict(list)
    for route in routes:
        routes_by_name[route["line_name"]].append(route)

    tl = reference
    tl_xy = stop_coordinates(gtfs_path, set(tl["code"]))

    wanted = list(lines) if lines else sorted(set(tl["line"]))
    print("Reading the MATSim passenger counts ...")
    counts = prepare_counts(simulation_path, scale, night_hours)

    to_wgs = Transformer.from_crs(visualize.NETWORK_CRS, "EPSG:4326", always_xy = True)
    data, tables, hourly_rows, asymmetry = {}, [], [], {}

    for line in wanted:
        tl_line = tl[tl["line"] == line]
        if line not in routes_by_name or tl_line.empty:
            print(f"Line {line}: {'not in the transit schedule' if line not in routes_by_name else 'not in the ' + reference_name + ' data'}, skipped")
            continue

        line_id, served, candidates = choose_line(routes_by_name[line], set(tl_line["code"]), tl_xy, xy)
        if candidates > 1:
            print(f"Line {line}: {candidates} transit lines with this name, {line_id} serves {served} of the {tl_line['code'].nunique()} {reference_name} stops")
        if served == 0:
            print(f"Line {line}: no transit line serves a {reference_name} stop, skipped")
            continue

        order = line_order(routes, line_id)
        matsim = read_matsim(counts, line, line_id, order, names)
        tl_data = read_tl(tl_line, line, order, xy, tl_xy)
        mode = next(route["mode"] for route in routes_by_name[line] if route["line_id"] == line_id)
        titles = {"fwd": f"{names[order[0]]} → {names[order[-1]]}", "rev": f"{names[order[-1]]} → {names[order[0]]}"}

        sources = {"matsim": matsim, "tl": tl_data}
        daily_flows, hourly_flows = {name: {} for name in sources}, {name: {} for name in sources}
        for orientation in ("fwd", "rev"):
            stops_order = order if orientation == "fwd" else order[::-1]
            flows = {name: (_cumulative_flow(source[orientation], stops_order) if orientation in source else np.full(len(order) - 1, np.nan))
                     for name, source in (("matsim", matsim), ("tl", tl_data))}

            for k, code in enumerate(stops_order):
                row = dict(line = line, line_id = line_id, direction = titles[orientation], stop = names[code], stop_code = code, position = k)
                for name, source in (("matsim", matsim), ("tl", tl_data)):
                    table = source.get(orientation)
                    daily = table[table["code"] == code][["boardings", "alightings"]].sum() if table is not None else None
                    column = name if name == "matsim" else prefix  # columns of the reference: <measure>_<prefix>
                    row[f"boardings_{column}"]    = np.nan if daily is None else daily["boardings"]
                    row[f"alightings_{column}"]   = np.nan if daily is None else daily["alightings"]
                    row[f"flow_to_next_{column}"] = flows[name][k] if k < len(order) - 1 else np.nan
                tables.append(row)

            for name, source in (("matsim", matsim), ("tl", tl_data)):
                if orientation not in source:
                    continue
                flow = _hourly_flow(source[orientation], stops_order)
                daily_flows[name][orientation], hourly_flows[name][orientation] = flows[name], flow
                for k, (a, b) in enumerate(zip(stops_order[:-1], stops_order[1:])):
                    hourly_rows += [dict(line = line, direction = titles[orientation], from_stop = names[a], to_stop = names[b],
                                         source = name if name == "matsim" else prefix, hour = h, passengers = flow[k][h]) for h in range(24)]

        totals = {name: float(sum(t["boardings"].sum() for t in source.values())) for name, source in (("matsim", matsim), ("tl", tl_data))}
        data[line] = {
            "submode": submode_of(line, mode), "line_id": line_id, "titles": titles, "order": order,
            "stops": {code: [names[code], *reversed(to_wgs.transform(*xy[code]))] for code in order if code in xy},
            "matsim": {o: _hours_by_stop(t) for o, t in matsim.items()},
            "tl": {o: _hours_by_stop(t) for o, t in tl_data.items()},
            "totals": totals,
        }
        asymmetry[line] = {
            **{k: data[line][k] for k in ("submode", "line_id", "titles", "order", "stops")},
            "src": {("ref" if name == "tl" else name): line_asymmetry.source_data(
                daily_flows[name], hourly_flows[name],
                {o: t.groupby("hour")["boardings"].sum().reindex(range(24), fill_value = 0.0).to_numpy() for o, t in source.items()},
                _hours_by_stop(pd.concat(source.values()).groupby(["code", "hour"], as_index = False)[["boardings", "alightings"]].sum())
                if source else {},
            ) for name, source in sources.items()},
        }

    table = pd.DataFrame(tables)
    for measure in ("boardings", "alightings"):
        table[f"{measure}_ratio"] = table[f"{measure}_matsim"] / table[f"{measure}_{prefix}"]
    table["flow_ratio"] = table["flow_to_next_matsim"] / table[f"flow_to_next_{prefix}"]
    table.to_csv(os.path.join(output_dir, f"{prefix}_line_stops_and_flows.csv"), index = False)
    pd.DataFrame(hourly_rows).to_csv(os.path.join(output_dir, f"{prefix}_line_hourly_flows.csv"), index = False)

    # who travels on the lines: the attributes of the simulated passengers, on the whole line and on each connection
    profiles = passenger_profiles.build(simulation_path, routes, {d["line_id"] for d in data.values()}, code_of)
    if profiles is not None:
        for d in data.values():
            d["profile"] = profiles["lines"].get(d["line_id"])
        profiles = {k: v for k, v in profiles.items() if k != "lines"}

    output_path = os.path.join(output_dir, f"{prefix}_line_comparison_map.html")
    build_page(output_path, data, reference_name, profiles)

    # asymmetries between the directions and between the morning and the evening
    elevations = line_asymmetry.stop_elevations(elevation_path, {c: xy[c] for d in asymmetry.values() for c in d["order"] if c in xy})
    for d in asymmetry.values():
        d["z"] = {c: elevations[c] for c in d["order"] if c in elevations}
        d["grade"] = line_asymmetry.segment_grades(d["order"], xy, elevations) if elevations else None
    line_asymmetry.write(output_dir, prefix, reference_name, asymmetry)

    summary = pd.DataFrame({line: d["totals"] for line, d in data.items()}).T
    summary["ratio"] = summary["matsim"] / summary["tl"]
    print(summary.rename(columns = {"tl": reference_name}).round(2).to_string())
    print(f"{len(data)} lines, map written to {output_path}")

    return output_dir


def build_page(output_path, data, reference_name = "TL", profiles = None):
    payload = json.dumps({"lines": data, "profiles": profiles}, separators = (",", ":"), ensure_ascii = False)
    html = (PAGE.replace("__TILE_URL__", visualize.TILE_URL).replace("__REF__", reference_name)
            .replace("__DATA__", payload.replace("</", "<\\/")))
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Page: left third = menu (line, charts), the rest = map. Everything is drawn in the browser from the hourly counts.
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__REF__ line comparison</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --matsim:#b0242a; --tl:#4a7bb5; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --ink:#f2f1ee; --ink2:#aeada8; --grid:#3a3a38; } }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--ink); font:13px/1.4 system-ui, sans-serif; }
  #app { display:flex; height:100vh; }
  #menu { width:33.333vw; min-width:320px; display:flex; flex-direction:column; border-right:1px solid var(--grid); }
  #top { padding:12px 14px; border-bottom:1px solid var(--grid); flex:none; }
  #graphs { padding:8px 14px 24px; overflow:auto; flex:1; }
  #map { flex:1; }
  h1 { font-size:15px; margin:0 0 8px; }
  h2 { font-size:13px; margin:14px 0 4px; }
  hr.divider { border:0; border-top:2px solid var(--ink2); margin:18px 0 0; opacity:0.6; }
  select, input[type=text] { width:100%; padding:6px 8px; font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--grid); border-radius:6px; }
  input[type=text] { margin-bottom:6px; }
  label { margin-right:12px; white-space:nowrap; }
  .row { margin-top:8px; }
  .legend { height:10px; border-radius:3px; margin-top:8px; }
  .legend-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--ink2); }
  .muted { color:var(--ink2); }
  .key span { display:inline-block; width:10px; height:10px; margin:0 4px 0 8px; border-radius:2px; }
  svg text { fill:var(--ink2); font-family:system-ui, sans-serif; }
  svg .title { fill:var(--ink); font-weight:600; }
  svg .grid { stroke:var(--grid); stroke-width:1; }
  table.t { border-collapse:collapse; font-size:12px; width:100%; }
  table.t td, table.t th { padding:1px 6px; text-align:right; }
  table.t td:first-child, table.t th:first-child { text-align:left; }
  .tip { font-size:12px; }
  .profile { font-size:12px; }
  .profile .pnote { color:var(--ink2); margin:2px 0 4px; }
  .profile .pkey { display:flex; flex-wrap:wrap; gap:4px 12px; color:var(--ink2); margin-top:4px; align-items:center; }
  .profile .pkey i { display:inline-block; vertical-align:middle; margin-right:4px; }
  .profile .pkey .kfill { width:14px; height:8px; border-radius:0 4px 4px 0; background:var(--matsim); }
  .profile .pkey .ktick { width:2px; height:12px; border-radius:1px; background:var(--ink); }
  .profile .pgrid { display:grid; grid-template-columns:1fr 1fr; gap:0 16px; }
  .profile .pdim { margin-top:8px; }
  .profile .pname { font-weight:600; margin-bottom:2px; }
  .profile .prow { display:grid; grid-template-columns:minmax(0, 8.5em) 1fr 2.6em; align-items:center; gap:6px; height:18px; }
  .profile .plab { color:var(--ink2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .profile .ptrack { position:relative; height:8px; border-radius:4px; background:var(--grid); }
  .profile .pfill { position:absolute; left:0; top:0; bottom:0; border-radius:0 4px 4px 0; background:var(--matsim); }
  .profile .ptick { position:absolute; top:-3px; bottom:-3px; width:2px; margin-left:-1px; border-radius:1px; background:var(--ink); }
  .profile .pval { text-align:right; font-variant-numeric:tabular-nums; }
  .profile .periods label { margin-right:8px; }
</style></head>
<body><div id="app">
  <div id="menu">
    <div id="top">
      <h1>Line comparison: MATSim against __REF__</h1>
      <input id="filter" type="text" placeholder="Filter the lines">
      <select id="line"></select>
      <div class="row">
        Colour by:
        <label><input type="radio" name="color" value="ratio" checked> MATSim / __REF__</label>
        <label><input type="radio" name="color" value="matsim"> MATSim</label>
        <label><input type="radio" name="color" value="tl"> __REF__</label>
      </div>
      <div class="row">Display:
        <label><input type="radio" name="measure" value="boardings" checked> boardings</label>
        <label><input type="radio" name="measure" value="alightings"> alightings</label>
        <label><input type="radio" name="measure" value="both"> sum</label>
        <label><input type="checkbox" id="numbers"> Show numbers on the map</label>
      </div>
      <div class="row key"><span style="background:var(--matsim)"></span>MATSim<span style="background:var(--tl)"></span>__REF__</div>
      <div id="summary" class="row"></div>
      <div class="legend" id="gradient"></div>
      <div class="legend-labels" id="legend-labels"></div>
    </div>
    <div id="graphs"></div>
  </div>
  <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PAYLOAD = __DATA__, DATA = PAYLOAD.lines, PROFILES = PAYLOAD.profiles;
const HOURS = Array.from({length: 24}, (_, h) => h);
const CSS = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MATSIM = "#b0242a", TL = "#4a7bb5", NEUTRAL = "#a9a8a3";   // MATSim in red, like the ratio scale where MATSim has too much
// the ratio scale: blue where MATSim misses demand (ratio < 1), mid grey, red where MATSim has too much (ratio > 1)
const TOO_LOW = "#4a7bb5", TOO_HIGH = "#b0242a";
const LANE = 12, SIDE = 8;   // pixels: the lanes keep their spacing at any zoom
const px = (v, lat) => v * 156543.03 * Math.cos(lat * Math.PI / 180) / Math.pow(2, map.getZoom());
const SUBMODE_ORDER = ["metro", "LEB", "train", "tram", "bus"];

let drawn = false;
const map = L.map("map", {preferCanvas: true}).setView([46.52, 6.63], 13);
L.tileLayer("__TILE_URL__", {attribution: "© swisstopo", maxZoom: 19}).addTo(map);
const layer = L.layerGroup().addTo(map);

let current = null;       // name of the selected line
let selection = null;     // {type: "stop", code} or {type: "segment", k}

// ---------------------------------------------------------------- numbers
const sum = a => a.reduce((s, v) => s + v, 0);
const fmt = v => v == null ? "–" : Math.round(v).toLocaleString("en");
const stopHours = (line, source, o, code) => (DATA[line][source][o] || {})[code] || [Array(24).fill(0), Array(24).fill(0)];
const hasSource = (line, source, o) => !!DATA[line][source][o];

function orderOf(line, o) { const order = DATA[line].order; return o === "fwd" ? order : order.slice().reverse(); }

// passengers on board after each stop of the direction, per hour: [stops - 1][24]
function hourlyFlow(line, source, o) {
  const order = orderOf(line, o), flow = [];
  let total = Array(24).fill(0);
  for (const code of order.slice(0, -1)) {
    const [b, a] = stopHours(line, source, o, code);
    total = total.map((t, h) => t + b[h] - a[h]);
    flow.push(total.map(v => Math.max(v, 0)));
  }
  return flow;
}
function dailyFlow(line, source, o) {
  const order = orderOf(line, o), flow = [];
  let total = 0;
  for (const code of order.slice(0, -1)) {
    const [b, a] = stopHours(line, source, o, code);
    total += sum(b) - sum(a);
    flow.push(total);
  }
  return flow;
}
const ratioOf = (m, t) => (t > 0 && m != null) ? m / t : null;

// ---------------------------------------------------------------- colors
function hex(c) { return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); }
function mix(c1, c2, t) { const a = hex(c1), b = hex(c2); return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join(""); }
function ratioColor(r) {
  if (r == null || !isFinite(r) || r <= 0) return "#9a9994";
  const p = 0.5 + 0.5 * Math.max(-1, Math.min(1, Math.log2(r) / 2));
  return p < 0.5 ? mix(TOO_LOW, NEUTRAL, p * 2) : mix(NEUTRAL, TOO_HIGH, (p - 0.5) * 2);
}
const SEQUENTIAL = {matsim: MATSIM, tl: TL};
const countColor = (source, v, max) => mix("#f1f0ec", SEQUENTIAL[source], 0.12 + 0.88 * Math.max(0, Math.min(1, v / (max || 1))));
function colorOf(mode, m, t, max) { return mode === "ratio" ? ratioColor(ratioOf(m, t)) : countColor(mode, mode === "matsim" ? m : t, max); }
function legend(mode, maxFlow, maxStop) {
  const stops = mode === "ratio" ? [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1].map(t => ratioColor(Math.pow(2, 2 * t)))
                                 : [0, 0.25, 0.5, 0.75, 1].map(t => countColor(mode, t, 1));
  byId("gradient").style.background = "linear-gradient(to right, " + stops.join(",") + ")";
  byId("legend-labels").innerHTML = mode === "ratio"
    ? "<span>&le; 1/4 (MATSim misses demand)</span><span>1</span><span>&ge; 4 (MATSim has too much)</span>"
    : `<span>0</span><span>lines: ${fmt(maxFlow)} on board; circles: ${fmt(maxStop)} passengers</span>`;
}

// ---------------------------------------------------------------- geometry
function offset(a, b, meters) {      // a, b: [lat, lon]; shifted to the right of a -> b
  const k = 111320, c = Math.cos(a[0] * Math.PI / 180);
  const dx = (b[1] - a[1]) * k * c, dy = (b[0] - a[0]) * k, len = Math.hypot(dx, dy) || 1;
  const nx = dy / len, ny = -dx / len;
  const f = p => [p[0] + ny * meters / k, p[1] + nx * meters / (k * c)];
  return [f(a), f(b)];
}

// ---------------------------------------------------------------- SVG charts
function svg(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img">${body}</svg>`; }
function esc(s) { return String(s).replace(/[&<>]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;"}[c])); }
function niceMax(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }

// categories: labels[], series: [{color, values[]|null, kind: "bar"|"line"}]
function categoryChart(title, labels, series, opts = {}) {
  const W = 600, H = opts.height || 250, L0 = 46, R0 = 8, T0 = 24, B0 = opts.bottom || 92;
  const vals = series.flatMap(s => s.values || []);
  const lo = Math.min(0, ...vals), hi = niceMax(Math.max(...vals, 1));
  const y = v => T0 + (H - T0 - B0) * (1 - (v - lo) / (hi - lo || 1));
  const n = labels.length, step = (W - L0 - R0) / Math.max(n, 1);
  let out = `<text class="title" x="${L0}" y="14" font-size="12">${esc(title)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    out += `<line class="grid" x1="${L0}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}"/><text x="${L0 - 4}" y="${y(v) + 3}" font-size="9" text-anchor="end">${fmt(v)}</text>`;
  }
  const active = series.filter(s => s.values);
  const bars = active.filter(s => s.kind === "bar"), bw = Math.max(1, Math.min(14, step * 0.8 / Math.max(bars.length, 1)));
  active.forEach(s => {
    if (s.kind === "bar") {
      const bi = bars.indexOf(s);
      s.values.forEach((v, i) => {
        const x = L0 + step * (i + 0.5) + (bi - (bars.length - 1) / 2) * bw - bw / 2;
        out += `<rect x="${x}" y="${Math.min(y(v), y(0))}" width="${bw}" height="${Math.abs(y(v) - y(0))}" fill="${s.color}"/>`;
      });
    } else {
      const pts = s.values.map((v, i) => `${L0 + step * (i + 0.5)},${y(v)}`).join(" ");
      out += `<polyline points="${pts}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    }
  });
  if (labels.length <= 60) labels.forEach((t, i) => {
    const x = L0 + step * (i + 0.5);
    out += `<text x="${x}" y="${H - B0 + 10}" font-size="${opts.font || 8}" text-anchor="end" transform="rotate(-60 ${x} ${H - B0 + 10})">${esc(t)}</text>`;
  });
  return svg(W, H, out);
}

function hourChart(title, series, ylabel) {      // series: [{color, values[24]|null}]
  const W = 300, H = 170, L0 = 38, R0 = 8, T0 = 22, B0 = 24;
  const vals = series.flatMap(s => s.values || []), hi = niceMax(Math.max(...vals, 1));
  const x = h => L0 + (W - L0 - R0) * h / 23, y = v => T0 + (H - T0 - B0) * (1 - v / hi);
  let out = `<text class="title" x="${L0}" y="13" font-size="11">${esc(title)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const v = hi * i / 4;
    out += `<line class="grid" x1="${L0}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}"/><text x="${L0 - 4}" y="${y(v) + 3}" font-size="9" text-anchor="end">${fmt(v)}</text>`;
  }
  for (let h = 0; h < 24; h += 3) out += `<text x="${x(h)}" y="${H - 8}" font-size="9" text-anchor="middle">${h}</text>`;
  series.forEach(s => { if (s.values) out += `<polyline points="${s.values.map((v, h) => x(h) + "," + y(v)).join(" ")}" fill="none" stroke="${s.color}" stroke-width="2"/>`; });
  if (!vals.some(v => v > 0)) out += `<text x="${W / 2}" y="${H / 2}" font-size="10" text-anchor="middle">none</text>`;
  return svg(W, H, out);
}

// ---------------------------------------------------------------- passenger profiles (as in GTFS_router)
let profilePeriod = -1;   // -1: all day, else the index of PROFILES.periods
const sumArrays = arrays => arrays[0].map((_, i) => arrays.reduce((s, a) => s + a[i], 0));

// a bar per category (share of these passengers) and a tick (share among all the PT passengers of the simulation)
function profileHtml(counts, reference, people, scope) {
  if (!PROFILES) return "";
  if (!counts || people < PROFILES.minPeople)
    return `<div class="profile"><div class="pnote">Fewer than ${PROFILES.minPeople} simulated passengers ${scope}: no profile.</div></div>`;
  let offset = 0, dims = "";
  for (const [name, labels] of PROFILES.dimensions) {
    const part = counts.slice(offset, offset + labels.length), base = reference.slice(offset, offset + labels.length);
    offset += labels.length;
    const total = sum(part), totalBase = sum(base);
    if (!total) continue;
    dims += `<div class="pdim"><div class="pname">${esc(name)}</div>` + labels.map((label, i) => {
      const share = part[i] / total, average = totalBase ? base[i] / totalBase : 0;
      return `<div class="prow"><span class="plab" title="${esc(label)}">${esc(label)}</span><span class="ptrack">` +
        `<span class="pfill" style="width:${(100 * share).toFixed(1)}%"></span>` +
        `<span class="ptick" style="left:${(100 * average).toFixed(1)}%" title="all PT passengers: ${(100 * average).toFixed(0)} %"></span></span>` +
        `<span class="pval">${(100 * share).toFixed(0)} %</span></div>`;
    }).join("") + "</div>";
  }
  return `<div class="profile"><div class="pnote">${fmt(people)} simulated passengers ${scope} (not scaled by the sample)</div>` +
    `<div class="pkey"><span><i class="kfill"></i>these passengers</span><span><i class="ktick"></i>all the PT passengers of the simulation</span></div>` +
    `<div class="pgrid">${dims}</div></div>`;
}

function lineProfile() {
  const d = DATA[current];
  if (!PROFILES || !d.profile) return "";
  const all = profilePeriod < 0;
  const counts = all ? sumArrays(d.profile.line) : d.profile.line[profilePeriod];
  const people = all ? sum(d.profile.legs) : d.profile.legs[profilePeriod];
  const reference = all ? sumArrays(PROFILES.overall) : PROFILES.overall[profilePeriod];
  const periods = [[-1, "All day"], ...PROFILES.periods.map((p, i) => [i, p])].map(([i, p]) =>
    `<label><input type="radio" name="pperiod" value="${i}"${i === profilePeriod ? " checked" : ""}> ${esc(p)}</label>`).join("");
  return `<h2>Profile of the passengers of the line</h2><div class="profile periods">${periods}</div>` +
    profileHtml(counts, reference, people, all ? "on the line (legs, all day)" : "on the line (legs, " + PROFILES.periods[profilePeriod].toLowerCase() + ")");
}

function sectionProfile(o, k) {
  const d = DATA[current];
  if (!PROFILES || !d.profile) return "";
  const order = orderOf(current, o), a = d.stops[order[k]], b = d.stops[order[k + 1]];
  const found = d.profile.sections[`${order[k]}>${order[k + 1]}`];
  return `<h2>Profile of the passengers ${esc(a ? a[0] : "")} → ${esc(b ? b[0] : "")}</h2>` +
    profileHtml(found ? found[1] : null, sumArrays(PROFILES.overall), found ? found[0] : 0, "on this connection (all day)");
}

document.addEventListener("change", e => {
  if (e.target.name !== "pperiod") return;
  profilePeriod = +e.target.value;
  byId("line-profile").innerHTML = lineProfile();
});

// ---------------------------------------------------------------- panels
function shortName(name) { return name.replace(/^(Lausanne|Renens VD|Ecublens VD|Chavannes-R\.|Prilly|Pully|Epalinges), /, ""); }

function selectionPanel() {
  const d = DATA[current];
  if (!selection) return `<h2>Selection</h2><div class="muted">Click a stop or a connection on the map to see its hourly profiles.</div>`;

  if (selection.type === "stop") {
    const code = selection.code, name = d.stops[code] ? d.stops[code][0] : code;
    let out = `<h2>${esc(name)}: boardings and alightings per hour</h2><div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">`;
    for (const o of ["fwd", "rev"]) for (const [i, measure] of [[0, "Boardings"], [1, "Alightings"]]) {
      const series = ["matsim", "tl"].map(s => ({color: s === "matsim" ? MATSIM : TL, values: hasSource(current, s, o) ? stopHours(current, s, o, code)[i] : null}));
      out += hourChart(`${d.titles[o]}: ${measure.toLowerCase()}`, series);
    }
    return out + "</div>";
  }

  const k = selection.k;
  let out = `<h2>Passengers on board per hour</h2><div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">`;
  for (const o of ["fwd", "rev"]) {
    const order = orderOf(current, o), kk = o === selection.o ? k : order.length - 2 - k;
    const a = d.stops[order[kk]], b = d.stops[order[kk + 1]];
    const series = ["matsim", "tl"].map(s => ({color: s === "matsim" ? MATSIM : TL, values: hasSource(current, s, o) ? hourlyFlow(current, s, o)[kk] : null}));
    out += hourChart(`${a ? a[0] : ""} → ${b ? b[0] : ""}`, series);
  }
  return out + "</div>" + sectionProfile(selection.o, k);
}

function lineCharts() {
  const d = DATA[current];
  let out = "";
  for (const o of ["fwd", "rev"]) {
    const order = orderOf(current, o), labels = order.map(c => shortName(d.stops[c] ? d.stops[c][0] : c));
    out += `<h2>${esc(d.titles[o])}</h2>`;
    for (const [i, title] of [[0, "Boardings per day"], [1, "Alightings per day"]]) {
      out += categoryChart(title, labels, ["matsim", "tl"].map(s => ({
        color: s === "matsim" ? MATSIM : TL, kind: "bar",
        values: hasSource(current, s, o) ? order.map(c => sum(stopHours(current, s, o, c)[i])) : null})));
    }
    out += categoryChart("Passengers on board between the stops", labels.slice(0, -1).map((a, k) => `${a} → ${labels[k + 1]}`),
      ["matsim", "tl"].map(s => ({color: s === "matsim" ? MATSIM : TL, kind: "line", values: hasSource(current, s, o) ? dailyFlow(current, s, o) : null})));
  }
  return out;
}

function renderPanels() {
  document.getElementById("graphs").innerHTML = `<div id="selection">${selectionPanel()}</div><hr class="divider"><h1 style="margin-top:12px">Whole line</h1><div id="line-profile">${lineProfile()}</div>${lineCharts()}`;
}

function summary() {
  const d = DATA[current], t = d.totals, r = ratioOf(t.matsim, t.tl);
  document.getElementById("summary").innerHTML =
    `<table class="t"><tr><th>${esc(current)} (${esc(d.submode)}, ${esc(d.line_id)})</th><th>MATSim</th><th>__REF__</th><th>ratio</th></tr>` +
    `<tr><td>boardings per day</td><td>${fmt(t.matsim)}</td><td>${fmt(t.tl)}</td><td>${r == null ? "–" : r.toFixed(2)}</td></tr></table>`;
}

// ---------------------------------------------------------------- map
function draw(fit = false) {
  layer.clearLayers();
  const d = DATA[current], mode = document.querySelector("input[name=color]:checked").value;
  const measure = document.querySelector("input[name=measure]:checked").value, numbers = byId("numbers").checked;
  const pts = d.order.filter(c => d.stops[c]).map(c => [d.stops[c][1], d.stops[c][2]]);
  if (!pts.length) return;

  // a value per source for each connection and direction, and for each stop (both directions)
  const flows = {};
  let maxFlow = 1;
  for (const o of ["fwd", "rev"]) for (const s of ["matsim", "tl"]) {
    flows[o + s] = hasSource(current, s, o) ? dailyFlow(current, s, o) : null;
    if (flows[o + s]) for (const v of flows[o + s]) maxFlow = Math.max(maxFlow, mode === "ratio" || mode === s ? v : 0);
  }
  const stopValues = {};
  let maxStop = 1;
  for (const code of d.order) {
    const v = {};
    for (const s of ["matsim", "tl"]) {
      v[s] = {boardings: 0, alightings: 0};
      for (const o of ["fwd", "rev"]) if (hasSource(current, s, o)) {
        const [b, a] = stopHours(current, s, o, code);
        v[s].boardings += sum(b); v[s].alightings += sum(a);
      }
      v[s].both = v[s].boardings + v[s].alightings;
    }
    stopValues[code] = v;
    maxStop = Math.max(maxStop, mode === "ratio" ? Math.max(v.matsim[measure], v.tl[measure]) : v[mode][measure]);
  }
  legend(mode, maxFlow, maxStop);

  for (const o of ["fwd", "rev"]) {
    const order = orderOf(current, o), side = o === "fwd" ? 1 : -1;
    for (let k = 0; k < order.length - 1; k++) {
      const a = d.stops[order[k]], b = d.stops[order[k + 1]];
      if (!a || !b) continue;
      const pa = [a[1], a[2]], pb = [b[1], b[2]];
      const m = flows[o + "matsim"] ? flows[o + "matsim"][k] : null, t = flows[o + "tl"] ? flows[o + "tl"][k] : null, r = ratioOf(m, t);
      const value = mode === "ratio" ? Math.max(m || 0, t || 0) : (mode === "matsim" ? m : t);
      if (value == null) continue;
      const tip = `${d.titles[o]}: ${a[0]} → ${b[0]}<br>per day: MATSim ${fmt(m)}, __REF__ ${fmt(t)}${r == null ? "" : ", ratio " + r.toFixed(2)}`;
      const click = () => { selection = {type: "segment", k, o}; byId("selection").innerHTML = selectionPanel(); };
      const shape = offset(pa, pb, px(side * (SIDE + LANE / 2), pa[0]));
      L.polyline(shape, {color: colorOf(mode, m, t, maxFlow), weight: 2 + 8 * Math.max(value, 0) / maxFlow, opacity: 0.9})
        .bindTooltip(tip, {sticky: true}).on("click", click).addTo(layer);
    }
  }

  d.order.forEach(code => {
    const p = d.stops[code];
    if (!p) return;
    const v = stopValues[code], m = v.matsim[measure], t = v.tl[measure];
    const size = mode === "ratio" ? Math.max(m, t) : v[mode][measure];
    const marker = L.circleMarker([p[1], p[2]], {
      radius: 4 + 14 * Math.sqrt(Math.max(size, 0) / maxStop), color: "#0b0b0b", weight: 1.2,
      fillColor: colorOf(mode, m, t, maxStop), fillOpacity: 0.95,
    }).bindTooltip(
      `<div class="tip"><b>${esc(p[0])}</b><br>boardings: MATSim ${fmt(v.matsim.boardings)}, __REF__ ${fmt(v.tl.boardings)}` +
      `<br>alightings: MATSim ${fmt(v.matsim.alightings)}, __REF__ ${fmt(v.tl.alightings)}</div>`, {direction: "top"})
      .on("click", () => { selection = {type: "stop", code}; byId("selection").innerHTML = selectionPanel(); });
    marker.addTo(layer);
    if (numbers) L.marker([p[1], p[2]], {interactive: false, icon: L.divIcon({
      className: "", iconSize: [0, 0],
      html: `<div style="position:absolute;left:10px;top:-7px;white-space:nowrap;font:11px system-ui;color:#0b0b0b;text-shadow:0 0 3px #fff,0 0 3px #fff">${fmt(m)} / ${fmt(t)}</div>`})}).addTo(layer);
  });

  if (fit) map.fitBounds(L.latLngBounds(pts).pad(0.2));
}

function byId(id) { return document.getElementById(id); }

function select(line) {
  current = line; selection = null;
  summary(); renderPanels(); draw(true);
}

// ---------------------------------------------------------------- menu
function fillSelect() {
  const text = byId("filter").value.trim().toLowerCase(), box = byId("line");
  const names = Object.keys(DATA).filter(n => !text || n.toLowerCase().includes(text) || DATA[n].submode.toLowerCase().includes(text));
  const bySub = {};
  names.forEach(n => (bySub[DATA[n].submode] = bySub[DATA[n].submode] || []).push(n));
  const subs = Object.keys(bySub).sort((a, b) => (SUBMODE_ORDER.indexOf(a) + 1 || 99) - (SUBMODE_ORDER.indexOf(b) + 1 || 99));
  box.innerHTML = subs.map(sub => `<optgroup label="${esc(sub)}">` + bySub[sub].sort((a, b) => a.localeCompare(b, "en", {numeric: true})).map(n => {
    const r = ratioOf(DATA[n].totals.matsim, DATA[n].totals.tl);
    return `<option value="${esc(n)}"${n === current ? " selected" : ""}>${esc(n)}${r == null ? "" : "  (ratio MATSim / __REF__: " + r.toFixed(2) + ")"}</option>`;
  }).join("") + "</optgroup>").join("");
}

byId("filter").addEventListener("input", () => { fillSelect(); if (byId("line").value) select(byId("line").value); });
byId("line").addEventListener("change", e => select(e.target.value));
byId("numbers").addEventListener("change", () => draw());
map.on("zoomend", () => { if (current) draw(); });
document.querySelectorAll("input[name=measure], input[name=color]").forEach(el => el.addEventListener("change", () => draw()));

fillSelect();
const first = Object.keys(DATA).find(n => DATA[n].submode === "metro") || Object.keys(DATA)[0];
if (first) { byId("line").value = first; select(first); }
</script></body></html>
"""


def main(argv = None):
    parser = argparse.ArgumentParser(description = "Map comparing the MATSim and TL passengers of the lines")
    parser.add_argument("--simulation-path", required = True, help = "MATSim output folder (pt_passenger_counts.csv.gz and transit schedule)")
    parser.add_argument("--tl-processed", required = True, help = "Folder with tl2024_agg_workday_<date>.csv")
    parser.add_argument("--gtfs", required = True, help = "GTFS of the simulated schedule: output folder of data.gtfs.cleaned (or a zip)")
    parser.add_argument("--input-downsampling", type = float, required = True)
    parser.add_argument("--lines", nargs = "*", default = [], help = "MATSim line names (default: all the lines of the TL data)")
    parser.add_argument("--night-hours", type = int, default = 5)
    parser.add_argument("--output-dir", default = None)
    parser.add_argument("--elevations", default = None, help = "Elevation cache of the pipeline (<data_path>/osm/elevations.csv)")
    args = parser.parse_args(argv)

    run(args.simulation_path, read_tl_table(args.tl_processed), args.gtfs, args.input_downsampling, args.lines, args.night_hours,
        args.output_dir, elevation_path = args.elevations)


if __name__ == "__main__":
    main()
