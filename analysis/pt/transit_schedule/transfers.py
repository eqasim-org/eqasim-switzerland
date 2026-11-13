"""Map of the GTFS transfer times, written by analysis.pt.transit_schedule.visualize (transfer_times_map.html).

Reads the transfer rules of the cleaned GTFS (transfers.txt, after data.gtfs.cleaned) touching the given perimeters
(the Geneva and Lausanne ones in the pipeline). The map is drawn by station, not by platform: the stops of a station
(the same DiDok number, else the same parent_station; see station_groups) are one point, coloured by the transfer times
within the station (between its platforms, or within a stop). The rules between two stations are one line per direction
(two when both given and computed rules join them), coloured by their most common minimum transfer time, the tooltip
giving all the times. Where data.gtfs.cleaned wrote its diagnostics, the rules it computed (between feeds, by hand or from
a nearby stop) are dashed. Nothing is flagged: the map shows the times only. transfer_times.csv still lists every rule."""
import json
import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from analysis.pt.transit_schedule import leaflet_page

# The origin of a rule: how data.gtfs.cleaned set its time (0: given by a source feed)
RULE_ORIGINS = {"feed": 0, "bridge": 1, "manual": 2, "nearby": 3}

# Normalized modes (basic GTFS route types and the extended ones)
MODES = ["Rail", "Metro", "Tram", "Bus", "Boat", "Funicular", "Cable car", "Other"]

# Classes of transfer time (upper bounds in seconds, the last one open) and their colours, light to dark: one hue,
# from the ordinal steps 250 to 700 of the blue reference ramp (the lightest still clears 2:1 on the map)
TIME_BOUNDS = [60, 120, 180, 300, 600]
TIME_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

# The UIC code of an SNCF stop (StopArea:OCE85011205, StopPoint:OCETrain TER-87745380): 8 digits, the last one a check digit
SNCF_CODE = re.compile(r"OCE\D*?(\d{8})$")


def load_transfers(gtfs_path, diagnostics_path, perimeter_paths):
    """The stops and the rules touching the perimeters ({name: path} of polygon files), with the distance between their
    stops, whether the origin of the rules is known, and the perimeters as GeoJSON (WGS84)."""
    perimeter_geojson, polygon = read_perimeters(perimeter_paths)
    stops = read_stops(gtfs_path)
    transfers, has_origins = read_transfers(gtfs_path, diagnostics_path)

    position = stops.set_index("stop_id")[["stop_lat", "stop_lon"]]
    transfers = transfers[transfers["from_stop_id"].isin(position.index) & transfers["to_stop_id"].isin(position.index)]
    inside = stops_inside(stops, polygon)
    transfers = transfers[transfers["from_stop_id"].isin(inside) | transfers["to_stop_id"].isin(inside)]

    transfers = transfers.join(position, on = "from_stop_id").join(position, on = "to_stop_id", rsuffix = "_to")
    transfers["distance_m"] = haversine_m(transfers["stop_lat"], transfers["stop_lon"],
                                          transfers["stop_lat_to"], transfers["stop_lon_to"]).round()
    transfers = transfers[["from_stop_id", "to_stop_id", "min_transfer_time", "rule", "distance_m"]].reset_index(drop = True)
    shown = set(transfers["from_stop_id"]) | set(transfers["to_stop_id"])
    shown |= set(stops.loc[stops["stop_id"].isin(shown), "parent_station"].dropna())   # their stations: their position and name
    return stops[stops["stop_id"].isin(shown)].reset_index(drop = True), transfers, has_origins, perimeter_geojson


def write_map(gtfs_path, transfers, output_path):
    """Writes the map of the transfers (the result of load_transfers) to output_path."""
    stops, rules, has_origins, perimeter_geojson = transfers
    stop_modes = read_stop_modes(gtfs_path, set(stops["stop_id"]))
    data = build_map_data(stops, rules, stop_modes, has_origins, perimeter_geojson)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok = True)
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(leaflet_page.page("GTFS transfer times", MENU, data, SCRIPT))
    print(f"Transfer times: {len(data['stations'])} station(s) for {data['stopCount']} stop(s); "
          f"{sum(r[5] for r in data['rules'])} possible transfer(s) between stations drawn as {len(data['rules'])} line(s), "
          f"{sum(n for st in data['stations'] for _, n in st[5])} within a station - saved to {output_path}")


def build_map_data(stops, transfers, stop_modes, has_origins, perimeter_geojson):
    """The data of the page, by station (station_groups): one point per station, not per platform.
    - stations: [name, lat, lon, key, modes (bits of MODES), times, stops]: times are the rules within the station
      (between two of its stops, or within a stop) as [[time (s), number of rules], ...] by time; stops are its stops as
      [[stop_id, platform], ...]
    - rules: the lines between different stations [from, to, time (s), origin (see RULE_ORIGINS), distance (m), number of
      rules, times]: one line per direction and kind of origin (given or computed), whose time is the most common one of
      its rules (the shortest of the ties), times as for the stations, and origin the most frequent one."""
    group = station_groups(stops)
    stops = stops.assign(station = stops["stop_id"].map(group))
    keys = list(dict.fromkeys(stops["station"]))
    index = {k: n for n, k in enumerate(keys)}
    mode_bits = {m: 1 << k for k, m in enumerate(MODES)}

    rules = transfers.assign(a = transfers["from_stop_id"].map(group), b = transfers["to_stop_id"].map(group),
                             origin = transfers["rule"].map(RULE_ORIGINS).fillna(0).astype(int))
    rules["computed"] = rules["origin"] > 0
    inside = rules[rules["a"] == rules["b"]]
    own_times = {k: _times(g["min_transfer_time"]) for k, g in inside.groupby("a")}

    station_rows, position = [], {}
    for key, members in stops.groupby("station", sort = False):
        head = _station_head(members)
        position[key] = (head.stop_lat, head.stop_lon) if head is not None else (members["stop_lat"].mean(), members["stop_lon"].mean())
        name = head.stop_name if head is not None else members["stop_name"].mode().iloc[0]
        modes = set().union(*(stop_modes.get(s, []) for s in members["stop_id"]))
        listed = members[members["location_type"] != "1"] if (members["location_type"] != "1").any() else members
        station_rows.append([name, round(position[key][0], 6), round(position[key][1], 6), key, sum(mode_bits[m] for m in modes),
                             own_times.get(key, []), [[s, p if isinstance(p, str) else ""] for s, p in zip(listed["stop_id"], listed["platform_code"])]])

    line_rows = []
    for (a, b, _), lines in rules[rules["a"] != rules["b"]].groupby(["a", "b", "computed"], sort = False):
        times = _times(lines["min_transfer_time"])
        distance = haversine_m(position[a][0], position[a][1], position[b][0], position[b][1]).round()
        line_rows.append([index[a], index[b], _typical(times), int(lines["origin"].mode().iloc[0]), int(distance), len(lines), times])

    return {"stations": station_rows, "rules": line_rows, "stopCount": len(stops[stops["location_type"] != "1"]), "modes": MODES,
            "hasOrigins": has_origins, "perimeter": perimeter_geojson, "bounds": TIME_BOUNDS, "colors": TIME_COLORS}


def station_groups(stops):
    """{stop_id: station key}: the stops of a station are drawn as one point. The station of a stop is its DiDok number
    (the didok column, else the UIC code of an SNCF stop area: StopArea:OCE85011205 -> 8501120), else that of its
    parent_station, else its parent_station, else the stop itself. So the platforms of Lausanne (ch:1:sloid:1120:0:...)
    and the SNCF stop area of Lausanne are one station, while Lausanne, gare (the buses and the metro, another DiDok
    number) is another one."""
    own = {s: _station_code(s, d) for s, d in zip(stops["stop_id"], stops["didok"])}
    result = {}
    for stop_id, parent in zip(stops["stop_id"], stops["parent_station"]):
        parent = parent if isinstance(parent, str) and parent else None
        result[stop_id] = own[stop_id] or (own.get(parent) or _station_code(parent, None) or parent if parent else None) or stop_id
    return result


def _station_code(stop_id, didok):
    """The DiDok number of a stop ("uic:8501120"), from its didok column or its SNCF code; None without one."""
    didok = str(didok).split(".")[0] if isinstance(didok, str) else ""
    if didok.isdigit():
        return "uic:" + didok
    match = SNCF_CODE.search(stop_id or "")
    return "uic:" + match.group(1)[:7] if match else None


def _station_head(members):
    """The stop that stands for its station (its position and name): a station (location_type 1) with a DiDok number,
    else any station; None without one."""
    heads = members[members["location_type"] == "1"]
    if not len(heads):
        return None
    with_didok = heads[heads["didok"].fillna("") != ""]
    return next((with_didok if len(with_didok) else heads).itertuples())


def _times(values):
    """The times of rules as [[time (s), number of rules], ...], by time."""
    return [[int(t), int(n)] for t, n in values.value_counts().sort_index().items()]


def _typical(times):
    """The most common time of [[time, number], ...] (the shortest of the ties)."""
    return max(times, key = lambda tn: (tn[1], -tn[0]))[0]


def read_perimeters(paths):
    """The perimeters {name: path} as one GeoJSON (simplified, WGS84) and their union (WGS84)."""
    parts = []
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} perimeter {path} does not exist")
        shape = gpd.read_file(path)
        shape = shape.to_crs(2056) if shape.crs is not None else shape.set_crs(4326).to_crs(2056)
        parts.append((name, shape.union_all().simplify(20)))
    shapes = gpd.GeoDataFrame({"name": [name for name, _ in parts]}, geometry = [geometry for _, geometry in parts], crs = 2056).to_crs(4326)
    return json.loads(shapes.to_json()), shapes.union_all()


def read_stops(gtfs_path):
    stops = pd.read_csv(os.path.join(gtfs_path, "stops.txt"), dtype = str, keep_default_na = False, na_values = [""],
                        usecols = lambda c: c in ("stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station", "location_type",
                                                  "platform_code", "didok"))
    for column in ("parent_station", "location_type", "platform_code", "didok"):
        if column not in stops:
            stops[column] = np.nan
    stops["location_type"] = stops["location_type"].fillna("0")
    stops = stops.dropna(subset = ["stop_lat", "stop_lon"]).drop_duplicates("stop_id").copy()
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    stops["stop_name"] = stops["stop_name"].fillna(stops["stop_id"])
    return stops.reset_index(drop = True)


def read_transfers(gtfs_path, diagnostics_path):
    """The rules (from_stop_id, to_stop_id, min_transfer_time, rule) and whether their origin is known. The rules removed
    by data.gtfs.cleaned are left out. Several rules for a pair of stops (e.g. per trip) keep the shortest time."""
    rules_path = os.path.join(diagnostics_path, "transfers.csv") if diagnostics_path else None
    if rules_path and os.path.exists(rules_path):
        transfers = pd.read_csv(rules_path, dtype = {"from_stop_id": str, "to_stop_id": str})
        transfers = transfers[transfers["kept"].astype(str).str.lower() == "true"]
        has_origins = True
    else:
        path = os.path.join(gtfs_path, "transfers.txt")
        columns = ["from_stop_id", "to_stop_id", "min_transfer_time"]
        transfers = pd.read_csv(path, dtype = str, usecols = lambda c: c in columns) if os.path.exists(path) else pd.DataFrame(columns = columns)
        transfers["rule"] = "feed"
        has_origins = False
        print("WARNING: no diagnostics of data.gtfs.cleaned (re-run it): the origin of the transfer rules is unknown")

    transfers = transfers.dropna(subset = ["from_stop_id", "to_stop_id"]).copy()
    transfers["min_transfer_time"] = pd.to_numeric(transfers["min_transfer_time"], errors = "coerce").fillna(0).astype(int)
    transfers = transfers.sort_values("min_transfer_time").drop_duplicates(["from_stop_id", "to_stop_id"])
    return transfers[["from_stop_id", "to_stop_id", "min_transfer_time", "rule"]].reset_index(drop = True), has_origins


def read_stop_modes(gtfs_path, stop_ids):
    """{stop_id: sorted modes} of the given stops, over all the services of the feed."""
    pairs = []
    reader = pd.read_csv(os.path.join(gtfs_path, "stop_times.txt"), dtype = str, usecols = ["trip_id", "stop_id"], chunksize = 2_000_000)
    for chunk in reader:
        pairs.append(chunk[chunk["stop_id"].isin(stop_ids)].drop_duplicates())
    stop_trips = pd.concat(pairs, ignore_index = True).drop_duplicates() if pairs else pd.DataFrame(columns = ["trip_id", "stop_id"])

    trips = pd.read_csv(os.path.join(gtfs_path, "trips.txt"), dtype = str, usecols = ["trip_id", "route_id"])
    routes = pd.read_csv(os.path.join(gtfs_path, "routes.txt"), dtype = str, usecols = ["route_id", "route_type"])
    routes["mode"] = routes["route_type"].fillna("3").map(mode_of)
    modes = stop_trips.merge(trips, on = "trip_id").merge(routes[["route_id", "mode"]], on = "route_id")
    return modes.groupby("stop_id")["mode"].agg(lambda m: sorted(set(m))).to_dict()


def mode_of(route_type):
    t = int(route_type)
    basic = {0: "Tram", 1: "Metro", 2: "Rail", 3: "Bus", 4: "Boat", 5: "Cable car", 6: "Cable car", 7: "Funicular", 11: "Bus", 12: "Rail"}
    if t in basic:
        return basic[t]
    for low, high, mode in ((100, 199, "Rail"), (200, 299, "Bus"), (400, 499, "Metro"), (700, 899, "Bus"), (900, 999, "Tram"),
                            (1000, 1099, "Boat"), (1300, 1399, "Cable car"), (1400, 1499, "Funicular")):
        if low <= t <= high:
            return mode
    return "Other"


def stops_inside(stops, polygon):
    points = gpd.GeoSeries(gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs = 4326)
    return set(stops["stop_id"][points.within(polygon).values])


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between paired arrays of coordinates."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype = float)) for a in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371000 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

MENU = r"""
  <div id="menu">
    <div id="top">
      <h1>GTFS transfer times</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row"><input id="search" type="text" placeholder="Search a station or stop id (e.g. Cornavin, Flon)"></div>
      <div class="list" id="results"></div>
      <div class="row legend" id="classes"></div>
      <div class="row legend" id="origins"></div>
      <div class="row"><label><input type="checkbox" id="showStops" checked> stations (zoomed in), coloured by their transfer time within the station</label></div>
    </div>
    <div id="panel"></div>
  </div>
  <style>
    .line { display:inline-block; width:22px; height:0; border-top:3px solid #52514e; vertical-align:middle; margin-right:5px; }
    .line.dashed { border-top-style:dashed; }
    .stopkey { display:inline-block; width:10px; height:10px; border-radius:50%; border:1px solid #0b0b0b; vertical-align:middle; margin-right:5px; }
    #top label { white-space:normal; }
  </style>
"""

SCRIPT = r"""
const S = DATA.stations, R = DATA.rules, B = DATA.bounds, COLORS = DATA.colors;
const NO_TIME = "#d4d3cf";
const ORIGINS = ["given by the GTFS", "computed: between the stations of different feeds", "set by hand (MANUAL_TRANSFERS)",
                 "computed: time of a nearby stop + walking"];
const minutes = s => String(Math.round(s / 6) / 10);   // 420 s -> "7", 441 s -> "7.4"
const classOf = t => { const k = B.findIndex(b => t < b); return k < 0 ? B.length : k; };
const classLabel = k => k === 0 ? `under ${minutes(B[0])} min` : k === B.length ? `${minutes(B[k - 1])} min or more` : `${minutes(B[k - 1])}–${minutes(B[k])} min`;
const computed = rule => rule[3] > 0 ? 1 : 0;
const shownClasses = new Set(COLORS.map((c, k) => k)), shownOrigins = new Set([0, 1]);
// times: [[time, number of possible transfers], ...] by time
const count = times => sum(times.map(t => t[1]));
const typical = times => times.reduce((best, t) => t[1] > best[1] ? t : best, times[0])[0];
// the interval of the times ("7–8.1 min"), with the most frequent one (the colour) when they differ
const range = times => { const low = minutes(times[0][0]), high = minutes(times[times.length - 1][0]);
  return low === high ? `${low} min` : `${low}–${high} min`; };
const rangeTypical = times => range(times) + (times.length > 1 ? ` <span class="muted">(most often ${minutes(typical(times))} min)</span>` : "");
const transfers = n => `${fmt(n)} possible transfer${n === 1 ? "" : "s"}`;

// the lines of each station, and whether the reverse of a line exists (both directions are then drawn side by side)
const OUT = S.map(() => []), IN = S.map(() => []), pairs = new Set(R.map(r => r[0] + "|" + r[1]));
R.forEach((rule, n) => { OUT[rule[0]].push(n); IN[rule[1]].push(n); rule.push(pairs.has(rule[1] + "|" + rule[0])); });
const ruleCount = sum(R.map(r => r[5])), withinCount = sum(S.map(s => count(s[5])));

byId("subtitle").textContent = `Minimum time to change between two stops, in the Geneva and Lausanne perimeters: ${fmt(S.length)} ` +
  `station(s) for ${fmt(DATA.stopCount)} stop(s), ${transfers(ruleCount)} between stations (one line per direction) and ` +
  `${transfers(withinCount)} within a station (between its platforms).`;

// legend: the classes of the most frequent time of the lines and where the times come from, with their numbers of possible transfers
const classCounts = COLORS.map(() => 0);
R.forEach(rule => classCounts[classOf(rule[2])] += rule[5]);
byId("classes").innerHTML = "<b>Minimum transfer time</b> <span class=\"muted\">(colour: the most frequent time of a line or station; number of possible transfers)</span>" + COLORS.map((color, k) =>
  `<div><label><input type="checkbox" class="cls" data-key="${k}" checked> <span class="swatch" style="background:${color};height:5px"></span>` +
  `${classLabel(k)} <span class="muted">(${fmt(classCounts[k])})</span></label></div>`).join("") +
  `<div class="muted"><span class="stopkey" style="background:${NO_TIME}"></span>station without a transfer time within it</div>`;
const originCounts = [0, 1].map(o => sum(R.filter(rule => computed(rule) === o).map(rule => rule[5])));
byId("origins").innerHTML = DATA.hasOrigins ?
  `<div><label><input type="checkbox" class="origin" data-key="0" checked> <span class="line"></span>given by the GTFS <span class="muted">(${fmt(originCounts[0])})</span></label></div>` +
  `<div><label><input type="checkbox" class="origin" data-key="1" checked> <span class="line dashed"></span>computed or set by hand <span class="muted">(${fmt(originCounts[1])})</span></label></div>` :
  `<div class="muted">Where the times come from is written by data.gtfs.cleaned: re-run it to tell the computed ones apart.</div>`;

const layer = L.layerGroup().addTo(map), highlight = L.layerGroup().addTo(map);
const modesOf = i => DATA.modes.filter((m, k) => S[i][4] & (1 << k)).join(", ") || "not served";
const stopsOf = i => S[i][6].length === 1 ? esc(S[i][6][0][0]) : `${S[i][6].length} stops`;
const stationLabel = i => `${esc(S[i][0])} <span class="muted">${stopsOf(i)}</span>`;
const stationColor = i => S[i][5].length ? COLORS[classOf(typical(S[i][5]))] : NO_TIME;

function ruleTip(rule) {
  const [a, b, time, origin, dist, n, times] = rule;
  return `<b>${rangeTypical(times)}</b>` + (DATA.hasOrigins ? ` <span class="muted">${esc(ORIGINS[origin])}</span>` : "") +
    `<br>${stationLabel(a)} (${esc(modesOf(a))})<br>→ ${stationLabel(b)} (${esc(modesOf(b))})<br>${fmt(dist)} m apart` +
    (n > 1 ? `<br><span class="muted">${transfers(n)} between the stops of these stations</span>` : "");
}

function stationTip(i) {
  const times = S[i][5];
  return `${stationLabel(i)}<br>${esc(modesOf(i))}<br>` +
    (times.length ? `transfer within the station: <b>${rangeTypical(times)}</b>` +
      ` <span class="muted">${transfers(count(times))}</span>` : "no transfer time within the station") +
    `<br>${OUT[i].length} line(s) from, ${IN[i].length} to other stations`;
}

const ruleShown = rule => shownClasses.has(classOf(rule[2])) && shownOrigins.has(computed(rule));

function draw() {
  layer.clearLayers();
  const view = map.getBounds().pad(0.2), zoom = map.getZoom();
  R.forEach(rule => {
    if (!ruleShown(rule)) return;
    const a = S[rule[0]], b = S[rule[1]];
    if (!view.contains([a[1], a[2]]) && !view.contains([b[1], b[2]])) return;
    let path = [[a[1], a[2]], [b[1], b[2]]];
    if (rule[7]) path = offsetPath(path, px(zoom >= 15 ? 3 : 2, a[1]));   // both directions: each on its right
    L.polyline(path, {color: COLORS[classOf(rule[2])], weight: zoom >= 15 ? 3 : 2, opacity: 0.9, dashArray: computed(rule) ? "5 4" : null})
      .bindTooltip(ruleTip(rule), {sticky: true}).addTo(layer);
  });
  if (byId("showStops").checked && zoom >= 14) S.forEach(([, la, lo], i) => {
    if (!view.contains([la, lo])) return;
    L.circleMarker([la, lo], {radius: zoom >= 16 ? 6 : 4.5, color: "#0b0b0b", weight: 1, fillColor: stationColor(i), fillOpacity: 1})
      .bindTooltip(stationTip(i)).on("click", () => showStation(i, false)).addTo(layer);
  });
}

// the transfers of a station: within it, and to and from the other stations
function showStation(i, center) {
  const [name, la, lo, , , times, members] = S[i];
  if (center) map.setView([la, lo], Math.max(map.getZoom(), 16));
  highlight.clearLayers();
  L.circleMarker([la, lo], {radius: 11, color: "#0b0b0b", weight: 3, fill: false, interactive: false}).addTo(highlight);
  const rows = (title, ids, other) => ids.length ? `<h2>${title} (${ids.length})</h2><table class="t"><tr><th></th><th>station</th><th class="n">min</th><th class="n">transfers</th><th class="n">m</th></tr>` +
    ids.map(n => R[n]).sort((x, y) => x[2] - y[2]).map(rule => `<tr><td><span class="swatch" style="background:${COLORS[classOf(rule[2])]}"></span></td>` +
      `<td><button class="link" data-stop="${rule[other]}">${esc(S[rule[other]][0])}</button>` + (computed(rule) ? ` <span class="muted">computed</span>` : "") +
      `</td><td class="n">${range(rule[6]).replace(" min", "")}</td><td class="n">${fmt(rule[5])}</td><td class="n">${fmt(rule[4])}</td></tr>`).join("") + "</table>" : "";
  const platforms = members.map(([id, platform]) => platform ? `${esc(platform)} <span class="muted">${esc(id)}</span>` : esc(id)).join("<br>");
  byId("panel").innerHTML = `<h2>${esc(name)}</h2><div class="muted">${esc(modesOf(i))}</div>` +
    `<div class="row">` + (times.length ? `Transfer within the station: <b>${rangeTypical(times)}</b>, ` +
      `${transfers(count(times))} between its platforms` : `<span class="muted">No transfer time within the station.</span>`) + "</div>" +
    `<details class="row"><summary>${members.length} stop(s) (platform, stop id)</summary>${platforms}</details>` +
    rows("To the other stations", OUT[i], 1) + rows("From the other stations", IN[i], 0);
  byId("panel").querySelectorAll("button[data-stop]").forEach(el => el.addEventListener("click", () => showStation(+el.dataset.stop, true)));
}

function search() {
  const text = byId("search").value.trim().toLowerCase();
  if (!text) { byId("results").innerHTML = ""; return; }
  const found = S.map((s, i) => [s, i]).filter(([s]) => s[0].toLowerCase().includes(text) || s[6].some(([id]) => id.toLowerCase().startsWith(text))).slice(0, 100);
  byId("results").innerHTML = found.map(([s, i]) => `<div data-stop="${i}"><span class="dot" style="background:${stationColor(i)};border:1px solid #0b0b0b"></span>` +
    `${stationLabel(i)}</div>`).join("") || `<div class="muted">no station</div>`;
  byId("results").querySelectorAll("div[data-stop]").forEach(el => el.addEventListener("click", () => showStation(+el.dataset.stop, true)));
}

document.querySelectorAll("input.cls").forEach(el => el.addEventListener("change", () => { el.checked ? shownClasses.add(+el.dataset.key) : shownClasses.delete(+el.dataset.key); draw(); }));
document.querySelectorAll("input.origin").forEach(el => el.addEventListener("change", () => { el.checked ? shownOrigins.add(+el.dataset.key) : shownOrigins.delete(+el.dataset.key); draw(); }));
byId("showStops").addEventListener("change", draw);
byId("search").addEventListener("input", search);
map.on("moveend", draw);
byId("panel").innerHTML = `<div class="muted">Hover a line for its transfer time, click a station (zoomed in) or search it to list its transfers.</div>`;
if (S.length) map.fitBounds(L.latLngBounds(S.map(s => [s[1], s[2]])).pad(0.02));
draw();
"""
