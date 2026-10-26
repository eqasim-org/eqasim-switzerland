"""Map of the asymmetries of the use of the lines, in the counts of the operator (TL / TPG) and in MATSim, written next to
the line comparison map (line_comparison.run) as <prefix>_line_asymmetry_map.html, with <prefix>_line_asymmetry_segments.csv
and <prefix>_line_asymmetry_lines.csv:
- between the two directions of a connection: more passengers one way than the other (in Lausanne, people ride uphill and
  walk downhill). With the elevations of the stops (osm/elevations.csv of the pipeline, the cache of the elevations of the
  network nodes), the connections climbing at least MIN_GRADE tell how much more people ride uphill than downhill;
- between the morning and the evening: a direction busier in the morning peak and the other one in the evening peak
  (commuters going to work and back home);
- at the stops: more boardings than alightings (or the opposite), over the day or in a peak.

The index of a connection is (A - B) / (A + B), A and B the passengers on board in its two directions (A: the direction of
the longest route of the line); -1 to 1, 0 when both directions carry the same number of passengers. The flows of the peaks
are the sums of the hourly flows (line_comparison._hourly_flow), the daily ones the cumulative daily flows."""
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PERIODS = {"day": None, "am": list(range(6, 9)), "pm": list(range(16, 19))}
PERIOD_NAMES = {"day": "All day", "am": "Morning peak (6-9 h)", "pm": "Evening peak (16-19 h)"}
MIN_GRADE = 0.02                  # connections flatter than 2 % are neither uphill nor downhill
MAX_ELEVATION_DISTANCE = 100.0    # m between a stop and the closest point of the elevation cache
SOURCES = ("ref", "matsim")


def stop_elevations(elevation_path, xy):
    """{stop code: elevation (m)} from the elevation cache (x, y, z in LV95) of the pipeline, for the stops with a point of
    the cache within MAX_ELEVATION_DISTANCE; {} without the cache."""
    if not elevation_path or not os.path.exists(elevation_path) or not xy:
        print(f"No elevation cache ({elevation_path}): no uphill / downhill analysis")
        return {}

    codes = list(xy)
    points = np.array([xy[c] for c in codes])
    (x0, y0), (x1, y1) = points.min(axis = 0) - 1000, points.max(axis = 0) + 1000
    cache = pd.read_csv(elevation_path, usecols = ["x", "y", "z"])
    cache = cache[cache["x"].between(x0, x1) & cache["y"].between(y0, y1) & (cache["z"] > 0)]  # 0: outside Switzerland
    if cache.empty:
        return {}

    distances, indices = cKDTree(cache[["x", "y"]].to_numpy()).query(points)
    z = cache["z"].to_numpy()
    return {code: round(float(z[i]), 1) for code, d, i in zip(codes, distances, indices) if d <= MAX_ELEVATION_DISTANCE}


def _clean(values):
    return [int(v) if float(v).is_integer() else round(float(v), 1) for v in values]


def source_data(flows, hourly, boardings, stops):
    """The data of a source (MATSim or the operator) for the page: flows {orientation: daily flows per connection},
    hourly {orientation: hourly flows (connections x 24)}, boardings {orientation: boardings per hour}, stops: {code:
    [boardings per hour, alightings per hour]} of both directions. None without any direction."""
    if not flows:
        return None
    return {
        "daily":     {o: _clean(np.clip(v, 0, None)) for o, v in flows.items()},
        "hourly":    {o: [_clean(row) for row in v] for o, v in hourly.items()},
        "boardings": {o: _clean(v) for o, v in boardings.items()},
        "stops":     stops,
    }


def period_flows(source, orientation, period):
    """Passengers on board of each connection in the period, in the order of the connections of the fwd direction."""
    if source is None or orientation not in source["daily"]:
        return None
    hours = PERIODS[period]
    values = np.array(source["daily"][orientation], dtype = float) if hours is None \
        else np.array(source["hourly"][orientation], dtype = float)[:, hours].sum(axis = 1)
    return values if orientation == "fwd" else values[::-1]


def _index(a, b):
    return (a - b) / (a + b) if a + b > 0 else np.nan


def segment_grades(order, xy, elevations):
    """Gradient of each connection of the fwd direction (None without the elevation of both stops)."""
    grades = []
    for a, b in zip(order[:-1], order[1:]):
        if a in elevations and b in elevations and a in xy and b in xy:
            distance = math.hypot(xy[b][0] - xy[a][0], xy[b][1] - xy[a][1])
            grades.append(round((elevations[b] - elevations[a]) / distance, 4) if distance > 0 else None)
        else:
            grades.append(None)
    return grades


def line_metrics(source, grades):
    """Indicators of the line for each period:
    - direction: (A - B) / (A + B) over all its connections (A: passengers on board in the fwd direction);
    - local: sum |A - B| / sum (A + B), the asymmetry of the connections, also when it cancels out along the line;
    - uphill: (up - down) / (up + down) over the connections climbing at least MIN_GRADE (positive: more passengers uphill);
    and tidal: the mean over the connections, weighted by their passengers in the peaks, of |index morning - index evening| / 2
    (1: everybody goes one way in the morning and the other way in the evening)."""
    if source is None:
        return None
    result = {}
    for period in PERIODS:
        f, r = period_flows(source, "fwd", period), period_flows(source, "rev", period)
        if f is None or r is None:
            result[period] = None
            continue
        up = down = 0.0
        for k, grade in enumerate(grades):
            if grade is not None and abs(grade) >= MIN_GRADE:
                up, down = (up + f[k], down + r[k]) if grade > 0 else (up + r[k], down + f[k])
        total = f.sum() + r.sum()
        result[period] = {
            "direction": _index(f.sum(), r.sum()),
            "local": np.abs(f - r).sum() / total if total > 0 else np.nan,
            "uphill": _index(up, down),
        }

    flows = {p: (period_flows(source, "fwd", p), period_flows(source, "rev", p)) for p in ("am", "pm")}
    if all(f is not None and r is not None for f, r in flows.values()):
        (fa, ra), (fp, rp) = flows["am"], flows["pm"]
        weights = fa + ra + fp + rp
        with np.errstate(invalid = "ignore", divide = "ignore"):
            change = np.abs(np.nan_to_num((fa - ra) / (fa + ra)) - np.nan_to_num((fp - rp) / (fp + rp))) / 2
        result["tidal"] = float((change * weights).sum() / weights.sum()) if weights.sum() > 0 else np.nan
    else:
        result["tidal"] = np.nan

    return result


def _json_number(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else round(float(v), 3)


def write(output_dir, prefix, reference_name, lines):
    """lines: {line: {submode, line_id, titles, order, stops, z, grade, src: {ref, matsim}}} (line_comparison.run)."""
    segment_rows, line_rows = [], []
    for line, d in lines.items():
        d["metrics"] = {s: line_metrics(d["src"][s], d["grade"]) for s in SOURCES}

        names = {code: d["stops"][code][0] if code in d["stops"] else code for code in d["order"]}
        for k, (a, b) in enumerate(zip(d["order"][:-1], d["order"][1:])):
            row = dict(line = line, line_id = d["line_id"], from_stop = names[a], to_stop = names[b], position = k,
                       elevation_from = d["z"].get(a), elevation_to = d["z"].get(b), grade = d["grade"][k])
            for s in SOURCES:
                column = "matsim" if s == "matsim" else prefix
                for period in PERIODS:
                    f, r = period_flows(d["src"][s], "fwd", period), period_flows(d["src"][s], "rev", period)
                    row[f"{period}_ab_{column}"] = np.nan if f is None else f[k]
                    row[f"{period}_ba_{column}"] = np.nan if r is None else r[k]
                    row[f"{period}_index_{column}"] = np.nan if f is None or r is None else _index(f[k], r[k])
            segment_rows.append(row)

        row = dict(line = line, line_id = d["line_id"], submode = d["submode"], direction_a = d["titles"]["fwd"])
        for s in SOURCES:
            column, m = ("matsim" if s == "matsim" else prefix), d["metrics"][s]
            for period in PERIODS:
                for name in ("direction", "local", "uphill"):
                    row[f"{period}_{name}_{column}"] = np.nan if not m or not m[period] else m[period][name]
            row[f"tidal_{column}"] = np.nan if not m else m["tidal"]
        line_rows.append(row)

        d["metrics"] = {s: None if m is None else {
            **{p: None if m[p] is None else {k: _json_number(v) for k, v in m[p].items()} for p in PERIODS},
            "tidal": _json_number(m["tidal"])} for s, m in d["metrics"].items()}

    pd.DataFrame(segment_rows).to_csv(os.path.join(output_dir, f"{prefix}_line_asymmetry_segments.csv"), index = False)
    pd.DataFrame(line_rows).to_csv(os.path.join(output_dir, f"{prefix}_line_asymmetry_lines.csv"), index = False)

    output_path = os.path.join(output_dir, f"{prefix}_line_asymmetry_map.html")
    payload = json.dumps({"lines": lines, "minGrade": MIN_GRADE, "periods": PERIOD_NAMES, "hours": PERIODS},
                         separators = (",", ":"), ensure_ascii = False)
    from visualize import TILE_URL  # pyright: ignore[reportMissingImports]
    html = PAGE.replace("__TILE_URL__", TILE_URL).replace("__REF__", reference_name).replace("__DATA__", payload.replace("</", "<\\/"))
    with open(output_path, "w", encoding = "utf-8") as f:
        f.write(html)
    print(f"Asymmetry map written to {output_path}")

    return output_path


# ---------------------------------------------------------------------------
# Page: left third = controls, line indicators, charts and the ranking of the lines; the rest = map
# ---------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__REF__ line asymmetries</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --matsim:#b0242a; --ref:#4a7bb5; --hover:#f1f0ec; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --ink:#f2f1ee; --ink2:#aeada8; --grid:#3a3a38; --hover:#2a2a28; } }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--ink); font:13px/1.4 system-ui, sans-serif; }
  #app { display:flex; height:100vh; }
  #menu { width:36vw; min-width:340px; display:flex; flex-direction:column; border-right:1px solid var(--grid); }
  #top { padding:12px 14px; border-bottom:1px solid var(--grid); flex:none; }
  #graphs { padding:8px 14px 24px; overflow:auto; flex:1; }
  #map { flex:1; }
  @media (max-width: 760px) { #app { flex-direction:column-reverse; } #menu { width:100%; min-width:0; height:55vh; } #map { min-height:45vh; } }
  h1 { font-size:15px; margin:0 0 8px; }
  h2 { font-size:13px; margin:16px 0 4px; }
  hr.divider { border:0; border-top:2px solid var(--ink2); margin:18px 0 0; opacity:0.6; }
  select, input[type=text] { width:100%; padding:6px 8px; font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--grid); border-radius:6px; }
  input[type=text] { margin-bottom:6px; }
  label { margin-right:10px; white-space:nowrap; }
  label.off { opacity:0.45; }
  .row { margin-top:7px; }
  .legend { height:10px; border-radius:3px; margin-top:8px; }
  .legend-labels { display:flex; justify-content:space-between; font-size:11px; color:var(--ink2); gap:8px; }
  .muted { color:var(--ink2); }
  .note { color:var(--ink2); font-size:12px; margin:2px 0 6px; }
  .key span { display:inline-block; width:10px; height:10px; margin:0 4px 0 8px; border-radius:2px; vertical-align:-1px; }
  .key span.dash { height:0; border-top:2px dashed var(--ink2); width:14px; border-radius:0; vertical-align:3px; }
  svg text { fill:var(--ink2); font-family:system-ui, sans-serif; }
  svg .title { fill:var(--ink); font-weight:600; }
  svg .grid { stroke:var(--grid); stroke-width:1; }
  svg .zero { stroke:var(--ink2); stroke-width:1; stroke-dasharray:3 3; }
  table.t { border-collapse:collapse; font-size:12px; width:100%; font-variant-numeric:tabular-nums; }
  table.t td, table.t th { padding:2px 6px; text-align:right; }
  table.t td:first-child, table.t th:first-child { text-align:left; }
  table.t th { font-weight:600; border-bottom:1px solid var(--grid); }
  table.rank tr.line { cursor:pointer; }
  table.rank tr.line:hover, table.rank tr.sel { background:var(--hover); }
  table.rank th.sort { cursor:pointer; text-decoration:underline dotted; }
  .tip { font-size:12px; }
</style></head>
<body><div id="app">
  <div id="menu">
    <div id="top">
      <h1>Line asymmetries: __REF__ and MATSim</h1>
      <input id="filter" type="text" placeholder="Filter the lines">
      <select id="line"></select>
      <div class="row">Data:
        <label><input type="radio" name="source" value="ref" checked> __REF__</label>
        <label><input type="radio" name="source" value="matsim"> MATSim</label>
        <label><input type="checkbox" id="all"> All the lines</label>
      </div>
      <div class="row" id="periods">Period:</div>
      <div class="row">Colour the connections by:<br>
        <label><input type="radio" name="color" value="direction" checked> busier direction</label>
        <label id="uphill-label"><input type="radio" name="color" value="uphill"> uphill / downhill</label>
        <label><input type="radio" name="color" value="tidal"> morning / evening reversal</label>
      </div>
      <div class="row"><label><input type="checkbox" id="stops" checked> Stops: boardings / alightings</label></div>
      <div class="legend" id="gradient"></div>
      <div class="legend-labels" id="legend-labels"></div>
      <div class="legend" id="stop-gradient"></div>
      <div class="legend-labels" id="stop-labels"></div>
    </div>
    <div id="graphs"></div>
  </div>
  <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const PAYLOAD = __DATA__, DATA = PAYLOAD.lines, MIN_GRADE = PAYLOAD.minGrade, PERIODS = PAYLOAD.periods, HOURS = PAYLOAD.hours;
const SRC = {ref: "__REF__", matsim: "MATSim"};
const SRC_COLOR = {ref: "#4a7bb5", matsim: "#b0242a"};
// diverging purple <-> orange (PuOr) for the connections, teal <-> magenta for the stops; grey midpoints
const ORANGE = "#c26a12", PURPLE = "#6b4fa8", TEAL = "#1b7f7a", MAGENTA = "#a8327a", MID = "#c9c8c3", FLAT = "#a9a8a3";
const DIR_A = "#1f6f8b", DIR_B = "#d08a1e";   // the two directions of a line in the charts
const STRONG = 0.6;                            // index giving the full colour
const SUBMODE_ORDER = ["metro", "LEB", "train", "tram", "bus"];
const HAS_Z = Object.values(DATA).some(d => d.grade && d.grade.some(g => g != null));

const map = L.map("map", {preferCanvas: true}).setView([46.52, 6.63], 13);
L.tileLayer("__TILE_URL__", {attribution: "© swisstopo", maxZoom: 19, opacity: 0.45}).addTo(map);   // faded: the colours are the data
const layer = L.layerGroup().addTo(map);

let current = null, selection = null, sortKey = "local";

// ---------------------------------------------------------------- numbers
const byId = id => document.getElementById(id);
const sum = a => a.reduce((s, v) => s + v, 0);
const fmt = v => v == null || !isFinite(v) ? "–" : Math.round(v).toLocaleString("en");
const pct = v => v == null || !isFinite(v) ? "–" : (v > 0 ? "+" : "") + Math.round(100 * v) + " %";
// share of the passengers in direction A from the index (A - B) / (A + B): -1 -> 0 %, 0 -> 50 %, 1 -> 100 %
const toShare = i => i == null || !isFinite(i) ? null : (1 + i) / 2;
const share = i => i == null || !isFinite(i) ? "–" : Math.round(50 * (1 + i)) + " %";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const index = (a, b) => a + b > 0 ? (a - b) / (a + b) : null;
const checked = name => document.querySelector(`input[name=${name}]:checked`).value;

// passengers on board of each connection in the period, in the order of the connections of direction A
function flows(line, s, o, period) {
  const src = DATA[line].src[s];
  if (!src || !src.daily[o]) return null;
  const v = period === "day" ? src.daily[o].slice() : src.hourly[o].map(row => sum(HOURS[period].map(h => row[h])));
  return o === "fwd" ? v : v.reverse();
}
function stopValues(line, s, code, period) {   // [boardings, alightings] of both directions
  const src = DATA[line].src[s], v = src && src.stops[code];
  if (!v) return [0, 0];
  const hours = period === "day" ? null : HOURS[period];
  return v.map(series => hours ? sum(hours.map(h => series[h])) : sum(series));
}

// ---------------------------------------------------------------- colours
function hex(c) { return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); }
function mix(c1, c2, t) { const a = hex(c1), b = hex(c2); return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join(""); }
const diverging = (v, neg, pos) => v == null ? FLAT : v < 0 ? mix(MID, neg, Math.min(1, -v / STRONG)) : mix(MID, pos, Math.min(1, v / STRONG));
const sequential = v => v == null ? FLAT : mix(MID, "#3d2f73", Math.min(1, Math.abs(v) / STRONG));

function legend(mode) {
  const g = byId("gradient"), labels = byId("legend-labels");
  const ramp = f => "linear-gradient(to right, " + [-1, -0.5, 0, 0.5, 1].map(t => f(t * STRONG)).join(",") + ")";
  if (mode === "uphill") {
    g.style.background = ramp(v => diverging(v, PURPLE, ORANGE));
    labels.innerHTML = `<span>more passengers downhill</span><span>same</span><span>more passengers uphill</span>`;
  } else {
    g.style.background = "linear-gradient(to right, " + [0, 0.5, 1].map(t => sequential(t * STRONG)).join(",") + ")";
    labels.innerHTML = mode === "tidal"
      ? `<span>same direction morning and evening</span><span>reversed (arrow: morning)</span>`
      : `<span>both directions alike</span><span>&ge; ${Math.round(100 * (1 + STRONG) / 2)} % one way (arrow)</span>`;
  }
  const show = byId("stops").checked;
  byId("stop-gradient").style.display = byId("stop-labels").style.display = show ? "" : "none";
  byId("stop-gradient").style.background = ramp(v => diverging(v, MAGENTA, TEAL));
  byId("stop-labels").innerHTML = `<span>stops: more alightings</span><span>balanced</span><span>more boardings</span>`;
}

// ---------------------------------------------------------------- charts (SVG)
function svg(w, h, body, label) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" role="img" aria-label="${esc(label)}">${body}</svg>`; }
function niceMax(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }

// series: [{color, values, kind: "bar" | "line", dash, name}]; opts: lo, hi, tick (axis labels), zero (dashed line)
function chart(title, labels, series, opts = {}) {
  const W = 600, H = opts.height || 220, L0 = 46, R0 = 8, T0 = 22, B0 = opts.bottom ?? 86;
  const vals = series.flatMap(s => (s.values || []).filter(v => v != null));
  const lo = opts.lo ?? Math.min(0, ...vals), hi = opts.hi ?? niceMax(Math.max(...vals, 1));
  const tick = opts.tick || fmt;
  const y = v => T0 + (H - T0 - B0) * (1 - (v - lo) / (hi - lo || 1));
  const n = labels.length, step = (W - L0 - R0) / Math.max(n, 1);
  let out = `<text class="title" x="${L0}" y="13" font-size="12">${esc(title)}</text>`;
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    out += `<line class="grid" x1="${L0}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}"/><text x="${L0 - 4}" y="${y(v) + 3}" font-size="9" text-anchor="end">${tick(v)}</text>`;
  }
  if (opts.zero != null) out += `<line class="zero" x1="${L0}" x2="${W - R0}" y1="${y(opts.zero)}" y2="${y(opts.zero)}"/>`;
  const active = series.filter(s => s.values), bars = active.filter(s => s.kind === "bar");
  const bw = Math.max(1, Math.min(12, step * 0.8 / Math.max(bars.length, 1) - 2));
  const base = opts.base ?? Math.max(lo, 0);   // the bars start from this value
  active.forEach(s => {
    if (s.kind === "bar") {
      const bi = bars.indexOf(s);
      s.values.forEach((v, i) => {
        if (v == null) return;
        const x = L0 + step * (i + 0.5) + (bi - (bars.length - 1) / 2) * (bw + 2) - bw / 2;
        out += `<rect x="${x}" y="${Math.min(y(v), y(base))}" width="${bw}" height="${Math.max(1, Math.abs(y(v) - y(base)))}" rx="2" fill="${s.color}"><title>${esc(labels[i])}: ${esc(s.name)} ${tick(v)}</title></rect>`;
      });
    } else {
      let path = "";
      s.values.forEach((v, i) => { if (v != null) path += (path && s.values[i - 1] != null ? " L" : " M") + `${L0 + step * (i + 0.5)},${y(v)}`; });
      out += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2"${s.dash ? ' stroke-dasharray="5 3"' : ""}/>`;
      s.values.forEach((v, i) => { if (v != null) out += `<circle cx="${L0 + step * (i + 0.5)}" cy="${y(v)}" r="6" fill="transparent"><title>${esc(labels[i])}: ${esc(s.name)} ${tick(v)}</title></circle>`; });
    }
  });
  if (n <= 60) labels.forEach((t, i) => {
    const x = L0 + step * (i + 0.5);
    if (opts.hours) { if (i % 3 === 0) out += `<text x="${x}" y="${H - B0 + 12}" font-size="9" text-anchor="middle">${esc(t)}</text>`; }
    else out += `<text x="${x}" y="${H - B0 + 10}" font-size="8" text-anchor="end" transform="rotate(-60 ${x} ${H - B0 + 10})">${esc(t)}</text>`;
  });
  return svg(W, H, out, title);
}
const HOUR_LABELS = Array.from({length: 24}, (_, h) => String(h));
const hourChart = (title, series, opts = {}) => chart(title, HOUR_LABELS, series, {height: 170, bottom: 24, hours: true, ...opts});
const key = items => `<div class="key">${items.map(([color, name, dash]) => `<span class="${dash ? "dash" : ""}" style="${dash ? "border-color:" + color : "background:" + color}"></span>${esc(name)}`).join("")}</div>`;

// ---------------------------------------------------------------- panels
function shortName(name) { return name.replace(/^(Lausanne|Renens VD|Ecublens VD|Chavannes-R\.|Prilly|Pully|Epalinges|Genève|Carouge GE|Lancy|Meyrin|Vernier|Onex), /, ""); }
const stopName = (d, code) => d.stops[code] ? d.stops[code][0] : code;

function selectionPanel() {
  const d = DATA[current];
  if (!selection || selection.line !== current) return `<h2>Selection</h2><div class="muted">Click a connection or a stop on the map for its hours.</div>`;
  const out = [];
  if (selection.type === "segment") {
    const k = selection.k, a = stopName(d, d.order[k]), b = stopName(d, d.order[k + 1]), g = d.grade ? d.grade[k] : null;
    out.push(`<h2>${esc(a)} ↔ ${esc(b)}: passengers on board per hour</h2>`);
    if (g != null) out.push(`<div class="note">${a} → ${b}: ${g > 0 ? "uphill" : "downhill"} ${Math.abs(100 * g).toFixed(1)} % (from the stop elevations)</div>`);
    out.push(key([[DIR_A, `${a} → ${b}`], [DIR_B, `${b} → ${a}`]]), `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">`);
    const n = d.order.length;
    for (const s of ["ref", "matsim"]) {
      const src = d.src[s];
      const series = [["fwd", DIR_A, k], ["rev", DIR_B, n - 2 - k]].map(([o, color, kk]) => ({color, kind: "line", name: o === "fwd" ? `${a} → ${b}` : `${b} → ${a}`,
        values: src && src.hourly[o] ? src.hourly[o][kk] : null}));
      out.push(hourChart(SRC[s], series));
    }
    out.push("</div>");
  } else {
    const code = selection.code;
    out.push(`<h2>${esc(stopName(d, code))}: boardings and alightings per hour (both directions)</h2>`,
      key([[TEAL, "boardings"], [MAGENTA, "alightings"]]), `<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">`);
    for (const s of ["ref", "matsim"]) {
      const v = d.src[s] && d.src[s].stops[code];
      out.push(hourChart(SRC[s], [{color: TEAL, kind: "line", name: "boardings", values: v ? v[0] : null},
                                  {color: MAGENTA, kind: "line", name: "alightings", values: v ? v[1] : null}]));
    }
    out.push("</div>");
  }
  return out.join("");
}

function metricsTable() {
  const d = DATA[current], period = checked("period");
  const cell = (s, f) => { const m = d.metrics[s]; return m ? f(m) : "–"; };
  const row = (name, f, title) => `<tr title="${esc(title)}"><td>${name}</td>${["ref", "matsim"].map(s => `<td>${cell(s, f)}</td>`).join("")}</tr>`;
  return `<table class="t"><tr><th>${esc(current)} (${esc(d.submode)}) – ${esc(PERIODS[period].toLowerCase())}</th><th>__REF__</th><th>MATSim</th></tr>` +
    row(`passengers going towards ${esc(shortName(stopName(d, d.order[d.order.length - 1])))}`, m => share(m[period] && m[period].direction),
        "share of the passengers on board (summed over the connections) going towards the end of the longest route; 50 %: both directions alike") +
    row("asymmetry of the connections", m => m[period] ? Math.round(100 * m[period].local) + " %" : "–",
        "sum |A - B| / sum (A + B): also the asymmetries that cancel out along the line") +
    (HAS_Z ? row(`passengers riding uphill (slopes ≥ ${100 * MIN_GRADE} %)`, m => share(m[period] && m[period].uphill), "on the connections climbing at least the gradient, share of the passengers riding uphill; 50 %: as many downhill") : "") +
    row("morning / evening reversal", m => m.tidal == null ? "–" : Math.round(100 * m.tidal) + " %",
        "mean of |index morning - index evening| / 2 over the connections, weighted by their passengers in the peaks") + "</table>";
}

function lineCharts() {
  const d = DATA[current], period = checked("period"), n = d.order.length, out = [];
  const end = shortName(stopName(d, d.order[n - 1])), start = shortName(stopName(d, d.order[0]));

  // share of the boardings towards the end of the line, per hour
  const share = s => {
    const src = d.src[s];
    if (!src || !src.boardings.fwd || !src.boardings.rev) return null;
    return src.boardings.fwd.map((v, h) => v + src.boardings.rev[h] > 0 ? v / (v + src.boardings.rev[h]) : null);
  };
  out.push(`<h2>Share of the boardings towards ${esc(end)}, per hour</h2>`,
    `<div class="note">Above 50 %: most passengers go towards ${esc(end)}; a curve crossing 50 % between the morning and the evening shows commuting.</div>`,
    key([[SRC_COLOR.ref, "__REF__"], [SRC_COLOR.matsim, "MATSim"]]),
    hourChart("", ["ref", "matsim"].map(s => ({color: SRC_COLOR[s], kind: "line", name: SRC[s], values: share(s)})),
              {lo: 0, hi: 1, zero: 0.5, tick: v => Math.round(100 * v) + " %"}));

  // index of each connection
  const labels = d.order.slice(0, -1).map((c, k) => `${shortName(stopName(d, c))} – ${shortName(stopName(d, d.order[k + 1]))}`);
  const idx = s => { const f = flows(current, s, "fwd", period), r = flows(current, s, "rev", period); return f && r ? f.map((v, k) => toShare(index(v, r[k]))) : null; };
  out.push(`<h2>Busier direction of each connection – ${esc(PERIODS[period].toLowerCase())}</h2>`,
    `<div class="note">Share of the passengers of each connection going towards ${esc(end)}: above 50 %, more passengers towards ${esc(end)}; below, towards ${esc(start)}.</div>`,
    key([[SRC_COLOR.ref, "__REF__"], [SRC_COLOR.matsim, "MATSim"]]),
    chart("", labels, ["ref", "matsim"].map(s => ({color: SRC_COLOR[s], kind: "bar", name: SRC[s], values: idx(s)})),
          {lo: 0, hi: 1, zero: 0.5, tick: v => Math.round(100 * v) + " %", base: 0.5}));

  // elevation profile
  if (d.grade && d.grade.some(g => g != null)) {
    const z = d.order.map(c => d.z[c] ?? null), known = z.filter(v => v != null);
    const lo = Math.floor(Math.min(...known) / 20) * 20, hi = Math.ceil(Math.max(...known) / 20) * 20;
    out.push(`<h2>Elevation of the stops (${esc(start)} → ${esc(end)})</h2>`,
      chart("", d.order.map(c => shortName(stopName(d, c))), [{color: "#52514e", kind: "line", name: "elevation", values: z}],
            {lo, hi: Math.max(hi, lo + 20), tick: v => Math.round(v) + " m"}));
  }
  return out.join("");
}

function rankTable() {
  const period = checked("period");
  const cols = [["direction", "towards end"], ["local", "asym. connections"], ...(HAS_Z ? [["uphill", "uphill"]] : []), ["tidal", "reversal"]];
  const value = (line, s, c) => { const m = DATA[line].metrics[s]; if (!m) return null; return c === "tidal" ? m.tidal : m[period] && m[period][c]; };
  const show = (v, c) => v == null ? "–" : (c === "local" || c === "tidal") ? Math.round(100 * v) + "" : share(v).replace(" %", "");
  const lines = Object.keys(DATA).filter(visible).sort((a, b) => Math.abs(value(b, "ref", sortKey) ?? -1) - Math.abs(value(a, "ref", sortKey) ?? -1));
  return `<h2>All the lines (%, __REF__ / MATSim) – ${esc(PERIODS[period].toLowerCase())}</h2>` +
    `<div class="note">towards end: share of the passengers going towards the end of the line; uphill: share riding uphill (50: balanced); asym. connections and reversal: as in the table above. Sorted by the __REF__ value of the underlined column, furthest from balance first (click a column to sort by it, a line to select it).</div>` +
    `<table class="t rank"><tr><th>line</th>${cols.map(([c, name]) => `<th class="sort" data-sort="${c}"${c === sortKey ? ' style="color:var(--ink)"' : ""}>${name}</th>`).join("")}</tr>` +
    lines.map(line => `<tr class="line${line === current ? " sel" : ""}" data-line="${esc(line)}"><td>${esc(line)}</td>` +
      cols.map(([c]) => `<td>${show(value(line, "ref", c), c)} / ${show(value(line, "matsim", c), c)}</td>`).join("") + "</tr>").join("") + "</table>";
}

function renderPanels() {
  byId("graphs").innerHTML = `<div id="selection">${selectionPanel()}</div><hr class="divider">` +
    `<h1 style="margin-top:12px">Selected line</h1>${metricsTable()}${lineCharts()}<hr class="divider">${rankTable()}`;
}

document.addEventListener("click", e => {
  const sort = e.target.closest("th.sort"), row = e.target.closest("tr.line");
  if (sort) { sortKey = sort.dataset.sort; renderPanels(); }
  else if (row) { byId("line").value = row.dataset.line; select(row.dataset.line); }
});

// ---------------------------------------------------------------- map
function arrow(pa, pb, color, size) {   // triangle in the middle of a -> b, pointing to b, size in pixels
  const A = map.latLngToLayerPoint(pa), B = map.latLngToLayerPoint(pb), dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy);
  if (len < 3 * size) return null;
  const ux = dx / len, uy = dy / len, mx = (A.x + B.x) / 2, my = (A.y + B.y) / 2;
  const pts = [[mx + ux * size, my + uy * size], [mx - ux * size - uy * size * 0.8, my - uy * size + ux * size * 0.8],
               [mx - ux * size + uy * size * 0.8, my - uy * size - ux * size * 0.8]];
  return L.polygon(pts.map(([x, y]) => map.layerPointToLatLng([x, y])), {color: "#0b0b0b", weight: 1, fillColor: color, fillOpacity: 1, interactive: false});
}

function draw(fit = false) {
  layer.clearLayers();
  const s = checked("source"), mode = checked("color"), period = mode === "tidal" ? "day" : checked("period");
  const lines = byId("all").checked ? Object.keys(DATA).filter(visible) : [current];
  legend(mode);

  const segments = [];
  for (const line of lines) {
    const d = DATA[line], f = flows(line, s, "fwd", period), r = flows(line, s, "rev", period);
    if (!f || !r) continue;
    const am = mode === "tidal" ? [flows(line, s, "fwd", "am"), flows(line, s, "rev", "am")] : null;
    const pm = mode === "tidal" ? [flows(line, s, "fwd", "pm"), flows(line, s, "rev", "pm")] : null;
    for (let k = 0; k < d.order.length - 1; k++) {
      const a = d.stops[d.order[k]], b = d.stops[d.order[k + 1]];
      if (!a || !b) continue;
      const seg = {line, k, a, b, f: f[k], r: r[k], total: f[k] + r[k], i: index(f[k], r[k]), grade: d.grade ? d.grade[k] : null};
      if (am) { seg.iam = index(am[0][k], am[1][k]); seg.ipm = index(pm[0][k], pm[1][k]); }
      segments.push(seg);
    }
  }
  const maxTotal = Math.max(1, ...segments.map(g => g.total));
  segments.sort((x, y) => x.total - y.total);

  for (const g of segments) {
    const pa = [g.a[1], g.a[2]], pb = [g.b[1], g.b[2]];
    let color, dir = g.i, strength = Math.abs(g.i ?? 0), extra = "";
    if (mode === "uphill") {
      const hilly = g.grade != null && Math.abs(g.grade) >= MIN_GRADE;
      color = hilly ? diverging(g.i == null ? null : g.i * Math.sign(g.grade), PURPLE, ORANGE) : FLAT;
      extra = g.grade == null ? "<br>no elevation" : `<br>${g.a[0]} → ${g.b[0]}: ${g.grade > 0 ? "uphill" : "downhill"} ${Math.abs(100 * g.grade).toFixed(1)} %${hilly ? "" : " (flat)"}`;
    } else if (mode === "tidal") {
      const t = g.iam != null && g.ipm != null ? (g.iam - g.ipm) / 2 : null;
      color = sequential(t); dir = g.iam; strength = Math.abs(t ?? 0);
      extra = `<br>going towards ${esc(g.b[0])}: ${share(g.iam)} of the passengers in the morning peak, ${share(g.ipm)} in the evening peak`;
    } else color = sequential(g.i);

    const tip = `<div class="tip"><b>${esc(g.line)}</b> ${esc(g.a[0])} ↔ ${esc(g.b[0])} (${SRC[s]}, ${esc(PERIODS[period].toLowerCase())})` +
      `<br>towards ${esc(g.b[0])}: ${fmt(g.f)} passengers (${share(g.i)}), towards ${esc(g.a[0])}: ${fmt(g.r)} (${share(g.i == null ? null : -g.i)})${extra}</div>`;
    L.polyline([pa, pb], {color, weight: 2 + 9 * Math.sqrt(g.total / maxTotal), opacity: 0.95, lineCap: "butt"})
      .bindTooltip(tip, {sticky: true})
      .on("click", () => { if (g.line !== current) { byId("line").value = g.line; select(g.line); } selection = {type: "segment", line: g.line, k: g.k}; byId("selection").innerHTML = selectionPanel(); })
      .addTo(layer);
    if (dir != null && strength >= 0.1 && g.total > 0) {
      const head = arrow(dir > 0 ? pa : pb, dir > 0 ? pb : pa, color, 4 + 4 * Math.sqrt(g.total / maxTotal));
      if (head) head.addTo(layer);
    }
  }

  if (byId("stops").checked) {
    const stops = {};
    for (const line of lines) {
      const d = DATA[line];
      for (const code of d.order) {
        if (!d.stops[code]) continue;
        const [b, a] = stopValues(line, s, code, period), v = stops[code] || (stops[code] = {p: d.stops[code], b: 0, a: 0, lines: []});
        v.b += b; v.a += a; v.lines.push(line);
      }
    }
    const maxStop = Math.max(1, ...Object.values(stops).map(v => v.b + v.a));
    for (const [code, v] of Object.entries(stops)) {
      const i = index(v.b, v.a);
      L.circleMarker([v.p[1], v.p[2]], {radius: 3 + 11 * Math.sqrt((v.b + v.a) / maxStop), color: "#0b0b0b", weight: 1, fillColor: diverging(i, MAGENTA, TEAL), fillOpacity: 0.95})
        .bindTooltip(`<div class="tip"><b>${esc(v.p[0])}</b> (${SRC[s]}, ${esc(PERIODS[period].toLowerCase())})<br>boardings ${fmt(v.b)} (${share(i)}), alightings ${fmt(v.a)} (${share(i == null ? null : -i)})` +
                     (v.lines.length > 1 ? `<br>lines ${esc(v.lines.join(", "))}` : "") + "</div>", {direction: "top"})
        .on("click", () => { const line = v.lines.includes(current) ? current : v.lines[0]; if (line !== current) { byId("line").value = line; select(line); }
                             selection = {type: "stop", line, code}; byId("selection").innerHTML = selectionPanel(); })
        .addTo(layer);
    }
  }

  if (fit) {
    const pts = (byId("all").checked ? lines : [current]).flatMap(line => Object.values(DATA[line].stops).map(p => [p[1], p[2]]));
    if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.1));
  }
}

function select(line) { current = line; selection = null; renderPanels(); draw(!byId("all").checked); }

// ---------------------------------------------------------------- menu
function visible(n) { const text = byId("filter").value.trim().toLowerCase(); return !text || n.toLowerCase().includes(text) || DATA[n].submode.toLowerCase().includes(text); }
function fillSelect() {
  const names = Object.keys(DATA).filter(visible), bySub = {};
  names.forEach(n => (bySub[DATA[n].submode] = bySub[DATA[n].submode] || []).push(n));
  const subs = Object.keys(bySub).sort((a, b) => (SUBMODE_ORDER.indexOf(a) + 1 || 99) - (SUBMODE_ORDER.indexOf(b) + 1 || 99));
  byId("line").innerHTML = subs.map(sub => `<optgroup label="${esc(sub)}">` + bySub[sub].sort((a, b) => a.localeCompare(b, "en", {numeric: true}))
    .map(n => `<option value="${esc(n)}"${n === current ? " selected" : ""}>${esc(n)}</option>`).join("") + "</optgroup>").join("");
}

byId("periods").innerHTML += Object.entries(PERIODS).map(([p, name], i) => ` <label><input type="radio" name="period" value="${p}"${i ? "" : " checked"}> ${esc(name)}</label>`).join("");
if (!HAS_Z) { byId("uphill-label").classList.add("off"); byId("uphill-label").querySelector("input").disabled = true; byId("uphill-label").title = "no elevations of the stops"; }
byId("filter").addEventListener("input", () => { fillSelect(); if (byId("line").value) select(byId("line").value); });
byId("line").addEventListener("change", e => select(e.target.value));
byId("all").addEventListener("change", () => draw(true));
byId("stops").addEventListener("change", () => draw());
document.querySelectorAll("input[name=source], input[name=color]").forEach(el => el.addEventListener("change", () => draw()));
document.querySelectorAll("input[name=period]").forEach(el => el.addEventListener("change", () => { renderPanels(); draw(); }));
map.on("zoomend", () => { if (current) draw(); });

fillSelect();
const first = Object.keys(DATA).find(n => DATA[n].submode === "metro") || Object.keys(DATA)[0];
if (first) { byId("line").value = first; select(first); }
</script></body></html>
"""
