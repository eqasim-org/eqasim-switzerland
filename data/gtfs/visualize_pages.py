"""HTML pages of data/gtfs/visualize.py: the left third of the page is the menu, the rest the map (Leaflet). The data is
embedded once as JSON and everything is drawn in the browser, as in analysis/pt/TL_comparison/line_comparison.py."""
import json

SWISSTOPO_TILES = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"


def page(kind, title, data):
    payload = json.dumps(data, separators = (",", ":"), ensure_ascii = False).replace("</", "<\\/")
    return (HEAD.replace("__TITLE__", title)
            + MENUS[kind]
            + COMMON_JS.replace("__DATA__", payload).replace("__TILES__", SWISSTOPO_TILES)
            + SCRIPTS[kind]
            + "</script></body></html>\n")


HEAD = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  :root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e4e3df; --accent:#33508f; --flag:#b0242a; }
  @media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --ink:#f2f1ee; --ink2:#aeada8; --grid:#3a3a38; } }
  * { box-sizing:border-box; }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--ink); font:13px/1.4 system-ui, sans-serif; }
  #app { display:flex; height:100vh; }
  #menu { width:33.333vw; min-width:320px; display:flex; flex-direction:column; border-right:1px solid var(--grid); }
  #top { padding:12px 14px; border-bottom:1px solid var(--grid); flex:none; }
  #panel { padding:8px 14px 24px; overflow:auto; flex:1; }
  #map { flex:1; }
  h1 { font-size:15px; margin:0 0 8px; }
  h2 { font-size:13px; margin:14px 0 4px; }
  input[type=text] { width:100%; padding:6px 8px; font:inherit; color:var(--ink); background:var(--bg); border:1px solid var(--grid); border-radius:6px; }
  input[type=range] { width:140px; vertical-align:middle; }
  label { margin-right:12px; white-space:nowrap; }
  .row { margin-top:8px; }
  .muted { color:var(--ink2); }
  .swatch { display:inline-block; width:14px; height:4px; border-radius:2px; vertical-align:middle; margin-right:5px; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; vertical-align:middle; margin-right:5px; }
  .legend div { margin:2px 0; }
  .list { margin-top:6px; max-height:28vh; overflow:auto; border:1px solid var(--grid); border-radius:6px; }
  .list div { padding:3px 8px; cursor:pointer; border-bottom:1px solid var(--grid); }
  .list div:hover { background:var(--grid); }
  table.t { border-collapse:collapse; font-size:12px; width:100%; }
  table.t td, table.t th { padding:1px 6px; text-align:left; vertical-align:top; }
  table.t td.n { text-align:right; }
  hr.divider { border:0; border-top:2px solid var(--ink2); margin:16px 0 0; opacity:0.6; }
  svg text { fill:var(--ink2); font-family:system-ui, sans-serif; }
  svg .title { fill:var(--ink); font-weight:600; }
  svg .grid { stroke:var(--grid); stroke-width:1; }
  button.link { background:none; border:0; padding:0; color:var(--accent); cursor:pointer; font:inherit; text-decoration:underline; }
  details summary { cursor:pointer; }
</style></head>
<body><div id="app">
"""

COMMON_JS = r"""
  <div id="map"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;
const byId = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const fmt = v => v == null ? "–" : Math.round(v).toLocaleString("en");
const sum = a => a.reduce((s, v) => s + v, 0);

const map = L.map("map", {preferCanvas: true, zoomControl: true}).setView([46.52, 6.63], 10);
const baseLayers = {
  "swisstopo (grayscale)": L.tileLayer("__TILES__", {attribution: "© swisstopo", maxZoom: 19}).addTo(map),
  "CARTO (outside Switzerland)": L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
                                             {attribution: "© OpenStreetMap, © CARTO", maxZoom: 19}),
};
const overlays = {};
if (DATA.perimeter) {
  overlays["Perimeter"] = L.geoJSON(DATA.perimeter, {style: {color: "#52514e", weight: 2, fill: false, dashArray: "6 4"}, interactive: false}).addTo(map);
}
L.control.layers(baseLayers, overlays, {collapsed: true}).addTo(map);

// pixels -> meters at the current zoom
const px = (v, lat) => v * 156543.03 * Math.cos(lat * Math.PI / 180) / Math.pow(2, map.getZoom());

// a path [[lat, lon], ...] shifted to the right of its direction by `meters`, each vertex along the mean normal
function offsetPath(path, meters) {
  if (path.length < 2 || !meters) return path;
  const k = 111320, c = Math.cos(path[0][0] * Math.PI / 180);
  const xy = path.map(p => [p[1] * k * c, p[0] * k]);
  return xy.map((p, i) => {
    const a = xy[Math.max(i - 1, 0)], b = xy[Math.min(i + 1, xy.length - 1)];
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1;
    return [path[i][0] + (-dx / len) * meters / k, path[i][1] + (dy / len) * meters / (k * c)];
  });
}

function niceMax(v) { if (v <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p; return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p; }

// bars per hour of the day
function hourBars(title, values, color) {
  const W = 560, H = 130, L0 = 34, R0 = 6, T0 = 18, B0 = 18, hi = niceMax(Math.max(...values, 1));
  const bw = (W - L0 - R0) / 24, y = v => T0 + (H - T0 - B0) * (1 - v / hi);
  let out = `<text class="title" x="${L0}" y="12" font-size="11">${esc(title)}</text>`;
  for (let i = 0; i <= 2; i++) { const v = hi * i / 2; out += `<line class="grid" x1="${L0}" x2="${W - R0}" y1="${y(v)}" y2="${y(v)}"/><text x="${L0 - 3}" y="${y(v) + 3}" font-size="9" text-anchor="end">${fmt(v)}</text>`; }
  values.forEach((v, h) => { out += `<rect x="${L0 + h * bw + 1}" y="${y(v)}" width="${bw - 2}" height="${y(0) - y(v)}" fill="${color}"><title>${h}:00 ${v}</title></rect>`; });
  for (let h = 0; h < 24; h += 3) out += `<text x="${L0 + (h + 0.5) * bw}" y="${H - 5}" font-size="9" text-anchor="middle">${h}</text>`;
  return `<svg viewBox="0 0 ${W} ${H}" width="100%">${out}</svg>`;
}
const hhmm = m => `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
"""

# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------

LINES_MENU = r"""
  <div id="menu">
    <div id="top">
      <h1>GTFS lines</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row"><input id="search" type="text" placeholder="Search a line or an agency (e.g. 8, TL, L2, SNCF)"></div>
      <div class="list" id="results"></div>
      <div class="row">Colour by:
        <label><input type="radio" name="colorBy" value="mode" checked> mode</label>
        <label><input type="radio" name="colorBy" value="source"> source feed</label>
        <label><input type="checkbox" id="showStops"> stops (zoomed in)</label>
      </div>
      <div class="row legend" id="legend"></div>
    </div>
    <div id="panel"></div>
  </div>
"""

LINES_JS = r"""
const LINES = DATA.lines, STOPS = DATA.stops;
let selected = null;
const hidden = new Set();   // modes or sources switched off in the legend

// the path of a direction: the shape, or the stops
LINES.forEach(line => line.d.forEach(dir => {
  dir.path = dir.p || dir.s.map(i => [STOPS[i][1], STOPS[i][2]]);
  let s = 90, w = 180, n = -90, e = -180;
  dir.path.forEach(([la, lo]) => { s = Math.min(s, la); n = Math.max(n, la); w = Math.min(w, lo); e = Math.max(e, lo); });
  dir.bounds = L.latLngBounds([s, w], [n, e]);
}));
const maxTrips = Math.max(1, ...LINES.flatMap(l => l.d.map(d => d.t)));
const colorBy = () => document.querySelector("input[name=colorBy]:checked").value;
const colorOf = line => colorBy() === "mode" ? DATA.modeColors[line.m] || "#888"
                                               : (line.s >= 0 ? DATA.sourceColors[line.s % DATA.sourceColors.length] : "#888");
const keyOf = line => colorBy() === "mode" ? line.m : (line.s >= 0 ? DATA.sources[line.s] : "unknown");

byId("subtitle").textContent = `${LINES.length.toLocaleString("en")} lines` + (DATA.date ? `, trips of ${DATA.date}` : ", all the trips of the feed") +
  ". Each direction is drawn on the right of its direction of travel; the width is the number of trips.";

const layer = L.layerGroup().addTo(map), highlight = L.layerGroup().addTo(map);

function draw() {
  layer.clearLayers();
  const view = map.getBounds().pad(0.3), zoom = map.getZoom();
  LINES.forEach((line, k) => {
    if (hidden.has(keyOf(line))) return;
    const color = colorOf(line);
    line.d.forEach(dir => {
      if (!view.intersects(dir.bounds)) return;
      const weight = 1 + 5 * Math.sqrt(dir.t / maxTrips);
      const shape = zoom >= 11 ? offsetPath(dir.path, px(weight / 2 + 1, dir.path[0][0])) : dir.path;
      L.polyline(shape, {color, weight, opacity: 0.75})
        .bindTooltip(`<b>${esc(line.n)}</b> ${esc(DATA.agencies[line.a])}<br>${esc(dir.l)}<br>${fmt(dir.t)} trips`, {sticky: true})
        .on("click", () => select(k)).addTo(layer);
    });
  });
  if (byId("showStops").checked && zoom >= 14) {
    STOPS.forEach(([name, la, lo]) => { if (view.contains([la, lo])) L.circleMarker([la, lo], {radius: 3, color: "#0b0b0b", weight: 1, fillColor: "#fff", fillOpacity: 1}).bindTooltip(esc(name)).addTo(layer); });
  }
  drawSelection();
}

function drawSelection() {
  highlight.clearLayers();
  if (selected == null) return;
  const line = LINES[selected];
  line.d.forEach(dir => {
    const shape = offsetPath(dir.path, px(5, dir.path[0][0]));
    L.polyline(shape, {color: "#0b0b0b", weight: 7, opacity: 0.9}).addTo(highlight);
    L.polyline(shape, {color: colorOf(line), weight: 4, opacity: 1}).bindTooltip(esc(dir.l), {sticky: true}).addTo(highlight);
    dir.s.forEach(i => L.circleMarker([STOPS[i][1], STOPS[i][2]], {radius: 4, color: "#0b0b0b", weight: 1.5, fillColor: "#fff", fillOpacity: 1}).bindTooltip(esc(STOPS[i][0])).addTo(highlight));
  });
}

function select(k, fit) {
  selected = k;
  const line = LINES[k];
  let html = `<h2>${esc(line.n)} <span class="muted">${esc(line.m)}, ${esc(DATA.agencies[line.a])}${line.s >= 0 ? ", " + esc(DATA.sources[line.s]) : ""}</span></h2>`;
  html += `<button class="link" onclick="selected=null;byId('panel').innerHTML='';drawSelection()">clear the selection</button>`;
  line.d.forEach(dir => {
    html += `<h2>${esc(dir.l)}</h2><div class="muted">${fmt(dir.t)} trips, ${dir.s.length} stops</div>`;
    html += hourBars("Departures per hour (first stop)", dir.h, colorOf(line));
    html += `<details><summary>Stops</summary><table class="t">${dir.s.map((i, n) => `<tr><td class="n">${n + 1}</td><td><button class="link" onclick="map.setView([${STOPS[i][1]}, ${STOPS[i][2]}], 16)">${esc(STOPS[i][0])}</button></td></tr>`).join("")}</table></details>`;
  });
  byId("panel").innerHTML = html;
  if (fit) map.fitBounds(L.latLngBounds(line.d.flatMap(d => d.path)).pad(0.1));
  drawSelection();
}

function legend() {
  const counts = {};
  LINES.forEach(line => counts[keyOf(line)] = (counts[keyOf(line)] || 0) + 1);
  const keys = colorBy() === "mode" ? DATA.modes.filter(m => counts[m]) : [...DATA.sources.filter(s => counts[s]), ...(counts.unknown ? ["unknown"] : [])];
  const colorOfKey = key => colorBy() === "mode" ? DATA.modeColors[key] : (key === "unknown" ? "#888" : DATA.sourceColors[DATA.sources.indexOf(key) % DATA.sourceColors.length]);
  byId("legend").innerHTML = (colorBy() === "source" && !DATA.sources.length ? `<div class="muted">The source feeds are written by data.gtfs.cleaned: re-run it to have them.</div>` : "") +
    keys.map(key => `<div><label><input type="checkbox" data-key="${esc(key)}" ${hidden.has(key) ? "" : "checked"}> <span class="swatch" style="background:${colorOfKey(key)}"></span>${esc(key)} <span class="muted">(${counts[key]})</span></label></div>`).join("");
  byId("legend").querySelectorAll("input").forEach(el => el.addEventListener("change", () => { el.checked ? hidden.delete(el.dataset.key) : hidden.add(el.dataset.key); draw(); }));
}

function search() {
  const text = byId("search").value.trim().toLowerCase();
  if (!text) { byId("results").innerHTML = ""; return; }
  const found = LINES.map((line, k) => [line, k]).filter(([line]) => line.n.toLowerCase().startsWith(text) || DATA.agencies[line.a].toLowerCase().includes(text) ||
    (line.s >= 0 && DATA.sources[line.s].toLowerCase().includes(text))).slice(0, 200);
  byId("results").innerHTML = found.map(([line, k]) => `<div data-k="${k}"><b>${esc(line.n)}</b> <span class="muted">${esc(line.m)}, ${esc(DATA.agencies[line.a])}</span></div>`).join("") ||
    `<div class="muted">no line</div>`;
  byId("results").querySelectorAll("div[data-k]").forEach(el => el.addEventListener("click", () => select(+el.dataset.k, true)));
}

byId("search").addEventListener("input", search);
document.querySelectorAll("input[name=colorBy]").forEach(el => el.addEventListener("change", () => { hidden.clear(); legend(); draw(); }));
byId("showStops").addEventListener("change", draw);
map.on("moveend", draw);
legend();
if (LINES.length) map.fitBounds(L.latLngBounds(LINES.flatMap(l => l.d.map(d => d.bounds.getCenter()))).pad(0.05));
draw();
"""

# ---------------------------------------------------------------------------
# Stops and transfers
# ---------------------------------------------------------------------------

TRANSFERS_MENU = r"""
  <div id="menu">
    <div id="top">
      <h1>GTFS transfers</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row legend">
        <div><label><input type="checkbox" class="origin" data-key="0" checked> <span class="line"></span>given by the GTFS <span class="muted" id="count0"></span></label></div>
        <div><label><input type="checkbox" class="origin" data-key="1" checked> <span class="line dashed"></span>computed or set by hand <span class="muted" id="count1"></span></label></div>
      </div>
      <div class="row legend">
        <div><label><input type="checkbox" class="issue" data-key="ok" checked> <span class="swatch" style="background:#8a8985"></span>plausible <span class="muted" id="count_ok"></span></label></div>
        <div><label><input type="checkbox" class="issue" data-key="fast" checked> <span class="swatch" style="background:#b0242a"></span>too fast: walking faster than <span id="fastValue"></span> m/s <span class="muted" id="count_fast"></span></label>
          <input type="range" id="fast" min="0.5" max="5" step="0.1"></div>
        <div><label><input type="checkbox" class="issue" data-key="slow" checked> <span class="swatch" style="background:#8a4fa8"></span>too slow: <span id="extraValue"></span> s longer than walking at 1 m/s <span class="muted" id="count_slow"></span></label>
          <input type="range" id="extra" min="60" max="900" step="30"></div>
        <div><label><input type="checkbox" class="issue" data-key="distance" checked> <span class="swatch" style="background:#d97a00"></span>too far: stops more than <span id="distanceValue"></span> m apart <span class="muted" id="count_distance"></span></label>
          <input type="range" id="distance" min="200" max="1500" step="50"></div>
      </div>
      <div class="row"><label><input type="checkbox" id="showStops" checked> stops (zoomed in), coloured by source feed</label></div>
      <div class="row legend muted" id="sources"></div>
    </div>
    <div id="panel"></div>
  </div>
  <style>
    .line { display:inline-block; width:22px; height:0; border-top:3px solid #52514e; vertical-align:middle; margin-right:5px; }
    .line.dashed { border-top-style:dashed; }
    .legend input[type=range] { width:100%; }
  </style>
"""

TRANSFERS_JS = r"""
const S = DATA.stops, R = DATA.rules, T = DATA.thresholds;
byId("fast").value = T.fast; byId("extra").value = T.extra; byId("distance").value = T.distance;
const ISSUES = {fast: ["Too fast", "#b0242a"], distance: ["Too far", "#d97a00"], slow: ["Too slow", "#8a4fa8"], ok: ["Plausible", "#8a8985"]};
const ORIGINS = ["given by the GTFS", "computed: between the stations of different feeds", "set by hand (MANUAL_TRANSFERS)",
                 "computed: rule of a nearby stop + walking"];
const shownIssues = new Set(Object.keys(ISSUES)), shownOrigins = new Set([0, 1]);
const computed = rule => rule[3] > 0 ? 1 : 0;

// the issues of a rule, the most serious first
function issuesOf(rule) {
  const [, , time, origin, dist] = rule, issues = [];
  if (dist > 50 && dist / Math.max(time, 1) > +byId("fast").value) issues.push("fast");
  if (dist > +byId("distance").value) issues.push("distance");
  // the time of the nearby rules includes the transfer at the nearby stop: not compared with walking alone
  if (origin !== 3 && time > dist / 1.0 + +byId("extra").value) issues.push("slow");
  return issues;
}
const issueOf = rule => issuesOf(rule)[0] || "ok";

byId("subtitle").textContent = `${S.length.toLocaleString("en")} stops, ${R.length.toLocaleString("en")} transfer rules between different stops (the removed ones are not shown).` +
  (DATA.hasRules ? "" : " Where the rules come from is written by data.gtfs.cleaned: re-run it to have it.");
byId("sources").innerHTML = DATA.sources.length ? "Stops: " + DATA.sources.map((s, k) => `<span class="dot" style="background:${DATA.sourceColors[k % DATA.sourceColors.length]}"></span>${esc(s)}`).join(" &nbsp;") : "";

const layer = L.layerGroup().addTo(map), highlight = L.layerGroup().addTo(map);
const stopLabel = i => `${esc(S[i][0])} <span class="muted">${esc(S[i][3])}</span>`;
const modesOf = i => DATA.modes.filter((m, k) => S[i][5] & (1 << k)).join(", ") || "not served";

function ruleTip(rule) {
  const [a, b, time, origin, dist] = rule, issues = issuesOf(rule);
  return `<b>${issues.length ? issues.map(i => ISSUES[i][0]).join(", ") : "Plausible"}</b> <span class="muted">${esc(ORIGINS[origin])}</span>` +
    `<br>${stopLabel(a)} (${modesOf(a)})<br>→ ${stopLabel(b)} (${modesOf(b)})<br>${fmt(dist)} m, ${fmt(time)} s` +
    (time > 0 ? `, ${(dist / time).toFixed(2)} m/s` : "");
}

function draw() {
  layer.clearLayers();
  const view = map.getBounds().pad(0.2), zoom = map.getZoom(), counts = {};
  R.forEach(rule => {
    const issue = issueOf(rule), origin = computed(rule);
    counts[issue] = (counts[issue] || 0) + 1;
    counts[origin] = (counts[origin] || 0) + 1;
    // the plausible rules are many: only zoomed in
    if (!shownIssues.has(issue) || !shownOrigins.has(origin) || (issue === "ok" && zoom < 13)) return;
    const a = S[rule[0]], b = S[rule[1]];
    if (!view.contains([a[1], a[2]]) && !view.contains([b[1], b[2]])) return;
    L.polyline([[a[1], a[2]], [b[1], b[2]]], {color: ISSUES[issue][1], weight: issue === "ok" ? 1.5 : 3, opacity: 0.85, dashArray: origin ? "5 4" : null})
      .bindTooltip(ruleTip(rule), {sticky: true}).addTo(layer);
  });
  if (byId("showStops").checked && zoom >= 14) S.forEach(([name, la, lo, id, source], i) => {
    if (!view.contains([la, lo])) return;
    const color = source >= 0 ? DATA.sourceColors[source % DATA.sourceColors.length] : "#888";
    L.circleMarker([la, lo], {radius: 3.5, color: "#0b0b0b", weight: 0.8, fillColor: color, fillOpacity: 1})
      .bindTooltip(`${stopLabel(i)}<br>${modesOf(i)}${source >= 0 ? "<br>" + esc(DATA.sources[source]) : ""}`).addTo(layer);
  });
  ["0", "1"].forEach(k => byId("count" + k).textContent = `(${fmt(counts[k] || 0)})`);
  Object.keys(ISSUES).forEach(k => byId("count_" + k).textContent = `(${fmt(counts[k] || 0)}${k === "ok" ? ", zoomed in" : ""})`);
  table();
}

// the flagged rules, to go through them
function table() {
  const order = {fast: 0, distance: 1, slow: 2};
  const rows = R.filter(rule => shownOrigins.has(computed(rule))).map(rule => [issueOf(rule), rule]).filter(([issue]) => issue !== "ok" && shownIssues.has(issue));
  rows.sort((x, y) => order[x[0]] - order[y[0]] || y[1][4] - x[1][4]);   // the longest first
  let html = `<h2>Flagged (${fmt(rows.length)})</h2><div class="muted">Click a row to go there.</div><table class="t"><tr><th></th><th>from</th><th>to</th><th class="n">m</th><th class="n">s</th></tr>`;
  rows.slice(0, 500).forEach(([issue, rule], n) => {
    html += `<tr data-n="${n}" style="cursor:pointer"><td><span class="swatch" style="background:${ISSUES[issue][1]}"></span></td><td>${esc(S[rule[0]][0])}</td><td>${esc(S[rule[1]][0])}</td><td class="n">${fmt(rule[4])}</td><td class="n">${fmt(rule[2])}</td></tr>`;
  });
  html += "</table>" + (rows.length > 500 ? `<div class="muted">The first 500 are listed.</div>` : "");
  byId("panel").innerHTML = html;
  byId("panel").querySelectorAll("tr[data-n]").forEach(el => el.addEventListener("click", () => {
    const rule = rows[+el.dataset.n][1], a = S[rule[0]], b = S[rule[1]];
    map.fitBounds(L.latLngBounds([[a[1], a[2]], [b[1], b[2]]]).pad(1.5), {maxZoom: 17});
    highlight.clearLayers();
    L.polyline([[a[1], a[2]], [b[1], b[2]]], {color: "#0b0b0b", weight: 7, opacity: 0.5}).addTo(highlight);
  }));
}

function sliders() {
  byId("fastValue").textContent = (+byId("fast").value).toFixed(1);
  byId("extraValue").textContent = byId("extra").value;
  byId("distanceValue").textContent = byId("distance").value;
}
["fast", "extra", "distance"].forEach(id => byId(id).addEventListener("input", () => { sliders(); draw(); }));
document.querySelectorAll("input.issue").forEach(el => el.addEventListener("change", () => { el.checked ? shownIssues.add(el.dataset.key) : shownIssues.delete(el.dataset.key); draw(); }));
document.querySelectorAll("input.origin").forEach(el => el.addEventListener("change", () => { el.checked ? shownOrigins.add(+el.dataset.key) : shownOrigins.delete(+el.dataset.key); draw(); }));
byId("showStops").addEventListener("change", draw);
map.on("moveend", draw);
sliders();
if (S.length) map.fitBounds(L.latLngBounds(S.map(s => [s[1], s[2]])).pad(0.02));
draw();
"""

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

SCHEDULES_MENU = r"""
  <div id="menu">
    <div id="top">
      <h1>GTFS schedules</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row"><input id="search" type="text" placeholder="Search a stop"></div>
      <div class="list" id="results"></div>
      <div class="row legend" id="legend"></div>
    </div>
    <div id="panel"><div class="muted">Click a stop on the map, or search it, to see the lines that serve it and their departures.</div></div>
  </div>
"""

SCHEDULES_JS = r"""
const G = DATA.groups, N = DATA.names, TR = DATA.trips, P = DATA.points;
byId("subtitle").textContent = `${G.length.toLocaleString("en")} stops, ${TR.length.toLocaleString("en")} trips` + (DATA.date ? ` on ${DATA.date}` : " (all the services of the feed)") + ".";
byId("legend").innerHTML = DATA.modes.filter(m => G.some(g => g[3].some(s => DATA.modes[s[3]] === m)))
  .map(m => `<span class="swatch" style="background:${DATA.modeColors[m]}"></span>${esc(m)}`).join(" &nbsp; ");

const stopsLayer = L.layerGroup().addTo(map), routeLayer = L.layerGroup().addTo(map);
G.forEach(([name, la, lo, services], k) => {
  const modes = [...new Set(services.map(s => s[3]))].sort((a, b) => a - b);
  L.circleMarker([la, lo], {radius: 4, color: "#0b0b0b", weight: 1, fillColor: DATA.modeColors[DATA.modes[modes[0]]], fillOpacity: 0.95})
    .bindTooltip(esc(name)).on("click", () => show(k)).addTo(stopsLayer);
});

function showRoute(key, color) {
  routeLayer.clearLayers();
  const path = (DATA.paths[key] || []).map(i => P[i]);
  if (path.length < 2) return;
  L.polyline(path, {color: "#0b0b0b", weight: 7, opacity: 0.6}).addTo(routeLayer);
  L.polyline(path, {color, weight: 4, opacity: 1}).addTo(routeLayer);
  path.forEach(p => L.circleMarker(p, {radius: 3, color: "#0b0b0b", weight: 1, fillColor: "#fff", fillOpacity: 1}).addTo(routeLayer));
  map.fitBounds(L.latLngBounds(path).pad(0.1));
}

function show(k) {
  const [name, la, lo, services] = G[k];
  map.setView([la, lo], Math.max(map.getZoom(), 14));
  const sorted = services.map((s, n) => [s, n]).sort((x, y) => N[x[0][0]].localeCompare(N[y[0][0]], "en", {numeric: true}) || y[0][5].length - x[0][5].length);
  let html = `<h2>${esc(name)}</h2><div class="muted">${services.length} line(s) and direction(s), ${fmt(sum(services.map(s => s[5].length)))} departures</div>`;
  sorted.forEach(([[line, place, terminal, mode, key, departures], n]) => {
    const color = DATA.modeColors[DATA.modes[mode]], hours = Array(24).fill(0);
    departures.forEach(([m]) => hours[Math.floor(m / 60) % 24]++);
    html += `<hr class="divider"><h2><span class="swatch" style="background:${color}"></span>${esc(N[line])} ${terminal ? "← from " : "→ "}${esc(N[place])}</h2>`;
    html += `<button class="link" data-key="${esc(key)}" data-color="${color}">show the route</button>`;
    html += hourBars(`${terminal ? "Arrivals" : "Departures"} per hour (${departures.length})`, hours, color);
    html += `<details><summary>${terminal ? "Arrival" : "Departure"} times</summary><table class="t">` + departures.map(([m, t]) => {
      const trip = TR[t];
      return `<tr><td>${hhmm(m)}</td><td class="muted">${esc(N[trip[0]])} ${hhmm(trip[2])} → ${esc(N[trip[1]])} ${hhmm(trip[3])}</td></tr>`;
    }).join("") + "</table></details>";
  });
  byId("panel").innerHTML = html;
  byId("panel").querySelectorAll("button[data-key]").forEach(el => el.addEventListener("click", () => showRoute(el.dataset.key, el.dataset.color)));
  routeLayer.clearLayers();
}

byId("search").addEventListener("input", () => {
  const text = byId("search").value.trim().toLowerCase();
  if (!text) { byId("results").innerHTML = ""; return; }
  const found = G.map((g, k) => [g, k]).filter(([g]) => g[0].toLowerCase().includes(text)).slice(0, 100);
  byId("results").innerHTML = found.map(([g, k]) => `<div data-k="${k}">${esc(g[0])} <span class="muted">${g[3].length} line(s)</span></div>`).join("") || `<div class="muted">no stop</div>`;
  byId("results").querySelectorAll("div[data-k]").forEach(el => el.addEventListener("click", () => show(+el.dataset.k)));
});
if (G.length) map.fitBounds(L.latLngBounds(G.map(g => [g[1], g[2]])).pad(0.05));
"""

MENUS   = {"lines": LINES_MENU, "transfers": TRANSFERS_MENU, "schedules": SCHEDULES_MENU}
SCRIPTS = {"lines": LINES_JS, "transfers": TRANSFERS_JS, "schedules": SCHEDULES_JS}
