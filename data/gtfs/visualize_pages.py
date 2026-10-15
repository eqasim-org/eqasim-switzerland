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
      <h1>GTFS transfers: checks of the merge</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row">Implausible times:
        faster than <input type="range" id="fast" min="0.5" max="5" step="0.1"> <span id="fastValue"></span> m/s,
        or <input type="range" id="extra" min="60" max="900" step="30"> <span id="extraValue"></span> s longer than walking at 1 m/s
      </div>
      <div class="row legend" id="legend"></div>
      <div class="row"><label><input type="checkbox" id="showStops" checked> stops (zoomed in), coloured by source feed</label></div>
    </div>
    <div id="panel"></div>
  </div>
"""

TRANSFERS_JS = r"""
const S = DATA.stops, R = DATA.rules, T = DATA.thresholds;
byId("fast").value = T.fast; byId("extra").value = T.extra;
const CATEGORIES = [
  // [key, label, color, dashed, always drawn (else only zoomed in)]
  ["missing", `No rule to a stop of another mode ≤ ${T.nearby} m away, though rules to that mode exist`, "#d4a017", true, true],
  ["fast", "Time too short for the distance", "#b0242a", false, true],
  ["slow", "Time much longer than walking", "#8a4fa8", false, true],
  ["removed_distance", `Removed: longer than ${T.removedDistance} m`, "#52514e", true, true],
  ["removed_duration", `Removed: longer than ${T.removedDuration} s`, "#8a8985", true, true],
  ["nearby", "Added between neighbouring stops without a rule (rule nearby + walk)", "#c77d1a", false, false],
  ["manual", "Set by hand (MANUAL_TRANSFERS)", "#6b4fbb", false, true],
  ["bridge", "Added between the stations of different feeds", "#2a78d6", false, false],
  ["feed", "From the source feeds", "#2a9d8f", false, false],
];
const COLOR = Object.fromEntries(CATEGORIES.map(c => [c[0], c[2]]));
const shown = new Set(CATEGORIES.map(c => c[0]));

function categoryOf(rule) {
  const [, , time, origin, status, dist] = rule;
  if (status === 1) return "removed_distance";
  if (status === 2) return "removed_duration";
  if (origin === 2) return "manual";
  if (origin === 3) return "nearby";  // walking is part of their time: not checked against it
  if (dist > 50 && dist / Math.max(time, 1) > +byId("fast").value) return "fast";
  if (time > dist / 1.0 + +byId("extra").value) return "slow";
  return origin === 1 ? "bridge" : "feed";
}

byId("subtitle").textContent = `${S.length.toLocaleString("en")} stops, ${R.length.toLocaleString("en")} rules between different stops.` +
  (DATA.hasRules ? "" : " The origin of the rules and the removed ones are written by data.gtfs.cleaned: re-run it to have them.");

const layer = L.layerGroup().addTo(map), highlight = L.layerGroup().addTo(map);
const stopLabel = i => `${esc(S[i][0])} <span class="muted">${esc(S[i][3])}</span>`;
const modesOf = i => DATA.modes.filter((m, k) => S[i][5] & (1 << k)).join(", ") || "not served";

function ruleTip(rule, cat) {
  const [a, b, time, , , dist] = rule;
  return `<b>${esc(CATEGORIES.find(c => c[0] === cat)[1])}</b><br>${stopLabel(a)} (${modesOf(a)})<br>→ ${stopLabel(b)} (${modesOf(b)})<br>${fmt(dist)} m, ${fmt(time)} s` +
    (time > 0 ? `, ${(dist / time).toFixed(2)} m/s` : "");
}

function draw() {
  layer.clearLayers();
  const view = map.getBounds().pad(0.2), zoom = map.getZoom(), counts = {};
  R.forEach(rule => {
    const cat = categoryOf(rule);
    counts[cat] = (counts[cat] || 0) + 1;
    const def = CATEGORIES.find(c => c[0] === cat);
    if (!shown.has(cat) || (!def[4] && zoom < 13)) return;
    const a = S[rule[0]], b = S[rule[1]];
    if (!view.contains([a[1], a[2]]) && !view.contains([b[1], b[2]])) return;
    L.polyline([[a[1], a[2]], [b[1], b[2]]], {color: def[2], weight: def[4] ? 3 : 1.5, opacity: 0.85, dashArray: def[3] ? "5 4" : null})
      .bindTooltip(ruleTip(rule, cat), {sticky: true}).addTo(layer);
  });
  counts.missing = DATA.missing.length;
  if (shown.has("missing")) DATA.missing.forEach(([a, b, dist]) => {
    if (!view.contains([S[a][1], S[a][2]])) return;
    L.polyline([[S[a][1], S[a][2]], [S[b][1], S[b][2]]], {color: COLOR.missing, weight: 3, opacity: 0.9, dashArray: "2 5"})
      .bindTooltip(`<b>No rule</b> between<br>${stopLabel(a)} (${modesOf(a)})<br>and ${stopLabel(b)} (${modesOf(b)})<br>${fmt(dist)} m`, {sticky: true}).addTo(layer);
  });
  if (byId("showStops").checked && zoom >= 14) S.forEach(([name, la, lo, id, source], i) => {
    if (!view.contains([la, lo])) return;
    const color = source >= 0 ? DATA.sourceColors[source % DATA.sourceColors.length] : "#888";
    L.circleMarker([la, lo], {radius: 3.5, color: "#0b0b0b", weight: 0.8, fillColor: color, fillOpacity: 1})
      .bindTooltip(`${stopLabel(i)}<br>${modesOf(i)}${source >= 0 ? "<br>" + esc(DATA.sources[source]) : ""}`).addTo(layer);
  });
  legend(counts);
  table(counts);
}

function legend(counts) {
  byId("legend").innerHTML = CATEGORIES.map(([key, label, color, dashed, always]) =>
    `<div><label><input type="checkbox" data-key="${key}" ${shown.has(key) ? "checked" : ""}> <span class="swatch" style="background:${color}${dashed ? ";opacity:0.6" : ""}"></span>${esc(label)} <span class="muted">(${fmt(counts[key] || 0)}${always ? "" : ", zoomed in"})</span></label></div>`).join("") +
    (DATA.sources.length ? `<div class="muted" style="margin-top:6px">Stops: ` + DATA.sources.map((s, k) => `<span class="dot" style="background:${DATA.sourceColors[k % DATA.sourceColors.length]}"></span>${esc(s)}`).join(" &nbsp;") + "</div>" : "");
  byId("legend").querySelectorAll("input").forEach(el => el.addEventListener("change", () => { el.checked ? shown.add(el.dataset.key) : shown.delete(el.dataset.key); draw(); }));
}

// the flagged rules and stop pairs, to go through them
function table(counts) {
  const rows = [];
  R.forEach(rule => { const cat = categoryOf(rule); if (["fast", "slow", "removed_distance", "removed_duration"].includes(cat)) rows.push([cat, rule]); });
  DATA.missing.forEach(m => rows.push(["missing", [m[0], m[1], 0, 0, 0, m[2]]]));
  const order = {missing: 0, fast: 1, slow: 2, removed_distance: 3, removed_duration: 4};
  // the closest stops without a rule first, the longest rules first for the others
  rows.sort((x, y) => order[x[0]] - order[y[0]] || (x[0] === "missing" ? x[1][5] - y[1][5] : y[1][5] - x[1][5]));
  let html = `<h2>Flagged (${fmt(rows.length)})</h2><div class="muted">Click a row to go there.</div><table class="t"><tr><th></th><th>from</th><th>to</th><th class="n">m</th><th class="n">s</th></tr>`;
  rows.slice(0, 500).forEach(([cat, rule], n) => {
    html += `<tr data-n="${n}" style="cursor:pointer"><td><span class="swatch" style="background:${COLOR[cat]}"></span></td><td>${esc(S[rule[0]][0])}</td><td>${esc(S[rule[1]][0])}</td><td class="n">${fmt(rule[5])}</td><td class="n">${cat === "missing" ? "–" : fmt(rule[2])}</td></tr>`;
  });
  html += "</table>" + (rows.length > 500 ? `<div class="muted">The first 500 are listed.</div>` : "");
  byId("panel").innerHTML = html;
  byId("panel").querySelectorAll("tr[data-n]").forEach(el => el.addEventListener("click", () => {
    const [cat, rule] = rows[+el.dataset.n], a = S[rule[0]], b = S[rule[1]];
    map.fitBounds(L.latLngBounds([[a[1], a[2]], [b[1], b[2]]]).pad(1.5), {maxZoom: 17});
    highlight.clearLayers();
    L.polyline([[a[1], a[2]], [b[1], b[2]]], {color: "#0b0b0b", weight: 7, opacity: 0.5}).addTo(highlight);
  }));
}

function sliders() {
  byId("fastValue").textContent = (+byId("fast").value).toFixed(1);
  byId("extraValue").textContent = byId("extra").value;
}
["fast", "extra"].forEach(id => byId(id).addEventListener("input", () => { sliders(); draw(); }));
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
