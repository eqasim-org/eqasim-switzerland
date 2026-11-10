"""Shared parts of the HTML maps of analysis/pt/transit_schedule (page.py and transfers.py): the left third of the page is
the menu, the rest the map (Leaflet on the swisstopo national map, with a background dimming control). The data is
embedded once as JSON and everything is drawn in the browser."""
import json

SWISSTOPO_TILES = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"
SWISSTOPO_COLOR_TILES = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-farbe/default/current/3857/{z}/{x}/{y}.jpeg"

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
  @media (max-width: 700px) {
    #app { flex-direction:column; }
    #menu { width:100%; min-width:0; max-height:45vh; border-right:0; border-bottom:1px solid var(--grid); }
    #top, #panel { padding-left:16px; padding-right:16px; }
  }
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
  .dimmer { background:#fff; color:#0b0b0b; padding:4px 8px; border-radius:4px; box-shadow:0 1px 4px rgba(0,0,0,.3); font:12px system-ui, sans-serif; }
  .dimmer input { width:110px; }
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
const attribution = '© <a href="https://www.swisstopo.admin.ch/">swisstopo</a>';
const baseLayers = {
  "swisstopo (grayscale)": L.tileLayer("__TILES__", {attribution, maxZoom: 19}).addTo(map),
  "swisstopo (colour)": L.tileLayer("__COLOR_TILES__", {attribution, maxZoom: 19}),
};
const overlays = {};
if (DATA.perimeter) {
  overlays["Perimeter"] = L.geoJSON(DATA.perimeter, {style: {color: "#52514e", weight: 2, fill: false, dashArray: "6 4"}, interactive: false}).addTo(map);
}
const layersControl = L.control.layers(baseLayers, overlays, {collapsed: true}).addTo(map);

// background opacity: dims the map tiles so that the data stands out
const Dimmer = L.Control.extend({
  onAdd() {
    const div = L.DomUtil.create("div", "dimmer");
    div.innerHTML = `<label style="margin:0">Background <input type="range" min="0.1" max="1" step="0.05" value="1"></label>`;
    L.DomEvent.disableClickPropagation(div); L.DomEvent.disableScrollPropagation(div);
    div.querySelector("input").addEventListener("input", e => { map.getPane("tilePane").style.opacity = e.target.value; });
    return div;
  },
});
new Dimmer({position: "bottomleft"}).addTo(map);

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


def page(title, menu, data, script):
    """The complete HTML page: the menu (HTML), the data embedded as JSON (DATA) and the script drawing it."""
    payload = json.dumps(data, separators = (",", ":"), ensure_ascii = False).replace("</", "<\\/")
    return (HEAD.replace("__TITLE__", title)
            + menu
            + COMMON_JS.replace("__DATA__", payload).replace("__TILES__", SWISSTOPO_TILES)
                       .replace("__COLOR_TILES__", SWISSTOPO_COLOR_TILES)
            + script
            + "</script></body></html>\n")
