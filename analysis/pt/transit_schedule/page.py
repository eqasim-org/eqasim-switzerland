"""HTML page of analysis.pt.transit_schedule.visualize: the menu on the left third (search, layers, the timetable of
a stop or the details of a line), the map on the rest. Same layout and helpers as the GTFS maps of
data/gtfs/visualize_pages.py; the data is embedded once as JSON and the timetables are computed in the browser."""
import json

from data.gtfs.visualize_pages import COMMON_JS, HEAD, SWISSTOPO_TILES


def page(title, data):
    payload = json.dumps(data, separators = (",", ":"), ensure_ascii = False).replace("</", "<\\/")
    return (HEAD.replace("__TITLE__", title)
            + MENU
            + COMMON_JS.replace("__DATA__", payload).replace("__TILES__", SWISSTOPO_TILES)
            + SCRIPT
            + "</script></body></html>\n")


MENU = r"""
  <div id="menu">
    <div id="top">
      <h1>Transit schedule</h1>
      <div class="muted" id="subtitle"></div>
      <div class="row"><input id="search" type="text" placeholder="Search a stop or a line (e.g. Cornavin, 12, L2)"></div>
      <div class="list" id="results"></div>
      <div class="row" id="check"></div>
      <details class="row" open><summary>Layers</summary><div class="legend" id="legend"></div></details>
    </div>
    <div id="panel"><div class="muted">Click on or near a stop on the map, or search it, to see its timetable. Search a line
      to highlight its route, its stops and its official route.</div></div>
  </div>
"""

SCRIPT = r"""
const NAMES = DATA.names, STOPS = DATA.stops, ROUTES = DATA.routes, REPS = DATA.reps, LINKS = DATA.links;
const MODES = DATA.modes, MODE_COLORS = DATA.modeColors, C = DATA.colors;
const modeColor = m => MODE_COLORS[MODES[m]] || "#888";
const lineName = r => NAMES[ROUTES[r][0]];
const byNumber = (a, b) => a.localeCompare(b, "en", {numeric: true});

// hh:mm of seconds since midnight, with the day after marked (MATSim times run past 24:00)
const clock = s => { const m = Math.round(s / 60), h = Math.floor(m / 60); return hhmm(((h % 24) * 60) + m % 60) + (h >= 24 ? "⁺¹" : ""); };

// The calls of every route at every stop: STOP_CALLS[stop] = [[route, position], ...]
const STOP_CALLS = STOPS.map(() => []);
ROUTES.forEach((route, r) => route[5].forEach(([stop], position) => { if (stop >= 0) STOP_CALLS[stop].push([r, position]); }));
const STOP_LINES = STOP_CALLS.map(calls => [...new Set(calls.map(([r]) => lineName(r)))].sort(byNumber));

const nDepartures = sum(ROUTES.map(r => r[4].length));
byId("subtitle").textContent = `${STOPS.length.toLocaleString("en")} stops, ${new Set(ROUTES.map(r => r[0])).size.toLocaleString("en")} lines, ` +
  `${ROUTES.length.toLocaleString("en")} routes` + (DATA.schedules ? `, ${nDepartures.toLocaleString("en")} departures.` : " (timetables not exported).");

// ---------------------------------------------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------------------------------------------

const visibleModes = new Set(DATA.defaultModes);
const show = {stops: true, busLanes: DATA.busLanes.length > 0};
const linkLayer = L.layerGroup().addTo(map), laneLayer = L.layerGroup().addTo(map), stopLayer = L.layerGroup().addTo(map);
const selectionLayer = L.layerGroup().addTo(map);

LINKS.forEach(link => link.bounds = L.latLngBounds([link[0], link[1]], [link[2], link[3]]));

function bearing(a, b) {
  const [la1, lo1, la2, lo2] = [a[0], a[1], b[0], b[1]].map(v => v * Math.PI / 180);
  const x = Math.sin(lo2 - lo1) * Math.cos(la2), y = Math.cos(la1) * Math.sin(la2) - Math.sin(la1) * Math.cos(la2) * Math.cos(lo2 - lo1);
  return (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;
}

const stopRadius = () => { const z = map.getZoom(); return z >= 16 ? 7 : z >= 14 ? 5.5 : z >= 12 ? 4 : 2.5; };

function draw() {
  linkLayer.clearLayers(); laneLayer.clearLayers(); stopLayer.clearLayers();
  const view = map.getBounds().pad(0.2), zoom = map.getZoom();

  // the links used by the routes, one segment per link: merged per mode when zoomed out, with the lines of every
  // link in its tooltip when zoomed in
  const merged = {};
  LINKS.forEach(link => {
    if (!visibleModes.has(link[4]) || !view.intersects(link.bounds)) return;
    const segment = [[link[0], link[1]], [link[2], link[3]]];
    if (zoom < 13) { (merged[link[4]] = merged[link[4]] || []).push(segment); return; }
    L.polyline(segment, {color: modeColor(link[4]), weight: 2.5, opacity: 0.65})
      .bindTooltip(`<b>${esc(MODES[link[4]])}</b>: ${esc(NAMES[link[5]])}`, {sticky: true}).addTo(linkLayer);
  });
  Object.entries(merged).forEach(([m, segments]) => L.polyline(segments, {color: modeColor(+m), weight: 2, opacity: 0.6, interactive: false}).addTo(linkLayer));

  if (show.busLanes) DATA.busLanes.forEach(lane => {
    const a = [lane[0], lane[1]], b = [lane[2], lane[3]];
    if (!view.contains(a) && !view.contains(b)) return;
    L.polyline([a, b], {color: C.busLane, weight: 3, opacity: 0.85, dashArray: "6 4"}).bindTooltip("Reserved bus/PT lane").addTo(laneLayer);
    if (zoom >= 15) L.marker([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], {interactive: false, icon: L.divIcon({className: "", iconSize: [12, 12],
      html: `<div style="transform:rotate(${bearing(a, b) - 90}deg);color:${C.busLane};font:bold 14px/12px sans-serif;text-align:center">&#9658;</div>`})}).addTo(laneLayer);
  });

  if (show.stops && (zoom >= 11 || STOPS.length < 3000)) {
    const radius = stopRadius();
    STOPS.forEach(([name, la, lo, modes], k) => {
      if (!view.contains([la, lo])) return;
      L.circleMarker([la, lo], {radius, color: "#0b0b0b", weight: 1, fillColor: modeColor(modes[0]), fillOpacity: 0.95})
        .bindTooltip(`<b>${esc(NAMES[name])}</b><br>${esc(STOP_LINES[k].slice(0, 12).join(", "))}${STOP_LINES[k].length > 12 ? " …" : ""}`).addTo(stopLayer);
    });
  }
  drawSelection();
}

// a click anywhere selects the closest stop within 25 pixels: no need to hit the marker itself
map.on("click", event => {
  const p = map.latLngToContainerPoint(event.latlng), view = map.getBounds().pad(0.05);
  let best = null, bestDistance = 25;
  STOPS.forEach(([, la, lo], k) => {
    if (!view.contains([la, lo])) return;
    const d = p.distanceTo(map.latLngToContainerPoint([la, lo]));
    if (d < bestDistance) { best = k; bestDistance = d; }
  });
  if (best != null) showStop(best, false);
});

function legend() {
  const counts = {};
  LINKS.forEach(link => counts[link[4]] = (counts[link[4]] || 0) + 1);
  let html = MODES.map((m, k) => counts[k] ? `<div><label><input type="checkbox" data-mode="${k}" ${visibleModes.has(k) ? "checked" : ""}> ` +
    `<span class="swatch" style="background:${modeColor(k)}"></span>${esc(m)} <span class="muted">(${fmt(counts[k])} links)</span></label></div>` : "").join("");
  if (DATA.busLanes.length) html += `<div><label><input type="checkbox" data-show="busLanes" ${show.busLanes ? "checked" : ""}> <span class="swatch" ` +
    `style="background:${C.busLane}"></span>Reserved bus/PT lanes <span class="muted">(${fmt(DATA.busLanes.length)}, arrows zoomed in)</span></label></div>`;
  html += `<div><label><input type="checkbox" data-show="stops" ${show.stops ? "checked" : ""}> <span class="dot" style="background:#fff;border:1px solid #0b0b0b"></span>` +
    `Stops, coloured by mode <span class="muted">(${fmt(STOPS.length)})</span></label></div>`;
  html += `<div class="muted" style="margin-top:6px"><span class="swatch" style="background:${C.selected}"></span>selected route ` +
    (Object.keys(DATA.references).length ? `&nbsp; <span class="swatch" style="background:${C.reference};opacity:.6"></span>official line route` : "") + "</div>";
  byId("legend").innerHTML = html;
  byId("legend").querySelectorAll("input[data-mode]").forEach(el => el.addEventListener("change", () => {
    el.checked ? visibleModes.add(+el.dataset.mode) : visibleModes.delete(+el.dataset.mode); draw(); }));
  byId("legend").querySelectorAll("input[data-show]").forEach(el => el.addEventListener("change", () => { show[el.dataset.show] = el.checked; draw(); }));
}

// ---------------------------------------------------------------------------------------------------------------
// Selection: a route (representative of a line, direction and branch) and/or a stop
// ---------------------------------------------------------------------------------------------------------------

let selectedRep = null, selectedStop = null;

function checkSummary(check) {
  let text = Math.round(check.share_inside * 100) + "% on the official route";
  if (check.n_uturns) text += ", " + check.n_uturns + " U-turn(s)";
  if (!check.reference_matched) text += " (no official route towards this terminus: compared with all the routes of the line)";
  return text;
}
const checkBadge = check => check ? ` <b style="color:${check.status === "ok" ? C.ok : C.deviates}" title="${esc(checkSummary(check))}">` +
  `${Math.round(check.share_inside * 100)}%${check.n_uturns ? " ↺" + check.n_uturns : ""}</b>` : "";

function drawSelection() {
  selectionLayer.clearLayers();
  if (selectedRep != null) {
    const rep = REPS[selectedRep];
    rep.references.forEach(id => (DATA.references[id] || []).forEach(path =>
      L.polyline(path, {color: C.reference, weight: 11, opacity: 0.55}).bindTooltip("Official line route").addTo(selectionLayer)));
    const line = L.polyline(rep.path, {color: C.selected, weight: 6, opacity: 1}).addTo(selectionLayer);
    line.bindTooltip(`<b>${esc(rep.label)}</b>` + (rep.check ? "<br>" + esc(checkSummary(rep.check)) : ""), {sticky: true});
    rep.platforms.forEach(p => L.circleMarker(p, {radius: 6, color: C.selected, weight: 2, fillColor: "#fff", fillOpacity: 1, interactive: false}).addTo(selectionLayer));
  }
  if (selectedStop != null) {
    const [, la, lo] = STOPS[selectedStop];
    L.circleMarker([la, lo], {radius: stopRadius() + 6, color: "#0b0b0b", weight: 3, fill: false, interactive: false}).addTo(selectionLayer);
  }
}

function selectRep(i, fit) {
  selectedRep = i;
  drawSelection();
  if (fit && i != null) map.fitBounds(L.latLngBounds(REPS[i].path), {maxZoom: 16, padding: [40, 40]});
}

// the routes a representative stands for (same line, terminus and branch)
const REP_ROUTES = REPS.map(() => []);
ROUTES.forEach((route, r) => { if (route[6] >= 0) REP_ROUTES[route[6]].push(r); });

function showLine(i) {
  selectRep(i, true);
  const rep = REPS[i], routes = REP_ROUTES[i], color = modeColor(rep.mode);
  const hours = Array(24).fill(0);
  routes.forEach(r => ROUTES[r][4].forEach(t => hours[Math.floor(t / 3600) % 24]++));
  const stops = rep.route != null ? ROUTES[rep.route][5] : [];
  let html = `<h2><span class="swatch" style="background:${color}"></span>${esc(rep.label)}</h2>`;
  html += `<button class="link" id="clearLine">clear the selection</button>`;
  if (rep.check) html += `<div class="row">Route check: <b style="color:${rep.check.status === "ok" ? C.ok : C.deviates}">${esc(checkSummary(rep.check))}</b></div>`;
  html += `<div class="muted">${fmt(sum(hours))} departures, ${stops.length} stops` + (routes.length > 1 ? `, ${routes.length} route variants` : "") + "</div>";
  if (DATA.schedules) html += hourBars("Departures per hour (first stop)", hours, color);
  html += `<table class="t"><tr><th class="n">min</th><th>stop</th></tr>` + stops.map(([stop, offset]) =>
    `<tr><td class="n">${Math.round((offset - stops[0][1]) / 60)}</td><td>` + (stop >= 0 ? `<button class="link" data-stop="${stop}">${esc(NAMES[STOPS[stop][0]])}</button>` :
    `<span class="muted">outside the map</span>`) + "</td></tr>").join("") + "</table>";
  byId("panel").innerHTML = html;
  byId("clearLine").addEventListener("click", () => { selectRep(null); byId("panel").innerHTML = ""; });
  byId("panel").querySelectorAll("button[data-stop]").forEach(el => el.addEventListener("click", () => showStop(+el.dataset.stop, true)));
}

// The timetable of a stop: the calls of every route, grouped by line, direction and terminus
function showStop(k, center) {
  selectedStop = k;
  const [name, la, lo] = STOPS[k];
  if (center) map.setView([la, lo], Math.max(map.getZoom(), 15));

  const services = {};
  STOP_CALLS[k].forEach(([r, position]) => {
    const route = ROUTES[r], calls = route[5], terminal = position === calls.length - 1 && calls.length > 1;
    const place = terminal ? route[2] : route[3];
    const key = `${lineName(r)}|${terminal ? 1 : 0}|${place}`;
    const service = services[key] = services[key] || {line: lineName(r), mode: route[1], terminal, place, times: [], reps: {}};
    route[4].forEach(t => service.times.push([t + calls[position][1], r, t]));
    if (route[6] >= 0) service.reps[route[6]] = (service.reps[route[6]] || 0) + Math.max(route[4].length, 1);
  });
  const list = Object.values(services).sort((a, b) => byNumber(a.line, b.line) || a.terminal - b.terminal || b.times.length - a.times.length);

  let html = `<h2>${esc(NAMES[name])}</h2><div class="muted">${list.length} line(s) and direction(s)` + (DATA.schedules ? `, ${fmt(sum(list.map(s => s.times.length)))} departures` : "") + "</div>";
  list.forEach((service, n) => {
    const color = modeColor(service.mode), hours = Array(24).fill(0);
    service.times.sort((a, b) => a[0] - b[0]).forEach(([t]) => hours[Math.floor(t / 3600) % 24]++);
    const rep = Object.entries(service.reps).sort((a, b) => b[1] - a[1])[0];
    html += `<hr class="divider"><h2><span class="swatch" style="background:${color}"></span>${esc(service.line)} ${service.terminal ? "← from " : "→ "}${esc(NAMES[service.place])}</h2>`;
    if (rep) html += `<button class="link" data-rep="${rep[0]}">show the route</button>${checkBadge(REPS[rep[0]].check)} &nbsp; <button class="link" data-line="${rep[0]}">line details</button>`;
    if (!DATA.schedules) return;
    html += hourBars(`${service.terminal ? "Arrivals" : "Departures"} per hour (${service.times.length})`, hours, color);
    html += `<details><summary>${service.terminal ? "Arrival" : "Departure"} times</summary><table class="t">` + service.times.map(([t, r, t0]) => {
      const route = ROUTES[r], calls = route[5];
      return `<tr><td>${clock(t)}</td><td class="muted">${esc(NAMES[route[2]])} ${clock(t0 + calls[0][1])} → ${esc(NAMES[route[3]])} ${clock(t0 + calls[calls.length - 1][1])}</td></tr>`;
    }).join("") + "</table></details>";
  });
  byId("panel").innerHTML = html;
  byId("panel").querySelectorAll("button[data-rep]").forEach(el => el.addEventListener("click", () => selectRep(+el.dataset.rep, true)));
  byId("panel").querySelectorAll("button[data-line]").forEach(el => el.addEventListener("click", () => showLine(+el.dataset.line)));
  drawSelection();
}

// ---------------------------------------------------------------------------------------------------------------
// Search and route check filter
// ---------------------------------------------------------------------------------------------------------------

const onlyDeviating = () => byId("deviating") && byId("deviating").checked;

function search() {
  const text = byId("search").value.trim().toLowerCase();
  if (!text && !onlyDeviating()) { byId("results").innerHTML = ""; return; }
  const lines = REPS.map((rep, i) => [rep, i])
    .filter(([rep]) => (!text || NAMES[rep.line].toLowerCase().startsWith(text) || rep.label.toLowerCase().includes(text)) &&
                       (!onlyDeviating() || (rep.check && rep.check.status !== "ok")))
    .sort((a, b) => byNumber(a[0].label, b[0].label)).slice(0, 200);
  const stops = text && !onlyDeviating() ? STOPS.map((s, k) => [s, k]).filter(([s]) => NAMES[s[0]].toLowerCase().includes(text)).slice(0, 100) : [];
  byId("results").innerHTML =
    stops.map(([s, k]) => `<div data-stop="${k}"><span class="dot" style="background:${modeColor(s[3][0])}"></span>${esc(NAMES[s[0]])} <span class="muted">${STOP_LINES[k].length} line(s)</span></div>`).join("") +
    lines.map(([rep, i]) => `<div data-rep="${i}"><span class="swatch" style="background:${modeColor(rep.mode)}"></span>${esc(rep.label)}${checkBadge(rep.check)}</div>`).join("") ||
    `<div class="muted">no stop or line</div>`;
  byId("results").querySelectorAll("div[data-stop]").forEach(el => el.addEventListener("click", () => showStop(+el.dataset.stop, true)));
  byId("results").querySelectorAll("div[data-rep]").forEach(el => el.addEventListener("click", () => showLine(+el.dataset.rep)));
}

if (DATA.routeCheck) {
  byId("check").innerHTML = `<div class="muted">Route check: ${DATA.routeCheck.deviating} of ${DATA.routeCheck.total} route(s) deviate from the official route ` +
    `(share of the length in its corridor, U-turns; details in route_check.csv)</div><label><input type="checkbox" id="deviating"> list the deviating routes only</label>`;
  byId("deviating").addEventListener("change", search);
}

byId("search").addEventListener("input", search);
map.on("moveend", draw);
legend();
if (STOPS.length) map.fitBounds(L.latLngBounds(STOPS.map(s => [s[1], s[2]])).pad(0.05));
draw();
"""
