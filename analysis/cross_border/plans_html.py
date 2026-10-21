"""
Interactive HTML view of the border-crossing plans, written by
analysis.cross_border.plans.

A single self-contained file (Leaflet from a CDN, data embedded): the sampled
agents move along their plans as the clock runs, from activity to activity in a
straight line over each leg's departure and travel time. Agents can be filtered by
residence (Swiss / foreign) and by mode, and clicking one shows its plan.
"""

import json

import geopandas as gpd
import numpy as np
import pandas as pd

CRS = "EPSG:2056"

# Speed assumed for a leg whose travel time is missing (m/s, beeline).
FALLBACK_SPEED = 10.0

GROUP_LABELS = {"swiss": "Swiss residents", "foreign": "Foreign residents", "unknown": "Unknown residence"}


def residence_group(od):
    """ crossBorderOD is "CH-FR" for Swiss residents and "FR-CH-FR" for foreign ones. """
    if not isinstance(od, str):
        return "unknown"
    parts = od.split("-")
    if len(parts) == 2 and parts[0] == "CH":
        return "swiss"
    if len(parts) == 3:
        return "foreign"
    return "unknown"


def sample_by_group(df_persons, sample_size, rng):
    """ Up to sample_size agents, split evenly between the residence groups. """
    groups = df_persons["group"].unique()
    quota = sample_size // max(len(groups), 1)
    sampled = []

    for group in groups:
        ids = df_persons.loc[df_persons["group"] == group, "person_id"].values
        sampled.extend(rng.choice(ids, size = min(quota, len(ids)), replace = False))

    remaining = np.setdiff1d(df_persons["person_id"].values, sampled)
    missing = min(sample_size - len(sampled), len(remaining))
    if missing > 0:
        sampled.extend(rng.choice(remaining, size = missing, replace = False))

    return np.array(sampled)


def build_agents(df_persons, df_activities, df_legs):
    activities = df_activities.copy()
    points = gpd.GeoSeries(gpd.points_from_xy(activities["x"], activities["y"]), crs = CRS).to_crs("EPSG:4326")
    activities["lon"] = np.round(points.x.values, 5)
    activities["lat"] = np.round(points.y.values, 5)

    activities_by_person = {k: v for k, v in activities.groupby("person_id")}
    legs_by_person = {k: v for k, v in df_legs.groupby("person_id")} if len(df_legs) > 0 else {}

    agents = []

    for person in df_persons.itertuples(index = False):
        acts = activities_by_person.get(person.person_id)
        if acts is None or len(acts) == 0:
            continue

        acts = acts.sort_values("activity_index")
        legs = legs_by_person.get(person.person_id)
        legs = legs.set_index("leg_index") if legs is not None else pd.DataFrame()

        out_acts, out_legs = [], []
        for index, act in enumerate(acts.itertuples(index = False)):
            out_acts.append([act.lon, act.lat, act.purpose,
                             None if np.isnan(act.start_time) else int(act.start_time),
                             None if np.isnan(act.end_time) else int(act.end_time)])

            if index == len(acts) - 1:
                break

            following = acts.iloc[index + 1]

            # Departure: the leg's, else the end of the activity; arrival: departure plus
            # travel time, else the start of the next activity, else a beeline guess.
            leg = legs.loc[index] if index in legs.index else None
            departure = leg["departure_time"] if leg is not None else np.nan
            if np.isnan(departure):
                departure = act.end_time
            if np.isnan(departure):
                departure = out_legs[-1][2] if out_legs else 0.0

            travel_time = leg["travel_time"] if leg is not None else np.nan
            if np.isnan(travel_time) or travel_time <= 0:
                if not np.isnan(following["start_time"]) and following["start_time"] > departure:
                    travel_time = following["start_time"] - departure
                else:
                    travel_time = np.hypot(following["x"] - act.x, following["y"] - act.y) / FALLBACK_SPEED

            mode = leg["mode"] if leg is not None and isinstance(leg["mode"], str) else "unknown"
            out_legs.append([mode, int(departure), int(departure + max(travel_time, 60))])

        agents.append(dict(
            id = str(person.person_id),
            group = person.group,
            od = person.od if isinstance(person.od, str) else "",
            acts = out_acts,
            legs = out_legs,
        ))

    return agents


def write_html(df_persons, df_activities, df_legs, swiss_border, total_persons, mode_colors, other_color, output_file):
    df_persons = df_persons.copy()
    if "crossBorderOD" not in df_persons.columns:
        df_persons["crossBorderOD"] = None
    df_persons["od"] = df_persons["crossBorderOD"]
    df_persons["group"] = df_persons["od"].map(residence_group)

    agents = build_agents(df_persons, df_activities, df_legs)

    border = gpd.GeoSeries(swiss_border)
    if border.crs is None:
        border = border.set_crs(CRS)
    border = border.boundary.simplify(100).to_crs("EPSG:4326")

    modes = sorted({leg[0] for agent in agents for leg in agent["legs"]})
    data = dict(
        agents = agents,
        total = int(total_persons),
        groups = GROUP_LABELS,
        modes = {mode: mode_colors.get(mode, other_color) for mode in modes},
        border = json.loads(border.to_json()),
    )

    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators = (",", ":")).replace("</", "<\\/"))

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(html)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cross-border plans</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {
    --bg: #ffffff; --fg: #1f2328; --muted: #5f6b7a; --line: #d8dee4; --soft: #f4f6f8;
    --accent: #1D4E89; --swiss: #D1495B; --foreign: #1D4E89; --unknown: #7f7f7f;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); background: var(--bg); }
  #map { position: absolute; inset: 0 0 0 340px; }
  #panel { position: absolute; top: 0; bottom: 0; left: 0; width: 340px; overflow-y: auto; padding: 16px; border-right: 1px solid var(--line); background: var(--bg); }
  h1 { font-size: 17px; margin: 0 0 2px; }
  .sub { color: var(--muted); margin-bottom: 14px; }
  section { border-top: 1px solid var(--line); padding: 12px 0; }
  section h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 0 0 8px; }
  #clock { font-size: 34px; font-variant-numeric: tabular-nums; font-weight: 600; letter-spacing: .02em; }
  .row { display: flex; align-items: center; gap: 8px; }
  input[type=range] { width: 100%; margin: 8px 0; }
  button, select { font: inherit; padding: 5px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--soft); color: var(--fg); cursor: pointer; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); min-width: 72px; }
  label.check { display: flex; align-items: center; gap: 8px; padding: 3px 0; cursor: pointer; }
  .swatch { width: 12px; height: 12px; border-radius: 50%; flex: none; }
  .ring { width: 12px; height: 12px; border-radius: 50%; border: 3px solid; flex: none; background: #fff; }
  .count { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  #search { width: 100%; padding: 5px 8px; border: 1px solid var(--line); border-radius: 6px; font: inherit; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  td { padding: 4px 4px; border-bottom: 1px solid var(--soft); vertical-align: top; }
  td.time { font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; width: 1%; }
  tr.leg td { color: var(--muted); font-size: 12px; }
  tr.current td { background: #fff4d6; }
  .hint { color: var(--muted); font-size: 12px; }
  .legend-note { color: var(--muted); font-size: 12px; margin-top: 6px; }
  #toggle { display: none; }
  @media (max-width: 720px) {
    #panel { width: 100%; bottom: auto; max-height: 55%; border-right: none; border-bottom: 1px solid var(--line); }
    #map { inset: 55% 0 0 0; }
    body.collapsed #panel { max-height: 56px; overflow: hidden; }
    body.collapsed #map { inset: 56px 0 0 0; }
    #toggle { display: inline-block; position: absolute; right: 16px; top: 12px; }
  }
</style>
</head>
<body>
<div id="panel">
  <button id="toggle" aria-label="Toggle panel">☰</button>
  <h1>Cross-border plans</h1>
  <div class="sub" id="subtitle"></div>

  <section>
    <h2>Time</h2>
    <div id="clock">07:00</div>
    <input id="slider" type="range" min="0" max="108000" step="60" value="25200" aria-label="Time of day">
    <div class="row">
      <button class="primary" id="play">Play</button>
      <select id="speed" aria-label="Playback speed">
        <option value="60">1 min / s</option>
        <option value="300" selected>5 min / s</option>
        <option value="900">15 min / s</option>
        <option value="1800">30 min / s</option>
      </select>
    </div>
  </section>

  <section>
    <h2>Residence <span class="hint">(dot outline)</span></h2>
    <div id="groups"></div>
  </section>

  <section>
    <h2>Mode <span class="hint">(dot fill while travelling · on the move now)</span></h2>
    <div id="modes"></div>
    <div class="legend-note">Grey fill: at an activity. Straight lines between activities, not routes.</div>
  </section>

  <section>
    <h2>Display</h2>
    <label class="check"><input type="checkbox" id="showPlans" checked> Plan lines of the shown agents</label>
    <label class="check"><input type="checkbox" id="onlyMoving"> Only agents on the move</label>
  </section>

  <section>
    <h2>Agent</h2>
    <input id="search" placeholder="Person id, or click an agent on the map" aria-label="Person id">
    <div id="agent" class="hint" style="margin-top:8px">No agent selected.</div>
  </section>
</div>
<div id="map"></div>

<script>
const DATA = __DATA__;
const GROUP_COLORS = { swiss: "#D1495B", foreign: "#1D4E89", unknown: "#7f7f7f" };
const IDLE_FILL = "#c9ced4";

const fmt = s => {
  if (s === null || s === undefined) return "";
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0");
};

// Map
const map = L.map("map", { preferCanvas: true, zoomControl: true });
L.tileLayer("https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg",
  { maxZoom: 18, attribution: "© swisstopo" }).addTo(map);
const border = L.geoJSON(DATA.border, { style: { color: "#4A4A4A", weight: 1.2, opacity: 0.8 }, interactive: false }).addTo(map);
map.fitBounds(border.getBounds(), { padding: [20, 20] });

const renderer = L.canvas({ padding: 0.3 });
const planLayer = L.layerGroup().addTo(map);
const dotLayer = L.layerGroup().addTo(map);
const selectionLayer = L.layerGroup().addTo(map);

document.getElementById("subtitle").textContent =
  DATA.agents.length + " agents shown, sampled from " + DATA.total + " border-crossing agents";

// State
const state = { t: 25200, playing: false, groups: {}, modes: {}, selected: null };
Object.keys(DATA.groups).forEach(g => state.groups[g] = true);
Object.keys(DATA.modes).forEach(m => state.modes[m] = true);

// Agents
const agents = DATA.agents.map(a => {
  const agentModes = new Set(a.legs.map(l => l[0]));
  const lines = a.legs.map((l, i) => L.polyline(
    [[a.acts[i][1], a.acts[i][0]], [a.acts[i + 1][1], a.acts[i + 1][0]]],
    { renderer, color: DATA.modes[l[0]], weight: 1, opacity: 0.25, interactive: true }));
  const dot = L.circleMarker([a.acts[0][1], a.acts[0][0]],
    { renderer, radius: 4, weight: 2, color: GROUP_COLORS[a.group], fillColor: IDLE_FILL, fillOpacity: 1 });
  const agent = { ...a, agentModes, lines, dot, visible: false };
  dot.on("click", () => select(agent));
  lines.forEach(line => line.on("click", () => select(agent)));
  return agent;
});
const byId = new Map(agents.map(a => [a.id, a]));

// Position of an agent at time t: [lat, lon, mode or null, leg or activity index]
function position(a, t) {
  for (let i = 0; i < a.legs.length; i++) {
    const [mode, dep, arr] = a.legs[i];
    if (t < dep) return [a.acts[i][1], a.acts[i][0], null, i];
    if (t < arr) {
      const f = (t - dep) / (arr - dep), o = a.acts[i], d = a.acts[i + 1];
      return [o[1] + f * (d[1] - o[1]), o[0] + f * (d[0] - o[0]), mode, i];
    }
  }
  const last = a.acts[a.acts.length - 1];
  return [last[1], last[0], null, a.acts.length - 1];
}

function isFiltered(a) {
  if (!state.groups[a.group]) return true;
  for (const m of a.agentModes) if (state.modes[m]) return false;
  return true;
}

function render() {
  const t = state.t;
  document.getElementById("clock").textContent = fmt(t);
  document.getElementById("slider").value = t;

  const onlyMoving = document.getElementById("onlyMoving").checked;
  const showPlans = document.getElementById("showPlans").checked;
  const moving = {};

  for (const a of agents) {
    const [lat, lon, mode] = position(a, t);
    const show = !isFiltered(a) && (!onlyMoving || mode !== null);
    if (mode !== null && !isFiltered(a)) moving[mode] = (moving[mode] || 0) + 1;

    if (show) {
      a.dot.setLatLng([lat, lon]);
      a.dot.setStyle({ fillColor: mode ? DATA.modes[mode] : IDLE_FILL, radius: mode ? 5 : 3.5 });
      if (!a.visible) { dotLayer.addLayer(a.dot); if (showPlans) a.lines.forEach(l => planLayer.addLayer(l)); a.visible = true; }
    } else if (a.visible) {
      dotLayer.removeLayer(a.dot); a.lines.forEach(l => planLayer.removeLayer(l)); a.visible = false;
    }
  }

  for (const m of Object.keys(DATA.modes)) {
    const el = document.getElementById("moving-" + m);
    if (el) el.textContent = moving[m] || 0;
  }
  if (state.selected) renderItinerary();
}

function refreshPlans() {
  planLayer.clearLayers();
  const showPlans = document.getElementById("showPlans").checked;
  for (const a of agents) if (a.visible && showPlans) a.lines.forEach(l => planLayer.addLayer(l));
}

// Filters
function checkbox(container, key, label, swatch, count, onChange, extraId) {
  const row = document.createElement("label");
  row.className = "check";
  row.innerHTML = `<input type="checkbox" checked> ${swatch} <span>${label}</span><span class="count"${extraId ? ` id="${extraId}"` : ""}>${count}</span>`;
  row.querySelector("input").addEventListener("change", e => onChange(key, e.target.checked));
  container.appendChild(row);
}

const groupCounts = {};
agents.forEach(a => groupCounts[a.group] = (groupCounts[a.group] || 0) + 1);
for (const [g, label] of Object.entries(DATA.groups)) {
  if (!groupCounts[g]) continue;
  checkbox(document.getElementById("groups"), g, label,
    `<span class="ring" style="border-color:${GROUP_COLORS[g]}"></span>`, groupCounts[g],
    (k, v) => { state.groups[k] = v; render(); });
}
for (const [m, color] of Object.entries(DATA.modes)) {
  checkbox(document.getElementById("modes"), m, m,
    `<span class="swatch" style="background:${color}"></span>`, 0,
    (k, v) => { state.modes[k] = v; render(); }, "moving-" + m);
}

// Selection
function select(a) {
  state.selected = a;
  selectionLayer.clearLayers();
  a.legs.forEach((l, i) => L.polyline(
    [[a.acts[i][1], a.acts[i][0]], [a.acts[i + 1][1], a.acts[i + 1][0]]],
    { color: DATA.modes[l[0]], weight: 4, opacity: 0.95, interactive: false }).addTo(selectionLayer));
  a.acts.forEach((act, i) => L.circleMarker([act[1], act[0]],
    { radius: 6, weight: 2, color: "#222", fillColor: act[2] === "border" ? "#D1495B" : "#fff", fillOpacity: 1 })
    .bindTooltip(`${i + 1}. ${act[2]} ${fmt(act[3])}–${fmt(act[4])}`).addTo(selectionLayer));
  document.getElementById("search").value = a.id;
  renderItinerary();
}

function renderItinerary() {
  const a = state.selected, [, , , index] = position(a, state.t);
  const [, , mode] = position(a, state.t);
  let rows = "";
  a.acts.forEach((act, i) => {
    const here = mode === null && index === i;
    rows += `<tr class="${here ? "current" : ""}"><td class="time">${fmt(act[3]) || "start"}</td><td><b>${act[2]}</b>${act[4] !== null ? " until " + fmt(act[4]) : ""}</td></tr>`;
    if (i < a.legs.length) {
      const [m, dep, arr] = a.legs[i];
      const on = mode !== null && index === i;
      rows += `<tr class="leg ${on ? "current" : ""}"><td class="time">${fmt(dep)}</td><td><span class="swatch" style="display:inline-block;vertical-align:middle;background:${DATA.modes[m]}"></span> ${m}, ${Math.round((arr - dep) / 60)} min</td></tr>`;
    }
  });
  document.getElementById("agent").innerHTML =
    `<div><b>${a.id}</b> · ${DATA.groups[a.group]}${a.od ? " · " + a.od : ""}</div>
     <table>${rows}</table>
     <div style="margin-top:8px" class="row"><button id="zoom">Zoom to plan</button><button id="clear">Clear</button></div>`;
  document.getElementById("zoom").onclick = () => map.fitBounds(L.latLngBounds(a.acts.map(x => [x[1], x[0]])), { padding: [40, 40], maxZoom: 13 });
  document.getElementById("clear").onclick = () => { state.selected = null; selectionLayer.clearLayers();
    document.getElementById("agent").textContent = "No agent selected."; document.getElementById("search").value = ""; };
}

document.getElementById("search").addEventListener("change", e => {
  const a = byId.get(e.target.value.trim());
  if (a) { select(a); map.fitBounds(L.latLngBounds(a.acts.map(x => [x[1], x[0]])), { padding: [40, 40], maxZoom: 13 }); }
});

// Controls
document.getElementById("slider").addEventListener("input", e => { state.t = +e.target.value; render(); });
document.getElementById("showPlans").addEventListener("change", refreshPlans);
document.getElementById("onlyMoving").addEventListener("change", render);
document.getElementById("toggle").addEventListener("click", () => document.body.classList.toggle("collapsed"));

let last = null;
function tick(now) {
  if (!state.playing) { last = null; return; }
  if (last !== null) {
    state.t += (now - last) / 1000 * +document.getElementById("speed").value;
    if (state.t > 108000) state.t = 0;
  }
  last = now;
  render();
  requestAnimationFrame(tick);
}
document.getElementById("play").addEventListener("click", e => {
  state.playing = !state.playing;
  e.target.textContent = state.playing ? "Pause" : "Play";
  if (state.playing) requestAnimationFrame(tick);
});

render();
</script>
</body>
</html>
"""
