"""
Compact deck.gl page of the cross-border flow maps (analysis.counts.cross_border_flow_cars / _pt). As in
analysis.pt.transit_schedule.visualize, the data is embedded once as compact JSON - coordinates rounded to ~1 m,
numbers rounded, repeated texts (stop names, lines) stored once in a table - and the colors, widths, shares and
tooltips of the selected metric are computed in the browser instead of being written out for every feature and metric.
"""

import json

import numpy as np
import pandas as pd

COORDINATE_DECIMALS = 5  # ~1 m


def _round_coords(coords):
    return [round(float(c), COORDINATE_DECIMALS) for xy in coords for c in xy[:2]]


def _quantile_max(values):
    values = np.asarray(values, dtype = float)
    positive = values[values > 0]
    return float(max(np.quantile(positive, 0.98), 1.0)) if len(positive) else 1.0


def flow_values(metric_key, values):
    """Values of a metric: a share ("<group>_share_pct") is computed from the flow of its group and the total flow"""
    if metric_key.endswith("_share_pct"):
        flow = np.asarray(values[metric_key[: -len("_share_pct")] + "_flow"], dtype = float)
        total = np.asarray(values["total_flow"], dtype = float)
        return np.where(total > 0, 100.0 * flow / np.maximum(total, 1e-9), 0.0)
    return np.asarray(values[metric_key], dtype = float)


def layer(kind, geometries, values, texts = None, tooltip = (), size_scale = 12.0, size_base = 2.0, metrics = ()):
    """One layer of the page.
    kind: "path" (geometries: lists of (lon, lat)) or "point" (geometries: (lon, lat))
    values: {name: numbers}, the flows (total_flow and the <group>_flow of the metrics) and other numbers to show
    texts: {name: strings}, shown in the tooltip
    tooltip: [(label, name, format)], format "int", "pct" (a share metric) or "text"
    The colors and widths are scaled off the 98th percentile of each metric in this layer, as before."""
    texts = texts or {}
    return dict(
        kind = kind,
        geometries = [_round_coords(g) for g in geometries] if kind == "path"
                     else [_round_coords([g]) for g in geometries],
        values = {k: [None if pd.isna(v) else round(float(v)) for v in vs] for k, vs in values.items()},
        texts = texts,
        tooltip = list(tooltip),
        size_scale = size_scale,
        size_base = size_base,
        maxima = {m["key"]: _quantile_max(flow_values(m["key"], values)) for m in metrics},
        basis_maxima = {m["key"]: _quantile_max(flow_values(m["width_basis"], values)) for m in metrics},
    )


def write_page(output_file, title, layers, metrics, default_metric, outlines = (), bounds = None, legend_lines = ()):
    """outlines: [(list of rings of (lon, lat), [r, g, b], width in pixels)]; bounds: (min lon, min lat, max lon, max lat)"""
    strings, index = [], {}

    def code(text):
        text = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
        if text not in index:
            index[text] = len(strings)
            strings.append(text)
        return index[text]

    data = dict(
        title = title,
        metrics = [dict(key = m["key"], label = m["label"], low = list(m["low"]), high = list(m["high"]),
                        gamma = m["gamma"], basis = m["width_basis"]) for m in metrics],
        defaultMetric = default_metric,
        layers = [dict(layer, texts = {k: [code(t) for t in ts] for k, ts in layer["texts"].items()}) for layer in layers],
        strings = strings,
        outlines = [dict(rings = [_round_coords(r) for r in rings], color = list(color), width = width)
                    for rings, color, width in outlines],
        bounds = [round(float(b), COORDINATE_DECIMALS) for b in bounds] if bounds is not None else None,
        legendLines = [dict(text = text, style = style) for text, style in legend_lines],
    )
    payload = json.dumps(data, separators = (",", ":"), ensure_ascii = False).replace("</", "<\\/")

    with open(output_file, "w", encoding = "utf-8") as f:
        f.write(PAGE.replace("__TITLE__", title).replace("__DATA__", payload))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font: 12px/1.3 Arial, sans-serif; }
  #map { position: absolute; inset: 0; }
  #legend { position: absolute; z-index: 2; bottom: 12px; left: 12px; padding: 10px 12px; border-radius: 7px;
            background: rgba(255,255,255,0.95); box-shadow: 0 2px 10px rgba(0,0,0,0.28); color: #222; max-width: 280px; }
  #bar { width: 180px; height: 12px; border: 1px solid #777; }
  #tooltip { position: absolute; z-index: 3; pointer-events: none; background: #000; color: #fff; padding: 8px 10px;
             border-radius: 4px; display: none; }
  #tooltip td { padding: 0 6px 0 0; }
  #credits { position: absolute; z-index: 2; bottom: 2px; right: 4px; font-size: 10px; color: #555; }
</style></head><body>
<div id="map"></div>
<div id="legend">
  <label style="display:block;margin-bottom:8px;"><div style="margin-bottom:3px;">Show</div><select id="metric" style="width:240px;"></select></label>
  <div id="legend-title" style="font-weight:700;margin-bottom:6px;"></div>
  <div id="bar"></div>
  <div style="display:flex;justify-content:space-between;margin-top:3px;"><span>0</span><span id="legend-max"></span></div>
  <div style="margin-top:4px;color:#777;">Gray = no value for this metric</div>
  <div id="legend-lines"></div>
</div>
<div id="tooltip"></div>
<div id="credits">© swisstopo</div>
<script>
const DATA = __DATA__;
const S = DATA.strings;
const byId = id => document.getElementById(id);

// Same color and width scales as before (flow_metrics.add_metric_columns), computed here
const shareFlow = key => key.endsWith("_share_pct") ? key.slice(0, -"_share_pct".length) + "_flow" : null;
function value(layer, key, i) {
  const flow = shareFlow(key);
  if (flow) { const t = layer.values.total_flow[i]; return t > 0 ? 100 * layer.values[flow][i] / t : 0; }
  return layer.values[key][i];
}
const mix = (low, high, t) => [0, 1, 2].map(k => Math.round(low[k] + (high[k] - low[k]) * t));
function color(layer, metric, i) {
  const v = value(layer, metric.key, i);
  if (v == null || !(v > 0)) return [...metric.low, 60];
  const t = Math.pow(Math.min(v, layer.maxima[metric.key]) / layer.maxima[metric.key], metric.gamma);
  return [...mix(metric.low, metric.high, t), Math.round(80 + 175 * t)];
}
function size(layer, metric, i) {
  const b = value(layer, metric.basis, i), max = layer.basis_maxima[metric.key];
  return b > 0 ? Math.sqrt(Math.min(b, max) / max) * layer.size_scale + layer.size_base : layer.size_base * 0.5;
}
const fmt = (v, kind) => v == null ? "" : kind === "pct" ? v.toFixed(1) + " %" : Math.round(v).toLocaleString("en");
function tooltip(layer, i) {
  return "<table>" + layer.tooltip.map(([label, name, kind]) => {
    const text = kind === "text" ? S[layer.texts[name][i]] : fmt(kind === "pct" ? value(layer, name, i) : layer.values[name][i], kind);
    return text === "" ? "" : `<tr><td><b>${label}</b></td><td>${text}</td></tr>`;
  }).join("") + "</table>";
}

// Layers
const metrics = Object.fromEntries(DATA.metrics.map(m => [m.key, m]));
let current = metrics[DATA.defaultMetric];
const indices = DATA.layers.map(l => Array.from(l.geometries, (_, i) => i));
const basemap = new deck.TileLayer({
  id: "basemap", data: "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg", minZoom: 0, maxZoom: 18, tileSize: 256,
  renderSubLayers: props => {
    const {west, south, east, north} = props.tile.bbox;
    return new deck.BitmapLayer(props, {data: null, image: props.data, bounds: [west, south, east, north]});
  },
});
const outlineLayers = DATA.outlines.map((o, k) => new deck.PathLayer({
  id: "outline-" + k, data: o.rings, getPath: r => r, positionFormat: "XY", getColor: o.color,
  getWidth: o.width, widthUnits: "pixels",
}));
function flowLayers() {
  return DATA.layers.map((layer, k) => layer.kind === "path"
    ? new deck.PathLayer({
        id: "flow-" + k, data: indices[k], pickable: true, autoHighlight: true, highlightColor: [0, 200, 255],
        getPath: i => layer.geometries[i], positionFormat: "XY", widthScale: 3, widthMinPixels: 2,
        getColor: i => color(layer, current, i), getWidth: i => size(layer, current, i),
        updateTriggers: {getColor: current.key, getWidth: current.key},
      })
    : new deck.ScatterplotLayer({
        id: "flow-" + k, data: indices[k], pickable: true, autoHighlight: true, stroked: true,
        getPosition: i => layer.geometries[i], getRadius: i => size(layer, current, i), radiusUnits: "pixels",
        radiusMinPixels: 4, radiusMaxPixels: 20, getFillColor: i => color(layer, current, i),
        getLineColor: [50, 50, 50], lineWidthMinPixels: 1,
        updateTriggers: {getFillColor: current.key, getRadius: current.key},
      }));
}

const [w, s, e, n] = DATA.bounds || [5.9, 45.8, 10.5, 47.8];
const viewport = new deck.WebMercatorViewport({width: window.innerWidth, height: window.innerHeight});
const view = viewport.fitBounds([[w, s], [e, n]], {padding: 20});
const tip = byId("tooltip");
const deckgl = new deck.DeckGL({
  container: "map", controller: true,
  initialViewState: {longitude: view.longitude, latitude: view.latitude, zoom: view.zoom},
  layers: [basemap, ...outlineLayers, ...flowLayers()],
  onHover: info => {
    if (info.object == null || !info.layer.id.startsWith("flow-")) { tip.style.display = "none"; return; }
    tip.innerHTML = tooltip(DATA.layers[+info.layer.id.slice(5)], info.object);
    tip.style.display = "block"; tip.style.left = (info.x + 12) + "px"; tip.style.top = (info.y + 12) + "px";
  },
});

// Legend
const select = byId("metric");
select.innerHTML = DATA.metrics.map(m => `<option value="${m.key}">${m.label}</option>`).join("");
select.value = current.key;
byId("legend-lines").innerHTML = DATA.legendLines.map(l => `<div style="margin-top:2px;${l.style}">${l.text}</div>`).join("");
function updateLegend() {
  const stops = [0, 0.2, 0.4, 0.6, 0.8, 1].map(t => {
    const g = Math.pow(t, current.gamma), c = mix(current.low, current.high, g);
    return `rgba(${c[0]},${c[1]},${c[2]},${((80 + 175 * g) / 255).toFixed(2)}) ${t * 100}%`;
  });
  byId("legend-title").textContent = current.label;
  byId("bar").style.background = `linear-gradient(to right, ${stops.join(", ")})`;
  byId("legend-max").textContent = Math.round(Math.max(...DATA.layers.map(l => l.maxima[current.key]))).toLocaleString("en") + "+";
}
select.addEventListener("change", () => {
  current = metrics[select.value];
  deckgl.setProps({layers: [basemap, ...outlineLayers, ...flowLayers()]});
  updateLegend();
});
updateLegend();
</script></body></html>
"""
