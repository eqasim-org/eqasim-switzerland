"""
Shared "pick a metric from a dropdown, recolor+rewidth a set of pydeck
layers" machinery used by both cross_border_flow_cars (per-link car flow)
and cross_border_flow_pt (per-segment/per-stop PT flow) maps, so the two
maps show a consistent set of rider-group layers (total / non-Swiss
cross-border / Swiss-resident cross-border / France-resident, each as a
flow and a share-of-total-traffic metric) with matching colors and controls.
"""

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def share_column(flow_column):
    """"<name>_flow" -> "<name>_share_pct" - the naming convention used by
    both maps for a rider group's share-of-total-traffic column, derived
    from its flow column."""
    assert flow_column.endswith("_flow")
    return flow_column[: -len("_flow")] + "_share_pct"


def flow_alpha(t, low=80, high=255):
    t = max(0.0, min(1.0, t))
    return int(low + (high - low) * t)


def gray_to_color(t, low_rgb, high_rgb):
    t = max(0.0, min(1.0, t))
    return [int(low_rgb[i] + (high_rgb[i] - low_rgb[i]) * t) for i in range(3)]


def add_metric_columns(df, metric, size_scale, size_base):
    """Adds "{key}_color" (RGBA) and "{key}_size" (line width / marker
    radius) columns to df for one METRICS-style entry (dict with key/low/
    high/gamma/width_basis), scaled off df's own 98th-percentile values.
    Returns that 98th-percentile value (used as the legend's "max")."""
    key, low, high, gamma = metric["key"], metric["low"], metric["high"], metric["gamma"]
    values = df[key]
    positive = values[values > 0]
    max_value = max(positive.quantile(0.98), 1.0) if len(positive) else 1.0

    basis = df[metric["width_basis"]]
    positive_basis = basis[basis > 0]
    max_basis = max(positive_basis.quantile(0.98), 1.0) if len(positive_basis) else 1.0

    def color_for(value):
        if pd.isna(value) or value <= 0:
            return [*low, 60]
        normalized = (min(value, max_value) / max_value) ** gamma
        return [*gray_to_color(normalized, low, high), flow_alpha(normalized)]

    df[f"{key}_color"] = values.apply(color_for)
    df[f"{key}_size"] = np.where(
        basis > 0, (basis.clip(upper=max_basis) / max_basis) ** 0.5 * size_scale + size_base, size_base * 0.5,
    )
    return max_value


def metric_color_stops(low, high, gamma):
    return ", ".join(
        f"rgba({gray_to_color(t ** gamma, low, high)[0]},{gray_to_color(t ** gamma, low, high)[1]},"
        f"{gray_to_color(t ** gamma, low, high)[2]},{flow_alpha(t ** gamma) / 255:.2f}) {t * 100:.0f}%"
        for t in np.linspace(0, 1, 6)
    )


def inject_metric_dropdown_legend(
    path_to_save, metrics, metric_maxima, default_metric, layer_prefixes, dom_id_prefix,
    extra_legend_lines=(), mode="visibility",
):
    """
    Adds a "Show <metric>" dropdown + gradient legend to a saved pydeck HTML
    map, plus a script that switches the map to the selected metric. Only
    layers whose id starts with one of layer_prefixes are touched - anything
    else (extent outlines, etc.) is left alone. dom_id_prefix keeps this
    map's DOM element ids from colliding with any other pydeck map's, if
    ever embedded on the same page. extra_legend_lines: [(text, css_style),
    ...] appended below the standard "gray = no value" note.

    mode="visibility" (default, e.g. cross_border_flow_pt.py): there is one
    real pdk.Layer per metric (each with its own copy of the row data), and
    switching metric toggles which one is visible. The calling code must
    build each such layer's id containing the substring f"::{metric_key}::"
    (e.g. f"{prefix}::{key}::{chunk}").

    mode="accessor" (e.g. cross_border_flow_cars.py, where duplicating the
    row data per metric would bloat the map): there is a single set of
    layers whose data already carries every metric's "{key}_color"/
    "{key}_size" columns (see add_metric_columns), and switching metric
    re-clones those layers with a different color/width accessor instead of
    swapping which layer is visible - so only one copy of the geometry/
    tooltip data ever needs to be embedded, however many metrics there are.
    """
    options_html = "\n".join(
        f'<option value="{m["key"]}"{" selected" if m["key"] == default_metric else ""}>{escape(m["label"])}</option>'
        for m in metrics
    )
    metrics_js = {
        m["key"]: {
            "label": m["label"],
            "stops": metric_color_stops(m["low"], m["high"], m["gamma"]),
            "max": f"{metric_maxima[m['key']]:.0f}+",
        }
        for m in metrics
    }

    select_id = f"{dom_id_prefix}-metric-select"
    title_id = f"{dom_id_prefix}-legend-title"
    bar_id = f"{dom_id_prefix}-legend-bar"
    max_id = f"{dom_id_prefix}-legend-max"

    extra_html = "".join(
        f'  <div style="margin-top:2px;{style}">{escape(text)}</div>\n' for text, style in extra_legend_lines
    )

    legend_html = f"""
<div id="{dom_id_prefix}-legend" style="position:absolute;z-index:20;bottom:12px;left:12px;padding:10px 12px;
            border-radius:7px;background:rgba(255,255,255,0.95);
            box-shadow:0 2px 10px rgba(0,0,0,0.28);color:#222;
            font:12px/1.3 Arial, sans-serif;">
  <label style="display:block;margin-bottom:8px;">
    <div style="margin-bottom:3px;">Show</div>
    <select id="{select_id}" style="width:240px;">{options_html}</select>
  </label>
  <div id="{title_id}" style="font-weight:700;margin-bottom:6px;">{escape(default_metric)}</div>
  <div id="{bar_id}" style="width:180px;height:12px;border:1px solid #777;"></div>
  <div style="display:flex;justify-content:space-between;margin-top:3px;">
    <span>0</span><span id="{max_id}"></span>
  </div>
  <div style="margin-top:4px;color:#777;">Gray = no measured value for this metric</div>
{extra_html}</div>
"""
    script = f"""
<script>
(function () {{
  const metrics = {json.dumps(metrics_js)};
  const layerPrefixes = {json.dumps(list(layer_prefixes))};
  const mode = {json.dumps(mode)};
  const select = document.getElementById({json.dumps(select_id)});
  const title = document.getElementById({json.dumps(title_id)});
  const bar = document.getElementById({json.dumps(bar_id)});
  const maxLabel = document.getElementById({json.dumps(max_id)});
  const deck = typeof deckInstance === "undefined" ? null : deckInstance;

  function applyMetric(key) {{
    const info = metrics[key];
    title.textContent = info.label;
    bar.style.background = "linear-gradient(to right, " + info.stops + ")";
    maxLabel.textContent = info.max;

    if (!deck || !deck.props || !deck.props.layers) {{
      return;
    }}
    const marker = "::" + key + "::";
    const layers = deck.props.layers.map(function (layer) {{
      if (!layerPrefixes.some(function (p) {{ return layer.id.startsWith(p); }})) {{
        return layer;
      }}
      if (mode === "accessor") {{
        // updateTriggers is required, not optional: deck.gl's PathLayer
        // does not reliably recompute its color/width GPU attributes just
        // because the getColor/getWidth accessor is a new function
        // reference - without a changed updateTriggers value telling it
        // the accessor's *output* changed, it can silently keep rendering
        // the previous colors/widths (verified empirically - the dropdown
        // updated the legend text and the JS layer.props.getColor, but the
        // canvas itself never changed color without this).
        return layer.clone({{
          getColor: function (d) {{ return d[key + "_color"]; }},
          getWidth: function (d) {{ return d[key + "_size"]; }},
          updateTriggers: {{getColor: key, getWidth: key}},
        }});
      }}
      return layer.clone({{visible: layer.id.includes(marker)}});
    }});
    deck.setProps({{layers: layers}});
  }}

  select.addEventListener("change", function () {{ applyMetric(select.value); }});
  applyMetric(select.value);
}})();
</script>
"""
    path = Path(path_to_save)
    html = path.read_text(encoding="utf-8")
    html = html.replace("</body>", legend_html + "\n</body>", 1)
    html = html.replace("</html>", script + "\n</html>", 1)
    path.write_text(html, encoding="utf-8")
