"""
Three HTML maps comparing MATSim and TPG total passenger events, all built
with folium (Leaflet):

  - build_hourly_map: one dot per stop, replayed hour by hour with folium's
    TimestampedGeoJson plugin (time slider/play control, bottom-left).
  - build_full_day_map: one dot per stop, colored by its 6AM-10PM aggregate;
    no time control - instead, clicking a stop pops up an hour-by-hour chart
    (TPG mean/CI vs MATSim) for that stop alone.
  - build_line_map: one route polyline per PT line (both directions drawn
    together, in the same color); clicking it pops up one hour-by-hour
    chart per direction the line has (just one for years with no direction
    field, e.g. 2025).

Background tiles: swisstopo's public WMTS service (wmts.geo.admin.ch) -
free, no API key or auth of any kind, and specific to Switzerland, which
fits a Geneva-area map well. Both plain OpenStreetMap tiles and, later,
cartodbpositron turned out to need something this network doesn't have
(blocked / now requires a key), so this is the third basemap tried here.
Uses the grayscale variant at reduced opacity, so it stays a quiet backdrop
rather than competing with the colored dots/lines. The study perimeter
shapefile is drawn on top as an outline for context.

Color encodes MATSim's total passenger events as a PERCENTAGE OF THE TPG
MEAN (see comparison.compute_matsim_pct_of_mean): 100% = exact match, 0% =
MATSim predicts nothing, 200% = MATSim predicts twice the TPG mean. A
color-scale legend with that unit is drawn on the map.
"""

import math
import random
from datetime import datetime, timedelta

import branca.colormap as cm
import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import TimestampedGeoJson

import plotting

_BASE_TIME = datetime(2024, 1, 1)
_TILE_URL  = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"


def _build_colormap(color_half_range_pct):
    # Blue -> gray -> red: gray at the 100% midpoint (MATSim == TPG mean)
    # reads as "neutral", and stays distinct from the low-data dots (see
    # _is_low_data), which are also grayish but much fainter/smaller.
    lo = max(100 - color_half_range_pct, 0)
    hi = 100 + color_half_range_pct
    colormap = cm.LinearColormap(
        colors = ["#0854a0", "#9e9e9e", "#c1121f"],
        index  = [lo, 100, hi],
        vmin   = lo, vmax = hi,
    )
    colormap.caption = (
        "MATSim as % of TPG 2024 mean (100% = exact match; "
        "blue = MATSim too low; red = MATSim too high)"
    )
    return colormap


def _marker_radius(tpg_mean, matsim_total, global_max, low_data = False):
    magnitude = max(tpg_mean, matsim_total, 0)
    radius = 7 + 20 * (magnitude ** 0.5) / (max(global_max, 1) ** 0.5)
    return radius * 0.4 if low_data else radius


def _is_low_data(tpg_mean, matsim_total, min_events):
    # Comparing against a TPG mean this small is mostly noise (a single
    # extra rider swings the ratio wildly) - e.g. TPG mean=1, MATSim=0 in
    # the near-empty small hours. Also flag the symmetric case where TPG
    # says ~nothing but MATSim reports a handful of events.
    return max(tpg_mean, matsim_total) < min_events


def _popup_html(row):
    return (
        f"<b>{row['stop_name']}</b><br>"
        f"MATSim (scaled): {row['matsim_total']:.0f}<br>"
        f"TPG 2024 mean: {row['tpg_mean']:.0f} "
        f"(95% CI {row['tpg_lo']:.0f}-{row['tpg_hi']:.0f})<br>"
        f"MATSim reaches {row['pct_of_mean']:.0f}% of the TPG mean"
    )


def _add_hourly_layer(m, stop_hour_df, colormap, global_max, low_data_min_events):
    features = []

    for _, row in stop_hour_df.iterrows():
        time = (_BASE_TIME + timedelta(hours = int(row["hour"]))).isoformat()
        low_data = _is_low_data(row["tpg_mean"], row["matsim_total"], low_data_min_events)

        if low_data:
            style = {
                "fillColor": "#9e9e9e", "fillOpacity": 0.12,
                "stroke": "true", "color": "#9e9e9e", "weight": 0.5,
                "radius": _marker_radius(row["tpg_mean"], row["matsim_total"], global_max, low_data = True),
            }
        else:
            style = {
                "fillColor": colormap(row["pct_of_mean"]), "fillOpacity": 0.95,
                "stroke": "true", "color": "#222222", "weight": 1.2,
                "radius": _marker_radius(row["tpg_mean"], row["matsim_total"], global_max),
            }

        popup = _popup_html(row) + (
            f"<br><i>Low data: TPG mean and MATSim both under {low_data_min_events:.0f} events - "
            "comparison not meaningful</i>" if low_data else ""
        )

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["stop_lon"], row["stop_lat"]]},
            "properties": {
                "time": time,
                "popup": popup,
                "icon": "circle",
                "iconstyle": style,
            },
        })

    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period = "PT1H",
        duration = "PT1H",
        transition_time = 300,
        auto_play = False,
        loop = False,
        add_last_point = False,
        time_slider_drag_update = True,
        date_options = "HH:mm",
    ).add_to(m)


def _add_perimeter_layer(m, perimeter_shapefile):
    boundary = gpd.read_file(perimeter_shapefile).to_crs("EPSG:4326")
    folium.GeoJson(
        boundary,
        name = "Study perimeter",
        style_function = lambda _: {"color": "#555555", "weight": 2, "fill": False},
    ).add_to(m)


def _base_map(center, perimeter_shapefile):
    m = folium.Map(location = center, zoom_start = 11, prefer_canvas = True, tiles = None)
    # Grayscale swisstopo layer, dimmed further via opacity, so the colored
    # dots (not the basemap) draw the eye - a full-color basemap plus a
    # busy diverging color scale on top of it was competing for attention.
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    if perimeter_shapefile:
        _add_perimeter_layer(m, perimeter_shapefile)

    return m


def build_hourly_map(stop_hour_df, output_path, perimeter_shapefile = None, color_half_range_pct = 100,
                      low_data_min_events = 5, hour_range = (6, 22)):
    """
    stop_hour_df: one row per (stop, hour), from comparison.build_stop_hour_table.
    perimeter_shapefile: optional path to the study perimeter shapefile, drawn
        as an outline overlay for context.
    color_half_range_pct: the color scale saturates at 100% +/- this value,
        e.g. the default 100 covers 0%-200% (MATSim predicts nothing, to
        MATSim predicts double the TPG mean).
    low_data_min_events: a (stop, hour) is drawn as a small, faint gray dot
        instead of its usual color/size when BOTH the TPG mean and the
        MATSim total are below this many events - e.g. TPG mean=1,
        MATSim=0 in the small hours, where the comparison is mostly noise.
    hour_range: (first_hour, last_hour) inclusive - only these hours are
        shown on the time slider. Default (6, 22) is 6AM-10PM.
    """

    first_hour, last_hour = hour_range
    stop_hour_df = stop_hour_df[stop_hour_df["hour"].between(first_hour, last_hour)]

    global_max = max(stop_hour_df[["tpg_mean", "matsim_total"]].max().max(), 1)

    center = [stop_hour_df["stop_lat"].mean(), stop_hour_df["stop_lon"].mean()]
    m = _base_map(center, perimeter_shapefile)

    _add_hourly_layer(m, stop_hour_df, _build_colormap(color_half_range_pct), global_max, low_data_min_events)

    _build_colormap(color_half_range_pct).add_to(m)
    folium.LayerControl(collapsed = False).add_to(m)

    m.save(output_path)


def build_full_day_map(stop_hour_df, full_day_df, output_path, perimeter_shapefile = None,
                        color_half_range_pct = 100, low_data_min_events = 5):
    """
    One dot per stop, colored/sized by its full_day_df aggregate (typically
    the 6AM-10PM window - see stages.run_global_comparison). No time
    control; instead, each stop's popup embeds an hour-by-hour chart
    (TPG mean/CI vs MATSim) built from its rows in stop_hour_df.

    stop_hour_df: one row per (stop, hour), already restricted to whichever
        hours the popup charts should cover.
    full_day_df: one row per stop (comparison.build_full_day_table), from
        the same hour window as stop_hour_df.
    """

    global_max = max(full_day_df[["tpg_mean", "matsim_total"]].max().max(), 1)
    colormap = _build_colormap(color_half_range_pct)

    hours_by_stop = {code: df for code, df in stop_hour_df.groupby("gtfs_code")}

    center = [full_day_df["stop_lat"].mean(), full_day_df["stop_lon"].mean()]
    m = _base_map(center, perimeter_shapefile)

    for _, row in full_day_df.iterrows():
        low_data = _is_low_data(row["tpg_mean"], row["matsim_total"], low_data_min_events)

        chart_html = ""
        stop_hours = hours_by_stop.get(row["gtfs_code"])
        if stop_hours is not None and not stop_hours.empty:
            chart_b64 = plotting.render_hourly_chart_png(stop_hours, row["stop_name"])
            chart_html = f'<br><img src="data:image/png;base64,{chart_b64}" width="320">'

        popup = _popup_html(row) + (
            f"<br><i>Low data: TPG mean and MATSim both under {low_data_min_events:.0f} events - "
            "comparison not meaningful</i>" if low_data else ""
        ) + chart_html

        if low_data:
            folium.CircleMarker(
                location = (row["stop_lat"], row["stop_lon"]),
                radius = _marker_radius(row["tpg_mean"], row["matsim_total"], global_max, low_data = True),
                color = "#9e9e9e", weight = 0.5, fill = True, fill_color = "#9e9e9e", fill_opacity = 0.12,
                popup = folium.Popup(popup, max_width = 360),
            ).add_to(m)
        else:
            folium.CircleMarker(
                location = (row["stop_lat"], row["stop_lon"]),
                radius = _marker_radius(row["tpg_mean"], row["matsim_total"], global_max),
                color = "#222222", weight = 1.2, fill = True, fill_color = colormap(row["pct_of_mean"]), fill_opacity = 0.95,
                popup = folium.Popup(popup, max_width = 360),
                tooltip = row["stop_name"],
            ).add_to(m)

    colormap.add_to(m)

    m.save(output_path)


def _offset_coords(coords, seed, magnitude_m = 12):
    """
    Shift every point of a polyline by the same small lat/lon delta, seeded
    so it's consistent across reruns. Purely cosmetic: separates a line's
    two directions, which usually run along the same streets and would
    otherwise be drawn exactly on top of each other.
    """

    if not coords:
        return coords

    rng = random.Random(seed)
    angle = rng.uniform(0, 2 * math.pi)
    lat0 = coords[0][0]
    dlat = (magnitude_m / 111_320) * math.sin(angle)
    dlon = (magnitude_m / (111_320 * math.cos(math.radians(lat0)))) * math.cos(angle)

    return [(lat + dlat, lon + dlon) for lat, lon in coords]


def build_line_map(line_hour_df, line_base_df, line_geometries_df, output_path, perimeter_shapefile = None,
                    color_half_range_pct = 100, low_data_min_events = 5):
    """
    One route polyline per (line, direction) - both directions of the same
    line drawn in the same color (line_base_df's aggregate, both directions
    combined), since they're meant to read as one line on the map. Clicking
    anywhere on a line pops up one hourly chart per direction it has in
    line_hour_df (just one for years with no direction field, e.g. 2025).

    line_hour_df: one row per (line_direction, hour), from
        comparison.build_line_hour_table, already restricted to whichever
        hours the popup charts should cover.
    line_base_df: one row per line (both directions combined), from
        comparison.build_line_base_table - drives color/low-data status.
    line_geometries_df: one row per (line, direction) with its route
        polyline, from tpg_data.build_line_route_geometries. A line with no
        matching geometry is skipped (count printed) - route shape is
        year-independent (see that function's docstring), so this can
        happen for a line whose stops didn't all resolve to a GTFS id.
    """

    global_max = max(line_base_df[["tpg_mean", "matsim_total"]].max().max(), 1)
    colormap = _build_colormap(color_half_range_pct)

    hours_by_line_base = {line: df for line, df in line_hour_df.groupby("line_base")}
    geometries_by_line = {line: df for line, df in line_geometries_df.groupby("line")}

    all_coords = [c for coords in line_geometries_df["coords"] for c in coords]
    center = [sum(c[0] for c in all_coords) / len(all_coords), sum(c[1] for c in all_coords) / len(all_coords)]
    m = _base_map(center, perimeter_shapefile)

    skipped = 0

    for _, row in line_base_df.iterrows():
        line_base = row["line_base"]
        geometries = geometries_by_line.get(line_base)
        if geometries is None or geometries.empty:
            skipped += 1
            continue

        low_data = _is_low_data(row["tpg_mean"], row["matsim_total"], low_data_min_events)
        color = "#9e9e9e" if low_data else colormap(row["pct_of_mean"])
        weight = 2 if low_data else 4
        opacity = 0.25 if low_data else 0.85

        chart_html = ""
        line_hours = hours_by_line_base.get(line_base)
        if line_hours is not None:
            for direction_letter, direction_hours in line_hours.groupby("direction_letter", dropna = False):
                title = f"Line {line_base}" + (f" ({direction_letter})" if pd.notna(direction_letter) else "")
                chart_b64 = plotting.render_hourly_chart_png(direction_hours, title)
                chart_html += f'<br><img src="data:image/png;base64,{chart_b64}" width="320">'

        popup_text = (
            f"<b>Line {line_base}</b><br>"
            f"MATSim (scaled): {row['matsim_total']:.0f}<br>"
            f"TPG mean: {row['tpg_mean']:.0f}<br>"
            f"MATSim reaches {row['pct_of_mean']:.0f}% of the TPG mean"
        ) + (
            f"<br><i>Low data: TPG mean and MATSim both under {low_data_min_events:.0f} events - "
            "comparison not meaningful</i>" if low_data else ""
        ) + chart_html
        for _, geom_row in geometries.iterrows():
            coords = _offset_coords(geom_row["coords"], seed = f"{line_base}_{geom_row['direction_letter']}")
            # A fresh Popup per polyline, not shared: a branca/folium Element
            # can only really be bound to one parent Layer (bindPopup only
            # gets emitted once, for whichever parent claims it last), so
            # reusing one Popup instance across a line's 2 direction
            # polylines silently left one of them with no popup at all -
            # the "not clickable" bug.
            folium.PolyLine(
                coords, color = color, weight = weight, opacity = opacity,
                popup = folium.Popup(popup_text, max_width = 360), tooltip = f"Line {line_base}",
            ).add_to(m)

    if skipped:
        print(f"build_line_map: skipped {skipped} line(s) with no route geometry")

    colormap.add_to(m)

    m.save(output_path)
