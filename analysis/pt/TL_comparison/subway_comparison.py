"""Map comparing the passengers of the subway lines (m1 and m2) in the MATSim passenger counts and in the TL counts.

For each line and direction the map shows
  - the boardings and alightings of every stop, MATSim against TL (a marker per direction, colored by the ratio
    MATSim / TL; the popup of the stop has the numbers and the hourly profiles),
  - the passenger flow between two stops (the passengers on board), MATSim and TL side by side, one lane per
    direction and source, and the ratio between the two,
and a panel with the same figures as charts.
"""
import argparse
import base64
import glob
import io
import math
import os
import re
import sys
from collections import defaultdict

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex
from pyproj import Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "transit_schedule"))

import visualize
from comparison_passenger_counts_lausanne import TL_TO_MATSIM_LINE

# the MATSim line -> the TL line
MATSIM_TO_TL_LINE = {matsim: tl for tl, matsim in TL_TO_MATSIM_LINE.items()}

# categorical slots 1 and 2 of the reference palette, and the neutral
MATSIM_COLOR, TL_COLOR = "#2a78d6", "#eb6834"
NEUTRAL = "#e4e3df"
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"

# diverging scale of the ratio MATSim / TL: orange below 1 (TL higher), gray at 1, blue above 1
RATIO_COLORMAP = LinearSegmentedColormap.from_list("ratio", [TL_COLOR, NEUTRAL, MATSIM_COLOR])
RATIO_LOG2_RANGE = 2.0  # ratios from 1/4 to 4

MAX_SNAP_DISTANCE = 250.0
LANE = 14.0   # m between the lanes of the flows
SIDE = 10.0   # m between the track and the first lane


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

    context.config("analysis.pt.tl_processed_path", default = "Lausanne_passenger_counts/tl data 2024/processed")
    context.config("analysis.pt.tl_gtfs_zip",       default = "gtfs/gtfs_fp2024_2024-11-11.zip")
    context.config("analysis.pt.night_hours",       default = 5)

    context.config("analysis.pt.subway_output_id", default = None)  # the run of output_id by default
    context.config("analysis.pt.subway_lines",     default = ["m1", "m2"])


def execute(context):
    data_path = context.config("data_path")
    output_id = context.config("analysis.pt.subway_output_id") or context.config("output_id")
    simulation_path = os.path.join(context.config("output_path"), output_id, context.config("simulation_directory"))

    lines = context.config("analysis.pt.subway_lines")
    if isinstance(lines, str):
        lines = [line.strip() for line in lines.split(",") if line.strip()]

    output_dir = run(
        simulation_path, os.path.join(data_path, context.config("analysis.pt.tl_processed_path")),
        os.path.join(data_path, context.config("analysis.pt.tl_gtfs_zip")),
        input_downsampling = context.config("input_downsampling"), lines = lines,
        night_hours = context.config("analysis.pt.night_hours"),
    )

    return dict(done = True, path = output_dir)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def code_of(stop_id):
    return str(stop_id).split(":")[0].split(".")[0]


def line_orders(routes, lines):
    """{line: stop codes in the order of its longest route}"""
    orders = {}
    for line in lines:
        candidates = [route for route in routes if route["line_name"] == line]
        if not candidates:
            raise RuntimeError(f"Line {line} not found in the transit schedule")

        order = []
        for stop_id in max(candidates, key = lambda route: len(route["stop_ids"]))["stop_ids"]:
            if not order or order[-1] != code_of(stop_id):
                order.append(code_of(stop_id))
        orders[line] = order

    return orders


def read_matsim(simulation_path, orders, names, scale, night_hours):
    """{(line, orientation): DataFrame(code, hour, boardings, alightings)}; orientation is fwd when the route runs in
    the order of the longest route of the line, rev otherwise."""
    counts = pd.read_csv(os.path.join(simulation_path, "pt_passenger_counts.csv.gz"), sep = ";")
    counts["hour"] = counts["time_bin"].str.extract(r"^(\d{1,2})").astype(int)

    if night_hours > 0:
        counts = counts[(counts["hour"] >= night_hours) & (counts["hour"] < 30)].copy()
        counts["hour"] %= 24
    else:
        counts = counts[counts["hour"] < 24]

    counts["code"] = counts["stop_id"].map(code_of)
    result = {}

    for line, order in orders.items():
        index = {names[code]: k for k, code in enumerate(order)}
        frame = counts[counts["line_name"] == line].copy()

        def orientation(direction):
            origin, _, destination = str(direction).partition("->")
            if origin in index and destination in index:
                return "fwd" if index[origin] < index[destination] else "rev"
            return None

        frame["orientation"] = frame["route_direction"].map(orientation)
        dropped = frame["orientation"].isna().sum()
        if dropped:
            print(f"  {line}: {dropped} count rows of routes with an unknown direction left out")

        for orientation_name, group in frame.dropna(subset = ["orientation"]).groupby("orientation"):
            result[(line, orientation_name)] = (
                group.groupby(["code", "hour"])[["boardings", "alightings"]].sum() * scale
            ).reset_index()

    return result


def stop_coordinates(gtfs_zip, codes):
    import zipfile
    with zipfile.ZipFile(gtfs_zip) as zf:
        member = next(n for n in zf.namelist() if n.endswith("stops.txt"))
        with zf.open(member) as f:
            stops = pd.read_csv(f, dtype = {"stop_id": str}, usecols = ["stop_id", "stop_lat", "stop_lon"])

    stops["code"] = stops["stop_id"].str.replace("^Parent", "", regex = True).str.split(":").str[0]
    stops = stops[stops["code"].isin(codes)].groupby("code")[["stop_lat", "stop_lon"]].mean()
    to_lv95 = Transformer.from_crs("EPSG:4326", visualize.NETWORK_CRS, always_xy = True)

    return {code: to_lv95.transform(row.stop_lon, row.stop_lat) for code, row in stops.iterrows()}


def _cumulative_flow(table, order):
    """Passengers on board after each stop of the order (daily): the boardings minus the alightings so far."""
    daily = table.groupby("code")[["boardings", "alightings"]].sum()
    flow, total = [], 0.0
    for code in order[:-1]:
        if code in daily.index:
            total += daily.loc[code, "boardings"] - daily.loc[code, "alightings"]
        flow.append(total)

    return np.array(flow)


def _hourly_flow(table, order):
    """Passengers on board after each stop of the order for each hour (array stops - 1 x 24): the boardings minus the
    alightings of that hour so far. The passengers that board in an hour and alight in the next one are not followed,
    so this is an approximation of the flow of the hour, computed the same way for both sources."""
    hours = np.arange(24)
    per_hour = {code: group.set_index("hour")[["boardings", "alightings"]].reindex(hours, fill_value = 0.0)
                for code, group in table.groupby("code")}
    flow, total = [], np.zeros(24)
    for code in order[:-1]:
        if code in per_hour:
            total = total + per_hour[code]["boardings"].to_numpy() - per_hour[code]["alightings"].to_numpy()
        flow.append(np.clip(total, 0, None))

    return np.array(flow)


def segment_chart(title, matsim_hours, tl_hours):
    """Hourly flow on a connection between two stops, MATSim against TL."""
    fig, ax = plt.subplots(figsize = (7.2, 3.2), facecolor = SURFACE)
    _style(ax, title)
    ax.tick_params(labelsize = 9)
    hours = np.arange(24)
    for values, color, label in ((matsim_hours, MATSIM_COLOR, "MATSim"), (tl_hours, TL_COLOR, "TL")):
        if values is not None:
            ax.plot(hours, values, color = color, linewidth = 2, label = label)
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylabel("passengers on board per hour", fontsize = 8, color = INK_SECONDARY)
    ax.set_xlabel("hour", fontsize = 8, color = INK_SECONDARY)
    ax.set_ylim(bottom = 0)
    ax.legend(frameon = False, fontsize = 9)
    fig.tight_layout()

    return _png(fig)


def read_tl(processed_dir, gtfs_zip, orders, xy):
    """{(line, orientation): DataFrame(code, hour, boardings, alightings)}; the TL stops are moved to the closest stop of
    the line (within 250 m) and the orientation is the one that gives a consistent flow (never negative, back to 0)."""
    candidates = sorted(
        path for path in glob.glob(os.path.join(processed_dir, "tl2024_agg_workday_*.csv"))
        if re.fullmatch(r"tl2024_agg_workday_\d{8}\.csv", os.path.basename(path))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed TL stats in {processed_dir}: run analysis.pt.TL_comparison.tl_2024 first")

    print(f"TL counts: {candidates[-1]}")
    tl = pd.read_csv(candidates[-1])
    tl["line"] = tl["line_direction"].str.rsplit("_", n = 1).str[0]
    tl["direction"] = tl["line_direction"].str[-1]
    tl["code"] = tl["gtfs_code"].astype(str).str.split(".").str[0]

    codes = set(tl["code"])
    tl_xy = stop_coordinates(gtfs_zip, codes)
    result = {}

    for line, order in orders.items():
        frame = tl[tl["line"] == MATSIM_TO_TL_LINE.get(line, line)]

        mapping = {}
        for code in set(frame["code"]):
            if code in order:
                mapping[code] = code
            elif code in tl_xy:
                distances = [(math.hypot(tl_xy[code][0] - xy[c][0], tl_xy[code][1] - xy[c][1]), c) for c in order if c in xy]
                if distances and min(distances)[0] <= MAX_SNAP_DISTANCE:
                    mapping[code] = min(distances)[1]

        unmatched = sorted(set(frame["code"]) - set(mapping))
        if unmatched:
            print(f"  {line}: TL stops without a stop of the line within {MAX_SNAP_DISTANCE:.0f} m, left out: {unmatched}")

        frame = frame.assign(code = frame["code"].map(mapping)).dropna(subset = ["code"])

        for direction, group in frame.groupby("direction"):
            table = (group.groupby(["code", "hour"])[["boardings_raw_mean", "alightings_raw_mean"]].sum()
                     .rename(columns = {"boardings_raw_mean": "boardings", "alightings_raw_mean": "alightings"}).reset_index())

            best = None
            for orientation, stops_order in (("fwd", order), ("rev", order[::-1])):
                flow  = _cumulative_flow(table, stops_order)
                score = abs(flow[-1]) + abs(min(flow.min(), 0))
                if best is None or score < best[0]:
                    best = (score, orientation)
            result[(line, best[1])] = table

    return result


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _style(ax, title = None):
    if title:
        ax.set_title(title, loc = "left", fontsize = 9, color = INK, fontweight = "bold")
    ax.set_facecolor(SURFACE)
    ax.grid(axis = "y", color = GRID, linewidth = 0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors = INK_SECONDARY, labelsize = 7, length = 0)


def _png(fig):
    """The figure as a base64 PNG, with a reduced palette (the charts have a few flat colors): the map embeds
    hundreds of them."""
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format = "png", dpi = 120, facecolor = SURFACE)
    plt.close(fig)

    buffer.seek(0)
    small = io.BytesIO()
    Image.open(buffer).convert("RGB").quantize(colors = 32).save(small, format = "PNG", optimize = True)

    return base64.b64encode(small.getvalue()).decode()


def hourly_chart(matsim, tl, code, title_by_orientation):
    """Hourly profile of a stop: a column per direction, the boardings above and the alightings below."""
    fig, axes = plt.subplots(2, 2, figsize = (8.4, 5.2), facecolor = SURFACE, sharex = True)
    hours = np.arange(24)

    for column, orientation in enumerate(("fwd", "rev")):
        for row, measure in enumerate(("boardings", "alightings")):
            ax = axes[row, column]
            _style(ax, f"{title_by_orientation[orientation]}: {measure}" if row == 0 else measure)
            for source, color, label in ((matsim, MATSIM_COLOR, "MATSim"), (tl, TL_COLOR, "TL")):
                table = source.get(orientation)
                if table is None:
                    continue

                rows = table[table["code"] == code].set_index("hour")[["boardings", "alightings"]].reindex(hours, fill_value = 0)
                ax.plot(hours, rows[measure], color = color, linewidth = 2, label = label)
            ax.set_xticks(range(0, 24, 3))
            ax.set_ylim(bottom = 0)
            if all(len(line.get_ydata()) == 0 or not np.any(line.get_ydata()) for line in ax.lines):
                ax.set_yticks([])
                ax.text(0.5, 0.5, f"no {measure} (end of the line)", transform = ax.transAxes, ha = "center", va = "center",
                        fontsize = 9, color = INK_SECONDARY)
            ax.tick_params(labelsize = 9)
            if row == 1:
                ax.set_xlabel("hour", fontsize = 8, color = INK_SECONDARY)

    axes[0, 0].legend(frameon = False, fontsize = 9)
    fig.tight_layout()

    return _png(fig)


def line_figure(line, order, names, matsim, tl, titles):
    """Per direction: boardings and alightings by stop, and the flow between the stops, MATSim against TL."""
    fig, axes = plt.subplots(3, 2, figsize = (10.5, 8.6), facecolor = SURFACE)

    for column, orientation in enumerate(("fwd", "rev")):
        stops_order = order if orientation == "fwd" else order[::-1]
        labels = [names[c].replace("Lausanne, ", "").replace("Ecublens VD, ", "").replace("Chavannes-R., ", "") for c in stops_order]
        x = np.arange(len(stops_order))

        for row, (measure, title) in enumerate((("boardings", "Boardings"), ("alightings", "Alightings"))):
            ax = axes[row, column]
            _style(ax, f"{titles[orientation]}: {title.lower()} per day")
            for k, (source, color, label) in enumerate(((matsim, MATSIM_COLOR, "MATSim"), (tl, TL_COLOR, "TL"))):
                table = source.get(orientation)
                daily = table.groupby("code")[measure].sum() if table is not None else pd.Series(dtype = float)
                ax.bar(x + (k - 0.5) * 0.38, [daily.get(c, 0) for c in stops_order], width = 0.36, color = color, label = label)
            ax.set_xticks(x, labels, rotation = 60, ha = "right")
            if row == 0 and column == 0:
                ax.legend(frameon = False, fontsize = 8)

        ax = axes[2, column]
        _style(ax, f"{titles[orientation]}: passengers on board between the stops")
        for source, color, label in ((matsim, MATSIM_COLOR, "MATSim"), (tl, TL_COLOR, "TL")):
            table = source.get(orientation)
            if table is not None:
                ax.step(x[:-1], _cumulative_flow(table, stops_order), where = "mid", color = color, linewidth = 2, label = label)
        ax.set_xticks(x[:-1], [f"{a} → {b}" for a, b in zip(labels[:-1], labels[1:])], rotation = 60, ha = "right", fontsize = 6)

    fig.suptitle(f"Line {line}", x = 0.01, ha = "left", fontsize = 11, color = INK, fontweight = "bold")
    fig.tight_layout()

    return _png(fig)


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

def ratio_color(ratio):
    if ratio is None or not np.isfinite(ratio) or ratio <= 0:
        return "#9a9994"
    position = 0.5 + 0.5 * np.clip(math.log2(ratio) / RATIO_LOG2_RANGE, -1, 1)
    return to_hex(RATIO_COLORMAP(position))


def _ratio(matsim, tl):
    return matsim / tl if tl > 0 else None


def _offset(a, b, meters):
    """Shift a segment sideways (to the right of a → b) by the given meters (LV95)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = dy / length, -dx / length

    return (a[0] + nx * meters, a[1] + ny * meters), (b[0] + nx * meters, b[1] + ny * meters)


def build_map(output_path, orders, names, xy, matsim, tl, hourly, segment_charts, figures, max_flow, summary_html):
    to_wgs = Transformer.from_crs(visualize.NETWORK_CRS, "EPSG:4326", always_xy = True)

    def latlon(point):
        lon, lat = to_wgs.transform(*point)
        return (lat, lon)

    all_points = [xy[c] for order in orders.values() for c in order if c in xy]
    center = np.mean(all_points, axis = 0)
    m = folium.Map(location = latlon(center), zoom_start = 13, tiles = None, prefer_canvas = True)
    folium.TileLayer(tiles = visualize.TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", control = False).add_to(m)

    titles = {}
    for line, order in orders.items():
        titles[line] = {"fwd": f"{names[order[0]]} → {names[order[-1]]}", "rev": f"{names[order[-1]]} → {names[order[0]]}"}

    def connection_popup(line, orientation, k):
        """Both directions of the connection, the one that was clicked first, each with its own daily flows and chart."""
        order = orders[line]
        sections = []
        for o in (orientation, "rev" if orientation == "fwd" else "fwd"):
            stops_order = order if o == "fwd" else order[::-1]
            index = k if o == orientation else len(order) - 2 - k
            a, b = stops_order[index], stops_order[index + 1]
            daily = {name: (_cumulative_flow(data[(line, o)], stops_order)[index] if (line, o) in data else None)
                     for name, data in (("MATSim", matsim), ("TL", tl))}
            ratio = _ratio(daily["MATSim"], daily["TL"]) if None not in daily.values() else None
            sections.append(
                f"<div style='margin-bottom:8px'><b>{line}: {names[a]} → {names[b]}</b>"
                f"{' (clicked)' if o == orientation else ''}<br>per day: MATSim {daily['MATSim'] or 0:.0f}, TL {daily['TL'] or 0:.0f}"
                + (f", ratio {ratio:.2f}" if ratio is not None else "")
                + f"<br><img src='data:image/png;base64,{segment_charts[(line, o, index)]}' style='width:560px'></div>"
            )

        return "".join(sections)

    # Flows: one lane per direction and source
    for line, order in orders.items():
        for orientation in ("fwd", "rev"):
            stops_order = order if orientation == "fwd" else order[::-1]
            side = 1 if orientation == "fwd" else -1
            flows = {
                "MATSim": _cumulative_flow(matsim[(line, orientation)], stops_order) if (line, orientation) in matsim else None,
                "TL": _cumulative_flow(tl[(line, orientation)], stops_order) if (line, orientation) in tl else None,
            }

            layers = {
                "MATSim": folium.FeatureGroup(name = f"flow {line} {titles[line][orientation]}: MATSim", show = True),
                "TL": folium.FeatureGroup(name = f"flow {line} {titles[line][orientation]}: TL", show = True),
                "ratio": folium.FeatureGroup(name = f"flow {line} {titles[line][orientation]}: MATSim / TL", show = False),
            }

            for k, (a, b) in enumerate(zip(stops_order[:-1], stops_order[1:])):
                if a not in xy or b not in xy:
                    continue

                values = {source: (flow[k] if flow is not None else None) for source, flow in flows.items()}
                ratio  = _ratio(values["MATSim"], values["TL"]) if None not in values.values() else None
                label  = (f"{line} {names[a]} → {names[b]}: MATSim {values['MATSim'] or 0:.0f}, TL {values['TL'] or 0:.0f}"
                          + (f", ratio {ratio:.2f}" if ratio is not None else ""))

                popup_html = connection_popup(line, orientation, k)

                for source, color, lane in (("TL", TL_COLOR, 0), ("MATSim", MATSIM_COLOR, 1)):
                    if values[source] is None:
                        continue
                    start, end = _offset(xy[a], xy[b], side * (SIDE + lane * LANE))
                    folium.PolyLine([latlon(start), latlon(end)], color = color, weight = 1 + 13 * max(values[source], 0) / max_flow,
                                    opacity = 0.85, tooltip = label, popup = folium.Popup(popup_html, max_width = 600)).add_to(layers[source])

                start, end = _offset(xy[a], xy[b], side * (SIDE + 0.5 * LANE))
                folium.PolyLine([latlon(start), latlon(end)], color = ratio_color(ratio), weight = 7, opacity = 0.9,
                                tooltip = label, popup = folium.Popup(popup_html, max_width = 600)).add_to(layers["ratio"])

            for layer in layers.values():
                layer.add_to(m)

    # Stops: a marker per direction (ratio of the boardings and alightings), a popup per stop
    stop_layer = folium.FeatureGroup(name = "stops: boardings + alightings, MATSim / TL", show = True)
    served = defaultdict(list)
    for line, order in orders.items():
        for code in order:
            served[code].append(line)

    for code, lines_here in served.items():
        if code not in xy:
            continue

        sections = []
        for line in lines_here:
            order = orders[line]
            rows = ""
            for orientation in ("fwd", "rev"):
                values = {}
                for source, data in (("MATSim", matsim), ("TL", tl)):
                    table = data.get((line, orientation))
                    daily = table[table["code"] == code][["boardings", "alightings"]].sum() if table is not None else None
                    values[source] = (0.0, 0.0) if daily is None else (float(daily["boardings"]), float(daily["alightings"]))

                total_ratio = _ratio(sum(values["MATSim"]), sum(values["TL"]))
                rows += (f"<tr><td>{titles[line][orientation]}</td>"
                         f"<td align='right'>{values['MATSim'][0]:.0f} / {values['TL'][0]:.0f}</td>"
                         f"<td align='right'>{values['MATSim'][1]:.0f} / {values['TL'][1]:.0f}</td>"
                         f"<td align='right'>{'' if total_ratio is None else f'{total_ratio:.2f}'}</td></tr>")

                # the marker of the direction, shifted to its side of the track
                index = order.index(code)
                neighbour = order[index + 1] if index + 1 < len(order) else order[index - 1]
                direction_sign = 1 if orientation == "fwd" else -1
                if index + 1 >= len(order):
                    direction_sign *= -1
                start, _ = _offset(xy[code], xy[neighbour], direction_sign * 28.0)
                size = sum(values["MATSim"]) + sum(values["TL"])
                folium.CircleMarker(
                    latlon(start), radius = 3 + 0.045 * math.sqrt(size) * 3, color = INK, weight = 1, fill = True,
                    fill_color = ratio_color(total_ratio), fill_opacity = 0.95,
                    tooltip = (f"{line} {names[code]}, {titles[line][orientation]}: boardings MATSim {values['MATSim'][0]:.0f} / TL {values['TL'][0]:.0f}, "
                               f"alightings MATSim {values['MATSim'][1]:.0f} / TL {values['TL'][1]:.0f}"),
                ).add_to(stop_layer)

            image = hourly[(line, code)]
            sections.append(
                f"<b>{line}</b><table style='font-size:12px'><tr><th align='left'>direction</th><th>boardings M / TL</th>"
                f"<th>alightings M / TL</th><th>ratio</th></tr>{rows}</table>"
                f"<img src='data:image/png;base64,{image}' style='width:600px'>"
            )

        folium.CircleMarker(
            latlon(xy[code]), radius = 5, color = INK, weight = 1.5, fill = True, fill_color = "#ffffff", fill_opacity = 0.9,
            popup = folium.Popup(f"<h4 style='margin:0 0 4px'>{names[code]}</h4>" + "<hr>".join(sections), max_width = 640),
            tooltip = names[code],
        ).add_to(stop_layer)
    stop_layer.add_to(m)

    folium.LayerControl(collapsed = False).add_to(m)

    gradient = ", ".join(ratio_color(2 ** (RATIO_LOG2_RANGE * t)) for t in np.linspace(-1, 1, 9))
    panel = f"""
    <details open style="position:fixed;top:10px;right:10px;z-index:9999;width:560px;max-height:94vh;overflow:auto;background:{SURFACE};
      color:{INK};border:1px solid {GRID};border-radius:8px;padding:10px 14px;font:13px/1.4 system-ui,sans-serif">
      <summary style="cursor:pointer;font-weight:600">Subway lines: MATSim against TL</summary>
      {summary_html}
      <p style="color:{INK_SECONDARY};margin:6px 0"><span style="color:{MATSIM_COLOR}">&#9632;</span> MATSim
         <span style="color:{TL_COLOR}">&#9632;</span> TL. Flows: the lanes are, from the track outwards, TL then MATSim, on the right of
         the direction of travel; the width is the number of passengers on board. Markers: one per direction, beside the track, colored by the ratio
         MATSim / TL of the boardings + alightings.</p>
      <div style="height:12px;background:linear-gradient(to right, {gradient});border-radius:3px"></div>
      <div style="display:flex;justify-content:space-between;font-size:11px;color:{INK_SECONDARY}"><span>&le; 1/4 (MATSim lower)</span><span>1</span><span>&ge; 4 (MATSim higher)</span></div>
      {''.join(f"<img src='data:image/png;base64,{image}' style='width:100%;margin-top:8px'>" for image in figures)}
    </details>"""
    m.get_root().html.add_child(folium.Element(panel))

    m.save(output_path)


# ---------------------------------------------------------------------------

def run(simulation_path, tl_processed_dir, gtfs_zip, input_downsampling = 1.0, lines = ("m1", "m2"), night_hours = 5, output_dir = None):
    schedule_path = visualize._find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    output_dir    = output_dir or os.path.join(simulation_path, "pt_subway_comparison")
    os.makedirs(output_dir, exist_ok = True)
    scale = 1.0 / input_downsampling

    stops, routes = visualize.load_transit_schedule(schedule_path)
    orders = line_orders(routes, list(lines))

    names, coordinates = {}, defaultdict(list)
    for stop_id, info in stops.items():
        names.setdefault(code_of(stop_id), info["name"])
        if info["x"] is not None:
            coordinates[code_of(stop_id)].append((info["x"], info["y"]))
    xy = {code: tuple(np.mean(points, axis = 0)) for code, points in coordinates.items()}

    print("Reading the MATSim passenger counts ...")
    matsim = read_matsim(simulation_path, orders, names, scale, night_hours)
    print("Reading the TL counts ...")
    tl = read_tl(tl_processed_dir, gtfs_zip, orders, xy)

    titles = {line: {"fwd": f"{names[order[0]]} → {names[order[-1]]}", "rev": f"{names[order[-1]]} → {names[order[0]]}"}
              for line, order in orders.items()}

    # The tables
    records = []
    for line, order in orders.items():
        for orientation in ("fwd", "rev"):
            stops_order = order if orientation == "fwd" else order[::-1]
            flows = {name: (_cumulative_flow(data[(line, orientation)], stops_order) if (line, orientation) in data else np.full(len(order) - 1, np.nan))
                     for name, data in (("matsim", matsim), ("tl", tl))}

            for k, code in enumerate(stops_order):
                row = dict(line = line, direction = titles[line][orientation], stop = names[code], stop_code = code, position = k)
                for name, data in (("matsim", matsim), ("tl", tl)):
                    table = data.get((line, orientation))
                    daily = table[table["code"] == code][["boardings", "alightings"]].sum() if table is not None else None
                    row[f"boardings_{name}"]  = np.nan if daily is None else daily["boardings"]
                    row[f"alightings_{name}"] = np.nan if daily is None else daily["alightings"]
                    row[f"flow_to_next_{name}"] = flows[name][k] if k < len(order) - 1 else np.nan
                records.append(row)

    table = pd.DataFrame(records)
    table["boardings_ratio"]  = table["boardings_matsim"] / table["boardings_tl"]
    table["alightings_ratio"] = table["alightings_matsim"] / table["alightings_tl"]
    table["flow_ratio"]       = table["flow_to_next_matsim"] / table["flow_to_next_tl"]
    table.to_csv(os.path.join(output_dir, "subway_stops_and_flows.csv"), index = False)

    hourly_rows = []
    for line, order in orders.items():
        for orientation in ("fwd", "rev"):
            stops_order = order if orientation == "fwd" else order[::-1]
            for name, data in (("matsim", matsim), ("tl", tl)):
                if (line, orientation) not in data:
                    continue
                flow = _hourly_flow(data[(line, orientation)], stops_order)
                for k, (a, b) in enumerate(zip(stops_order[:-1], stops_order[1:])):
                    hourly_rows += [dict(line = line, direction = titles[line][orientation], from_stop = names[a], to_stop = names[b],
                                         source = name, hour = h, passengers = flow[k][h]) for h in range(24)]
    pd.DataFrame(hourly_rows).to_csv(os.path.join(output_dir, "subway_hourly_flows.csv"), index = False)

    hourly = {}
    for line, order in orders.items():
        for code in order:
            hourly[(line, code)] = hourly_chart(
                {o: matsim[(line, o)] for o in ("fwd", "rev") if (line, o) in matsim},
                {o: tl[(line, o)] for o in ("fwd", "rev") if (line, o) in tl}, code, titles[line],
            )

    segment_charts = {}
    for line, order in orders.items():
        for orientation in ("fwd", "rev"):
            stops_order = order if orientation == "fwd" else order[::-1]
            hours = {name: (_hourly_flow(data[(line, orientation)], stops_order) if (line, orientation) in data else None)
                     for name, data in (("MATSim", matsim), ("TL", tl))}

            for k, (a, b) in enumerate(zip(stops_order[:-1], stops_order[1:])):
                segment_charts[(line, orientation, k)] = segment_chart(
                    f"{line} {names[a]} → {names[b]}", None if hours["MATSim"] is None else hours["MATSim"][k],
                    None if hours["TL"] is None else hours["TL"][k],
                )

    figures = [
        line_figure(line, order, names, {o: matsim[(line, o)] for o in ("fwd", "rev") if (line, o) in matsim},
                    {o: tl[(line, o)] for o in ("fwd", "rev") if (line, o) in tl}, titles[line])
        for line, order in orders.items()
    ]

    max_flow = max(
        [np.nanmax(_cumulative_flow(t, order if o == "fwd" else order[::-1]))
         for data in (matsim, tl) for (line, o), t in data.items() for order in [orders[line]]] + [1.0]
    )

    totals = table.groupby("line")[["boardings_matsim", "boardings_tl", "alightings_matsim", "alightings_tl"]].sum()
    summary_html = "<table style='font-size:12px;border-collapse:collapse'><tr><th align='left'>line</th><th>boardings MATSim</th><th>TL</th><th>ratio</th></tr>" + "".join(
        f"<tr><td>{line}</td><td align='right'>{r.boardings_matsim:.0f}</td><td align='right'>{r.boardings_tl:.0f}</td>"
        f"<td align='right'>{r.boardings_matsim / r.boardings_tl:.2f}</td></tr>" for line, r in totals.iterrows()
    ) + "</table>"

    output_path = os.path.join(output_dir, "subway_comparison_map.html")
    build_map(output_path, orders, names, xy, matsim, tl, hourly, segment_charts, figures, max_flow, summary_html)
    print(totals.round(0).to_string())
    print(f"Map written to {output_path}")

    return output_dir


def main(argv = None):
    parser = argparse.ArgumentParser(description = "Map comparing the MATSim and TL passengers of the subway lines")
    parser.add_argument("--simulation-path", required = True, help = "MATSim output folder (pt_passenger_counts.csv.gz and transit schedule)")
    parser.add_argument("--tl-processed", required = True, help = "Folder with tl2024_agg_workday_<date>.csv")
    parser.add_argument("--gtfs-zip", required = True)
    parser.add_argument("--input-downsampling", type = float, required = True)
    parser.add_argument("--lines", nargs = "*", default = ["m1", "m2"])
    parser.add_argument("--night-hours", type = int, default = 5)
    parser.add_argument("--output-dir", default = None)
    args = parser.parse_args(argv)

    run(args.simulation_path, args.tl_processed, args.gtfs_zip, args.input_downsampling, args.lines, args.night_hours, args.output_dir)


if __name__ == "__main__":
    main()
