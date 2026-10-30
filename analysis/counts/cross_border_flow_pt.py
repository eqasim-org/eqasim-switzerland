import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xml.etree.ElementTree as ET
import xopen
from shapely.geometry import LineString
from shapely.ops import unary_union

from . import flow_map_page
from .cross_border_flow_cars import _extent_bounds, _find_output_file, _outline_rings
from .flow_metrics import share_column
from .cross_border_flow_events import GROUPS as EVENT_GROUPS
from .paths import (
    configure_flow_map_extents, configure_simulation_path, get_analysis_output_path, get_flow_map_extents,
    get_simulation_path,
)

logger = logging.getLogger("synpp")

STOP_COORDINATE_CRS = "epsg:2056"

# Rider-group flow columns (the counts of analysis.counts.cross_border_flow_events),
# on top of total_flow. "crossborder_pt_flow" is
# non-Swiss cross-border residents only (foreign commuters/travelers) -
# Swiss residents crossing the border for an activity get their own
# swiss_crossborder_pt_flow instead, so the two never overlap (see execute).
# Each column gets a matching "<name>_share_pct" column (flow_metrics.share_column)
# and a pair of METRICS entries.
RIDER_GROUP_FLOW_COLUMNS = ["crossborder_pt_flow", "swiss_crossborder_pt_flow", "french_pt_flow"]


def configure(context):
    context.stage("analysis.counts.cross_border_flow_events")
    configure_simulation_path(context)
    configure_flow_map_extents(context)
    context.config("only_weekday", default=True)
    context.config("input_downsampling")
    context.config("pt_flow_min_daily_count", default=1)


def execute(context):
    simulation_path = get_simulation_path(context)
    output_path = get_analysis_output_path(context)
    os.makedirs(output_path, exist_ok=True)

    schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")
    stop_coords, stop_names, _ = load_transit_schedule(schedule_path)
    logger.info("Loaded %d transit stops from %s", len(stop_coords), schedule_path)

    # the events are read once for the car and PT maps (analysis.counts.cross_border_flow_events)
    events = context.stage("analysis.counts.cross_border_flow_events")
    df_hourly = pd.read_csv(events["pt_hourly"], dtype={"from_stop": str, "to_stop": str})
    df_hourly = df_hourly.rename(columns={"total": "total_flow", **dict(zip(EVENT_GROUPS, RIDER_GROUP_FLOW_COLUMNS))})
    segment_lines = pd.read_csv(events["pt_lines"], dtype=str, keep_default_na=False)
    lines_by_segment = {(a, b): lines.split(", ") for a, b, lines in segment_lines.itertuples(index=False)}
    logger.info("PT ridership on %d distinct stop-to-stop segments", len(df_hourly.groupby(["from_stop", "to_stop"])))

    sample_size = context.config("input_downsampling")
    for column in ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]:
        df_hourly[column] = df_hourly[column] / sample_size

    df_daily = aggregate_daily(df_hourly)
    df_daily = attach_names(df_daily, stop_names, lines_by_segment)

    stop_totals = load_official_pt_stop_totals(simulation_path, sample_size)
    df_daily = attach_official_stop_totals(df_daily, stop_totals)
    df_daily.to_csv(os.path.join(output_path, "cross_border_pt_segment_flow_daily.csv"), index=False)

    min_daily_count = context.config("pt_flow_min_daily_count")
    df_plot = df_daily[df_daily.total_flow >= min_daily_count].copy()

    for name, extent_path in get_flow_map_extents(context).items():
        plot_crossborder_pt_flow_map(df_plot, stop_coords, extent_path, output_path, stop_names, stop_totals, name=name)

    return dict(done=True, path=output_path)


# ---------------------------------------------------------------------------
# Transit schedule parsing
# ---------------------------------------------------------------------------

def load_transit_schedule(schedule_path, line_labels=None):
    stop_coords = {}
    stop_names = {}
    vehicle_lines = {}
    current_line_label = None

    with xopen.xopen(schedule_path, "rb") as f:
        for event, elem in ET.iterparse(f, events=["start", "end"]):
            if event == "start":
                if elem.tag == "transitLine":
                    current_line_label = elem.attrib.get("name") or elem.attrib["id"]
                    if line_labels is not None:
                        line_labels[elem.attrib["id"]] = current_line_label
                continue

            if elem.tag == "stopFacility":
                stop_coords[elem.attrib["id"]] = (float(elem.attrib["x"]), float(elem.attrib["y"]))
                name = elem.attrib.get("name")
                if name:
                    stop_names[elem.attrib["id"]] = name
            elif elem.tag == "departure":
                vehicle_id = elem.attrib.get("vehicleRefId")
                if vehicle_id and current_line_label:
                    vehicle_lines[vehicle_id] = current_line_label
            elif elem.tag == "transitLine":
                current_line_label = None

            elem.clear()

    return stop_coords, stop_names, vehicle_lines


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_daily(df_hourly):
    df_daily = df_hourly.groupby(["from_stop", "to_stop"], as_index=False)[
        ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]
    ].sum()
    for column in RIDER_GROUP_FLOW_COLUMNS:
        df_daily[share_column(column)] = np.where(
            df_daily.total_flow > 0, 100.0 * df_daily[column] / df_daily.total_flow, np.nan
        )
    return df_daily.sort_values("crossborder_pt_flow", ascending=False).reset_index(drop=True)


def attach_names(df_daily, stop_names, lines_by_segment):
    """Adds from_stop_name/to_stop_name (falling back to the raw stop id
    where no name was found) and lines (comma-joined, sorted line labels
    seen on that segment - empty string if none)."""
    df_daily = df_daily.copy()
    df_daily["from_stop_name"] = df_daily["from_stop"].map(lambda s: stop_names.get(s, s))
    df_daily["to_stop_name"] = df_daily["to_stop"].map(lambda s: stop_names.get(s, s))
    df_daily["lines"] = [
        ", ".join(sorted(lines_by_segment.get((from_stop, to_stop), ())))
        for from_stop, to_stop in zip(df_daily.from_stop, df_daily.to_stop)
    ]
    return df_daily


# ---------------------------------------------------------------------------
# Official MATSim passenger counts (independent check on total_flow)
# ---------------------------------------------------------------------------

def load_official_pt_stop_totals(simulation_path, sample_size):
    try:
        path = _find_output_file(simulation_path, "pt_passenger_counts.csv.gz")
    except FileNotFoundError:
        logger.warning("pt_passenger_counts.csv.gz not found in %s; skipping official PT totals.", simulation_path)
        return pd.Series(dtype=float)

    counts = pd.read_csv(path, sep=";", usecols=["stop_id", "boardings", "alightings"])
    totals = (counts["boardings"] + counts["alightings"]).groupby(counts["stop_id"]).sum()
    return totals / sample_size


def attach_official_stop_totals(df_daily, stop_totals):
    df_daily = df_daily.copy()
    df_daily["from_stop_total_activity"] = df_daily["from_stop"].map(stop_totals)
    df_daily["to_stop_total_activity"] = df_daily["to_stop"].map(stop_totals)
    return df_daily


def aggregate_stop_level(df_daily, stop_coords, stop_names, stop_totals):
    flow_columns = ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]
    long = pd.concat([
        df_daily[["from_stop", *flow_columns]].rename(columns={"from_stop": "stop_id"}),
        df_daily[["to_stop", *flow_columns]].rename(columns={"to_stop": "stop_id"}),
    ], ignore_index=True)

    stops = long.groupby("stop_id", as_index=False)[flow_columns].sum()
    for column in RIDER_GROUP_FLOW_COLUMNS:
        stops[share_column(column)] = np.where(
            stops.total_flow > 0, 100.0 * stops[column] / stops.total_flow, np.nan
        )
    stops["stop_name"] = stops["stop_id"].map(lambda s: stop_names.get(s, s))
    stops["official_total_activity"] = stops["stop_id"].map(stop_totals)

    stops = stops[stops.stop_id.isin(stop_coords)].copy()
    stops["x"] = stops["stop_id"].map(lambda s: stop_coords[s][0])
    stops["y"] = stops["stop_id"].map(lambda s: stop_coords[s][1])
    return stops


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


METRICS = [
    {
        "key": "total_flow", "label": "Total PT flow (riders/day)",
        "low": (225, 225, 225), "high": (35, 139, 69), "gamma": 0.5, "width_basis": "total_flow",
    },
    {
        "key": "crossborder_pt_flow", "label": "Cross-border PT riders/day (non-Swiss residents)",
        "low": (225, 225, 225), "high": (8, 29, 225), "gamma": 0.5, "width_basis": "crossborder_pt_flow",
    },
    {
        "key": "crossborder_pt_share_pct", "label": "Cross-border share of PT riders (%, non-Swiss residents)",
        "low": (225, 225, 225), "high": (230, 85, 13), "gamma": 1.0, "width_basis": "total_flow",
    },
    {
        "key": "swiss_crossborder_pt_flow", "label": "Swiss-resident cross-border PT riders/day",
        "low": (225, 225, 225), "high": (0, 128, 128), "gamma": 0.5, "width_basis": "swiss_crossborder_pt_flow",
    },
    {
        "key": "swiss_crossborder_pt_share_pct", "label": "Swiss-resident cross-border share of PT riders (%)",
        "low": (225, 225, 225), "high": (166, 86, 40), "gamma": 1.0, "width_basis": "total_flow",
    },
    {
        "key": "french_pt_flow", "label": "France-resident PT riders/day",
        "low": (225, 225, 225), "high": (117, 25, 128), "gamma": 0.5, "width_basis": "french_pt_flow",
    },
    {
        "key": "french_pt_share_pct", "label": "France-resident share of PT riders (%)",
        "low": (225, 225, 225), "high": (203, 24, 29), "gamma": 1.0, "width_basis": "total_flow",
    },
]
DEFAULT_METRIC = "crossborder_pt_flow"


def plot_crossborder_pt_flow_map(df_daily, stop_coords, extent_path, output_path, stop_names=None, stop_totals=None, name=None):
    """Writes cross_border_pt_flow_map_<name>.html: the segments that intersect the extent."""
    if not os.path.exists(extent_path):
        logger.warning("Extent %s does not exist; skipping cross-border PT flow map %s.", extent_path, name)
        return

    has_coords = df_daily.from_stop.isin(stop_coords) & df_daily.to_stop.isin(stop_coords)
    if (~has_coords).any():
        logger.warning(
            "%d stop-to-stop segment(s) reference a stop missing from the transit schedule; dropped.",
            (~has_coords).sum(),
        )
    df_daily = df_daily[has_coords].copy()
    if df_daily.empty:
        logger.info("No PT segments with known stop coordinates; skipping cross-border PT flow map.")
        return

    df_daily["geometry"] = [
        LineString([stop_coords[from_stop], stop_coords[to_stop]])
        for from_stop, to_stop in zip(df_daily.from_stop, df_daily.to_stop)
    ]
    segments = gpd.GeoDataFrame(df_daily, geometry="geometry", crs=STOP_COORDINATE_CRS)

    boundary = gpd.read_file(extent_path)
    if boundary.crs != segments.crs:
        boundary = boundary.to_crs(segments.crs)
    region = shapely.force_2d(unary_union(boundary.geometry))

    segments = segments[segments.intersects(region)]
    if segments.empty:
        logger.info("No cross-border PT segments intersect the scenario extent; skipping map.")
        return

    # Compact page (flow_map_page): the data once, the colors/widths/tooltips of the selected metric computed in the browser
    flow_columns = ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]
    group_tooltip = [
        (m["label"], m["key"], "pct" if m["key"].endswith("_share_pct") else "int") for m in METRICS if m["key"] != "total_flow"
    ]

    segments = segments.to_crs(epsg=4326)
    line_layer = flow_map_page.layer(
        "path", [list(g.coords) for g in segments.geometry],
        {**{c: segments[c].to_numpy() for c in flow_columns},
         "from_stop_total_activity": segments["from_stop_total_activity"].to_numpy(),
         "to_stop_total_activity": segments["to_stop_total_activity"].to_numpy()},
        texts={"from": segments.from_stop_name.tolist(), "to": segments.to_stop_name.tolist(), "lines": segments.lines.tolist()},
        tooltip=[("From", "from", "text"), ("To", "to", "text"), ("Lines", "lines", "text"),
                 ("Total PT flow", "total_flow", "int"), *group_tooltip,
                 ("Boardings + alightings at origin", "from_stop_total_activity", "int"),
                 ("Boardings + alightings at destination", "to_stop_total_activity", "int")],
        size_scale=12, size_base=2, metrics=METRICS,
    )

    # Points at every stop, sized/colored by the same metrics as the lines
    stops = aggregate_stop_level(
        df_daily, stop_coords,
        stop_names if stop_names is not None else {},
        stop_totals if stop_totals is not None else pd.Series(dtype=float),
    )
    point_gdf = gpd.GeoDataFrame(stops, geometry=gpd.points_from_xy(stops.x, stops.y), crs=STOP_COORDINATE_CRS)
    point_gdf = point_gdf[point_gdf.intersects(region)].to_crs(epsg=4326)
    point_layer = flow_map_page.layer(
        "point", list(zip(point_gdf.geometry.x, point_gdf.geometry.y)),
        {**{c: point_gdf[c].to_numpy() for c in flow_columns},
         "official_total_activity": point_gdf["official_total_activity"].to_numpy()},
        texts={"stop": point_gdf.stop_name.tolist()},
        tooltip=[("Stop", "stop", "text"), ("Total PT flow", "total_flow", "int"), *group_tooltip,
                 ("Boardings + alightings", "official_total_activity", "int")],
        size_scale=14, size_base=5, metrics=METRICS,
    )

    output_file = os.path.join(output_path, f"cross_border_pt_flow_map_{name}.html" if name else "cross_border_pt_flow_map.html")
    flow_map_page.write_page(
        output_file, f"Cross-border PT flows{' - ' + name if name else ''}", [line_layer, point_layer], METRICS, DEFAULT_METRIC,
        outlines=[(_outline_rings(region, STOP_COORDINATE_CRS), [0, 0, 0], 2)],
        bounds=_extent_bounds(region, STOP_COORDINATE_CRS),
        legend_lines=[("Black outline = scenario extent", "color:#222;")],
    )
    logger.info("Saved cross-border PT flow map of %s to %s", name, output_file)
