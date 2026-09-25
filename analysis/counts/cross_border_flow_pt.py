import logging
import os
from collections import Counter, defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
import xml.etree.ElementTree as ET
import xopen
from pydeck.data_utils.viewport_helpers import compute_view
from shapely.geometry import LineString
from shapely.ops import unary_union

from .cross_border_flow_cars import (
    _find_output_file, _polygon_outline_layer,
    load_cross_border_person_ids, load_french_resident_cross_border_person_ids,
    load_swiss_resident_cross_border_person_ids,
)
from .flow_metrics import add_metric_columns, inject_metric_dropdown_legend, share_column
from .matching.plots import Plotter
from .paths import configure_simulation_path, get_analysis_output_path, get_simulation_path

logger = logging.getLogger("synpp")

PERSON_ENTERS_VEHICLE_EVENT = "PersonEntersVehicle"
PERSON_LEAVES_VEHICLE_EVENT = "PersonLeavesVehicle"

VEHICLE_DEPARTS_EVENT = "VehicleDepartsAtFacility"
VEHICLE_ARRIVES_EVENT = "VehicleArrivesAtFacility"
PROGRESS_EVERY = 2_000_000

STOP_COORDINATE_CRS = "epsg:2056"

# Rider-group flow columns produced by count_pt_stop_segment_riders /
# build_hourly_dataframe, on top of total_flow. "crossborder_pt_flow" is
# non-Swiss cross-border residents only (foreign commuters/travelers) -
# Swiss residents crossing the border for an activity get their own
# swiss_crossborder_pt_flow instead, so the two never overlap (see execute).
# Each column gets a matching "<name>_share_pct" column (flow_metrics.share_column)
# and a pair of METRICS entries.
RIDER_GROUP_FLOW_COLUMNS = ["crossborder_pt_flow", "swiss_crossborder_pt_flow", "french_pt_flow"]


def configure(context):
    configure_simulation_path(context)
    context.config("only_weekday", default=False)
    context.config("input_downsampling")
    context.config("pt_flow_min_daily_count", default=1)
    context.config("extent_path", default="")


def execute(context):
    simulation_path = get_simulation_path(context)
    output_path = get_analysis_output_path(context)
    os.makedirs(output_path, exist_ok=True)

    persons_path = _find_output_file(simulation_path, "output_persons.csv.gz")
    events_path = _find_output_file(simulation_path, "output_events.xml.gz")
    schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")

    cross_border_ids = load_cross_border_person_ids(persons_path)
    swiss_crossborder_ids = load_swiss_resident_cross_border_person_ids(persons_path)
    french_resident_ids = load_french_resident_cross_border_person_ids(persons_path)
    foreign_crossborder_ids = cross_border_ids - swiss_crossborder_ids
    logger.info(
        "Found %d cross-border persons (%d Swiss-resident, %d France-resident) in %s",
        len(cross_border_ids), len(swiss_crossborder_ids), len(french_resident_ids), persons_path,
    )

    stop_coords, stop_names, vehicle_lines = load_transit_schedule(schedule_path)
    logger.info("Loaded %d transit stops and %d vehicle-to-line mappings from %s",
                len(stop_coords), len(vehicle_lines), schedule_path)

    rider_group_ids = {
        "crossborder_pt_flow": foreign_crossborder_ids,
        "swiss_crossborder_pt_flow": swiss_crossborder_ids,
        "french_pt_flow": french_resident_ids,
    }
    total_counts, group_counts, lines_by_segment = count_pt_stop_segment_riders(
        events_path, rider_group_ids, vehicle_lines,
    )
    logger.info("Counted PT ridership on %d distinct stop-to-stop segments", len(total_counts))

    df_hourly = build_hourly_dataframe(total_counts, group_counts)

    sample_size = context.config("input_downsampling")
    for column in ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]:
        df_hourly[column] = df_hourly[column] / sample_size

    df_daily = aggregate_daily(df_hourly)
    df_daily = attach_names(df_daily, stop_names, lines_by_segment)

    stop_totals = load_official_pt_stop_totals(simulation_path, sample_size)
    df_daily = attach_official_stop_totals(df_daily, stop_totals)

    min_daily_count = context.config("pt_flow_min_daily_count")
    df_plot = df_daily[df_daily.total_flow >= min_daily_count].copy()

    plot_crossborder_pt_flow_map(
        df_plot, stop_coords, context.config("extent_path"), output_path, stop_names, stop_totals,
    )

    return dict(done=True, path=output_path)


# ---------------------------------------------------------------------------
# Transit schedule parsing
# ---------------------------------------------------------------------------

def load_transit_schedule(schedule_path):
    stop_coords = {}
    stop_names = {}
    vehicle_lines = {}
    current_line_label = None

    with xopen.xopen(schedule_path, "r") as f:
        for event, elem in ET.iterparse(f, events=["start", "end"]):
            if event == "start":
                if elem.tag == "transitLine":
                    current_line_label = elem.attrib.get("name") or elem.attrib["id"]
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
# Streaming events parsing
# ---------------------------------------------------------------------------

def count_pt_stop_segment_riders(events_path, rider_group_ids, vehicle_lines):
    """rider_group_ids: {flow_column_name: set_of_person_ids} - e.g.
    RIDER_GROUP_FLOW_COLUMNS mapped to their id sets. Returns total_counts,
    {flow_column_name: Counter} (one per rider_group_ids entry), and
    lines_by_segment."""
    total_counts = Counter()
    group_counts = {group: Counter() for group in rider_group_ids}
    lines_by_segment = defaultdict(set)

    onboard = defaultdict(set)
    last_departure_facility = {}

    processed = 0
    with xopen.xopen(events_path, "r") as f:
        parser = ET.iterparse(f, events=["start", "end"])
        _, root = next(parser)  # <events> start tag, kept to periodically release finished events

        for event, elem in parser:
            if event != "end" or elem.tag != "event":
                continue

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                logger.info("Processed %d events...", processed)

            etype = elem.attrib.get("type")

            if etype == PERSON_ENTERS_VEHICLE_EVENT:
                onboard[elem.attrib["vehicle"]].add(elem.attrib["person"])
            elif etype == PERSON_LEAVES_VEHICLE_EVENT:
                onboard[elem.attrib["vehicle"]].discard(elem.attrib["person"])
            elif etype == VEHICLE_DEPARTS_EVENT:
                last_departure_facility[elem.attrib["vehicle"]] = elem.attrib["facility"]
            elif etype == VEHICLE_ARRIVES_EVENT:
                vehicle = elem.attrib["vehicle"]
                arrival_facility = elem.attrib["facility"]
                departure_facility = last_departure_facility.get(vehicle)

                if departure_facility is not None and departure_facility != arrival_facility:
                    riders = onboard.get(vehicle, ())
                    if riders:
                        hour = int(float(elem.attrib["time"]) // 3600) % 24
                        segment = (departure_facility, arrival_facility)
                        key = (*segment, hour)

                        total_counts[key] += len(riders)
                        for group, ids in rider_group_ids.items():
                            n = sum(1 for person in riders if person in ids)
                            if n:
                                group_counts[group][key] += n

                        line = vehicle_lines.get(vehicle)
                        if line:
                            lines_by_segment[segment].add(line)

            elem.clear()
            if processed % PROGRESS_EVERY == 0:
                root.clear()

    logger.info("Finished streaming %d events from %s", processed, events_path)
    return total_counts, group_counts, lines_by_segment


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_hourly_dataframe(total_counts, group_counts):
    rows = [
        dict(
            from_stop=from_stop, to_stop=to_stop, hour=hour, total_flow=count,
            **{group: counts.get((from_stop, to_stop, hour), 0) for group, counts in group_counts.items()},
        )
        for (from_stop, to_stop, hour), count in total_counts.items()
    ]
    df = pd.DataFrame(
        rows, columns=["from_stop", "to_stop", "hour", "total_flow", *RIDER_GROUP_FLOW_COLUMNS],
    )
    return df.sort_values(["from_stop", "to_stop", "hour"]).reset_index(drop=True)


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


def plot_crossborder_pt_flow_map(df_daily, stop_coords, extent_path, output_path, stop_names=None, stop_totals=None):
    if not extent_path:
        logger.info("extent_path is not configured; skipping cross-border PT flow map.")
        return
    if not os.path.exists(extent_path):
        logger.warning("extent_path %s does not exist; skipping cross-border PT flow map.", extent_path)
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

    line_fields = [
        "from_stop_name", "to_stop_name", "lines", "total_flow",
        *[field for column in RIDER_GROUP_FLOW_COLUMNS for field in (column, share_column(column))],
        "from_stop_total_activity", "to_stop_total_activity",
    ]
    path_data = Plotter._prepare_path_data(segments.to_crs(epsg=4326), line_fields)
    path_data["_tooltip_html"] = path_data.apply(
        lambda row: Plotter._build_tooltip_html(row, line_fields), axis=1,
    )

    # Points at every stop, sized/colored by the same three metrics as the
    # lines - see analysis/pt_passenger_counts/TPG_comparison/interactive_map.py's
    # CircleMarker-per-stop maps for the display this is modeled on (a
    # marker per stop, sized by sqrt(ridership), colored by the metric).
    stops = aggregate_stop_level(
        df_daily, stop_coords,
        stop_names if stop_names is not None else {},
        stop_totals if stop_totals is not None else pd.Series(dtype=float),
    )
    point_gdf = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops.x, stops.y), crs=STOP_COORDINATE_CRS,
    )
    point_gdf = point_gdf[point_gdf.intersects(region)].to_crs(epsg=4326)

    point_fields = [
        "stop_name", "total_flow",
        *[field for column in RIDER_GROUP_FLOW_COLUMNS for field in (column, share_column(column))],
        "official_total_activity",
    ]
    point_data = point_gdf[point_fields].copy()
    # get_position needs one column of [lon, lat] pairs, not two separate
    # column names - passing ["lon", "lat"] as get_position (two column
    # names) is not a valid accessor and was rendering every point huge.
    point_data["coordinates"] = list(zip(point_gdf.geometry.x, point_gdf.geometry.y))
    point_data["_tooltip_html"] = point_data.apply(
        lambda row: Plotter._build_tooltip_html(row, point_fields), axis=1,
    )

    metric_maxima = {}
    line_layers = []
    point_layers = []
    for metric in METRICS:
        key = metric["key"]
        metric_maxima[key] = add_metric_columns(path_data, metric, size_scale=12, size_base=2)
        add_metric_columns(point_data, metric, size_scale=14, size_base=5)

        line_layers.append(pdk.Layer(
            "PathLayer",
            path_data,
            id=f"pt-line::{key}::0",
            visible=key == DEFAULT_METRIC,
            pickable=True,
            auto_highlight=True,
            get_path="path",
            get_width=f"{key}_size",
            get_color=f"{key}_color",
            highlight_color=[0, 200, 255],
            width_min_pixels=2,
            width_scale=3,
        ))
        point_layers.append(pdk.Layer(
            "ScatterplotLayer",
            point_data,
            id=f"pt-point::{key}::0",
            visible=key == DEFAULT_METRIC,
            pickable=True,
            auto_highlight=True,
            get_position="coordinates",
            get_radius=f"{key}_size",
            get_fill_color=f"{key}_color",
            get_line_color=[50, 50, 50],
            line_width_min_pixels=1,
            stroked=True,
            # Plain pdk.types.String, not a bare Python string: pydeck
            # auto-wraps a bare string kwarg value as a "@@=<value>" data
            # accessor (right for get_color="some_column", wrong for a
            # literal deck.gl enum like this) - a bare "pixels" here silently
            # became the accessor "d => d.pixels" (always undefined), which
            # is what made every point render at a wildly wrong huge size.
            radius_units=pdk.types.String("pixels"),
            radius_min_pixels=4,
            radius_max_pixels=20,
        ))

    extent_layer = _polygon_outline_layer(region, segments.crs, color=[0, 0, 0], width=2)

    extent_bounds_ll = gpd.GeoSeries([region], crs=segments.crs).to_crs(epsg=4326).total_bounds
    minx, miny, maxx, maxy = extent_bounds_ll
    margin_x, margin_y = (maxx - minx) * 0.05, (maxy - miny) * 0.05
    view_state = compute_view([
        [minx - margin_x, miny - margin_y],
        [maxx + margin_x, maxy + margin_y],
    ])
    view_state.pitch = 0

    tooltip = {
        "html": "{_tooltip_html}",
        "style": {"color": "white", "background-color": "black", "padding": "10px"},
    }
    deck = pdk.Deck(
        layers=[extent_layer, *line_layers, *point_layers],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    )

    output_file = os.path.join(output_path, "cross_border_pt_flow_map.html")
    deck.to_html(output_file, notebook_display=False)
    inject_metric_dropdown_legend(
        output_file, METRICS, metric_maxima, DEFAULT_METRIC,
        layer_prefixes=["pt-line::", "pt-point::"], dom_id_prefix="pt",
        extra_legend_lines=[("Black outline = scenario extent", "color:#222;")],
    )
    logger.info("Saved cross-border PT flow map to %s", output_file)
