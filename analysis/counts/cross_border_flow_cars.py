import glob
import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import shapely
from shapely.ops import unary_union

from . import flow_map_page
from analysis.output_files import find_simulation_output
from .flow_metrics import share_column
from .cross_border_flow_events import GROUPS as EVENT_GROUPS
from .paths import (
    configure_flow_map_extents, configure_simulation_path, get_analysis_output_path, get_flow_map_extents,
)

logger = logging.getLogger("synpp")

ROAD_TYPES_FOR_MAP         = ["motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"]
MIN_SPEED_KMH_FOR_FLOW_MAP = 50
RIDER_GROUP_FLOW_COLUMNS   = ["crossborder_flow", "swiss_crossborder_flow", "french_flow"]


def configure(context):
    context.stage("analysis.counts.cross_border_flow_events")
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")
    context.config("only_weekday", default=True)
    context.config("input_downsampling")
    context.config("cross_border_flow_min_daily_count", default = 20)
    # Roads outside Switzerland are drawn up to this distance (m) from the extent of the map
    context.config("cross_border_flow_foreign_distance", default = 50_000)

    configure_simulation_path(context)
    configure_flow_map_extents(context)


def execute(context):
    output_path = get_analysis_output_path(context)
    os.makedirs(output_path, exist_ok=True)

    # the events are read once for the car and PT maps (analysis.counts.cross_border_flow_events)
    df_hourly = pd.read_csv(context.stage("analysis.counts.cross_border_flow_events")["car_hourly"], dtype = {"link_id": str})
    df_hourly = df_hourly.rename(columns = {"total": "total_flow", **dict(zip(EVENT_GROUPS, RIDER_GROUP_FLOW_COLUMNS))})
    logger.info("Car entries on %d distinct links", df_hourly.link_id.nunique())

    sample_size = context.config("input_downsampling")
    for column in ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]:
        df_hourly[column] = df_hourly[column] / sample_size

    df_daily = aggregate_daily(df_hourly)

    network  = context.stage("analysis.counts.matching.network")
    df_daily = attach_road_attributes(df_daily, network)

    hourly_csv = os.path.join(output_path, "cross_border_link_flow_hourly.csv")
    daily_csv  = os.path.join(output_path, "cross_border_link_flow_daily.csv")
    df_hourly.to_csv(hourly_csv, index = False)
    df_daily.to_csv(daily_csv, index = False)

    min_daily_count = context.config("cross_border_flow_min_daily_count")
    df_plot = df_daily[df_daily.total_flow >= min_daily_count].copy()

    swiss_border = context.stage("data.spatial.swiss_border")
    geometry = network.straight_geometry()
    for name, extent_path in get_flow_map_extents(context).items():
        plot_crossborder_flow_map(
            df_plot, network, extent_path, swiss_border, output_path, name = name, geometry = geometry,
            foreign_distance = context.config("cross_border_flow_foreign_distance"),
        )

    return dict(done=True, path=output_path, daily_csv=daily_csv, hourly_csv=hourly_csv)


def _find_output_file(simulation_path, filename):
    return find_simulation_output(simulation_path, filename)


def load_cross_border_person_ids(persons_path):
    df = pd.read_csv(
        persons_path, sep = ";", usecols = ["person", "subpopulation"],
        dtype = {"person": str}, low_memory = False,
    )
    return set(df.loc[df.subpopulation == "crossborder", "person"])


def load_swiss_resident_cross_border_person_ids(persons_path):
    df = pd.read_csv(
        persons_path, sep = ";", usecols = ["person", "subpopulation", "crossBorderOD"],
        dtype = {"person": str}, low_memory = False,
    )
    is_swiss_resident = df.subpopulation.eq("crossborder") & df.crossBorderOD.str.startswith("CH", na = False)
    return set(df.loc[is_swiss_resident, "person"])


def load_french_resident_person_ids(persons_path):
    df = pd.read_csv(
        persons_path, sep = ";", usecols = ["person", "isExternalFR"],
        dtype = {"person": str}, low_memory = False,
    )
    is_french_resident = df.isExternalFR.astype(str).str.lower().eq("true")
    return set(df.loc[is_french_resident, "person"])


def aggregate_daily(df_hourly):
    df_daily = df_hourly.groupby("link_id", as_index=False)[
        ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]
    ].sum()

    for column in RIDER_GROUP_FLOW_COLUMNS:
        df_daily[share_column(column)] = np.where(
            df_daily.total_flow > 0, 100.0 * df_daily[column] / df_daily.total_flow, np.nan
        )

    return df_daily.sort_values("total_flow", ascending = False).reset_index(drop = True)


def attach_road_attributes(df_daily, network):
    links = network.links[["link_id", "highway", "permlanes"]].copy()
    links["link_id"] = links["link_id"].astype(str)
    df_daily = df_daily.copy()
    df_daily["link_id"] = df_daily["link_id"].astype(str)
    return df_daily.merge(links, on="link_id", how="left")


def plot_share_by_road_type(df_daily, output_path):
    df = df_daily[df_daily.highway.notna()].copy()
    if df.empty:
        logger.info("No links with highway information; skipping road-type bar chart.")
        return

    df_avg = df.groupby("highway", as_index = False).agg(
        total_flow       = ("total_flow", "mean"),
        crossborder_flow = ("crossborder_flow", "mean"),
        n_links          = ("link_id", "nunique"),
    )

    df_avg["crossborder_share_pct"] = 100.0 * df_avg.crossborder_flow / df_avg.total_flow
    df_avg = df_avg.sort_values("total_flow", ascending = False).reset_index(drop = True)

    _, ax = plt.subplots(figsize = (12, 6))
    x = np.arange(len(df_avg))
    width = 0.38

    ax.bar(x - width / 2, df_avg.total_flow, width, label="Total flow", color="#4c72b0")
    ax.bar(x + width / 2, df_avg.crossborder_flow, width, label="Cross-border flow", color="#dd8452")

    for i, row in df_avg.iterrows():
        ax.text(
            i, max(row.total_flow, row.crossborder_flow) * 1.02,
            f"{row.crossborder_share_pct:.0f}% (n={row.n_links})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df_avg.highway, rotation=30, ha="right")
    ax.set_ylabel("Average daily flow per link (vehicles/day)")
    ax.set_title("Average Link Flow by Road Type: Total vs. Cross-Border Agents")
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_file = os.path.join(output_path, "cross_border_share_by_road_type.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved road-type breakdown to %s", output_file)


def plot_share_map(df_daily, network, output_path):
    geometry = network.get_ways(geo_df=network.straight_geometry(), road_types=ROAD_TYPES_FOR_MAP)
    if geometry.empty:
        logger.info("No major-road geometry available; skipping cross-border share map.")
        return

    geometry = geometry.copy()
    geometry["link_id"] = geometry["link_id"].astype(str)

    merged = geometry.merge(df_daily[["link_id", "total_flow", "crossborder_share_pct"]], on="link_id", how="inner")
    if merged.empty:
        logger.info("No major-road links have cross-border flow data; skipping map.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    geometry.plot(ax=ax, color="#dddddd", linewidth=0.5, zorder=1)

    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)
    merged.plot(
        ax=ax, column="crossborder_share_pct", cmap="inferno_r",
        linewidth=merged.total_flow.clip(upper=merged.total_flow.quantile(0.95)) / merged.total_flow.max() * 3 + 0.5,
        legend=True, legend_kwds={"label": "Cross-border share of daily flow (%)", "shrink": 0.6},
        zorder=2,
    )

    ax.set_title("Simulated Cross-Border Share of Traffic on Major Roads")
    ax.set_axis_off()
    plt.tight_layout()

    output_file = os.path.join(output_path, "cross_border_share_map.png")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved cross-border share map to %s", output_file)


# Same set of rider-group layers, keys and colors as cross_border_flow_pt.py's
# METRICS, for visual consistency between the car and PT flow maps.
METRICS = [
    {
        "key": "total_flow", "label": "Total car flow (vehicles/day)",
        "low": (225, 225, 225), "high": (35, 139, 69), "gamma": 0.5, "width_basis": "total_flow",
    },
    {
        "key": "crossborder_flow", "label": "Cross-border car flow (vehicles/day, non-Swiss residents)",
        "low": (225, 225, 225), "high": (8, 29, 225), "gamma": 0.5, "width_basis": "crossborder_flow",
    },
    {
        "key": "crossborder_share_pct", "label": "Cross-border share of car traffic (%, non-Swiss residents)",
        "low": (225, 225, 225), "high": (230, 85, 13), "gamma": 1.0, "width_basis": "total_flow",
    },
    {
        "key": "swiss_crossborder_flow", "label": "Swiss-resident cross-border car flow (vehicles/day)",
        "low": (225, 225, 225), "high": (0, 128, 128), "gamma": 0.5, "width_basis": "swiss_crossborder_flow",
    },
    {
        "key": "swiss_crossborder_share_pct", "label": "Swiss-resident cross-border share of car traffic (%)",
        "low": (225, 225, 225), "high": (166, 86, 40), "gamma": 1.0, "width_basis": "total_flow",
    },
    {
        "key": "french_flow", "label": "France-resident car flow (vehicles/day)",
        "low": (225, 225, 225), "high": (117, 25, 128), "gamma": 0.5, "width_basis": "french_flow",
    },
    {
        "key": "french_share_pct", "label": "France-resident share of car traffic (%)",
        "low": (225, 225, 225), "high": (203, 24, 29), "gamma": 1.0, "width_basis": "total_flow",
    },
]
DEFAULT_METRIC = "crossborder_flow"


def plot_crossborder_flow_map(df_daily, network, extent_path, swiss_border, output_path, name = None, geometry = None,
                              foreign_distance = 50_000):
    """Writes cross_border_flow_map_<name>.html: the flows inside the extent, and outside Switzerland up to
    foreign_distance metres from it. geometry: network.straight_geometry(), if already computed."""
    if not os.path.exists(extent_path):
        logger.warning("Extent %s does not exist; skipping cross-border flow map %s.", extent_path, name)
        return

    boundary = gpd.read_file(extent_path)

    # Deliberately not network.get_geometry(): when export_detailed_network
    # is enabled that returns the pt2matsim detailed-geometry CSV, whose
    # link_id is the pre-simplification/pre-merge id, not the simulation
    # link_id that output_events (and therefore crossborder_flow) is keyed
    # by. The interior Swiss network goes through that simplification and
    # so silently fails to join there, while cross-border links added later
    # never got simplified and happen to join correctly - which is exactly
    # why only the cross-border side of the extent showed flow before this
    # fix. straight_geometry() gives straight node-to-node link
    # geometry, built from the same link table the flow counts use, so the
    # join is correct everywhere.
    if geometry is None:
        geometry = network.straight_geometry()

    if boundary.crs != geometry.crs:
        boundary = boundary.to_crs(geometry.crs)
    # MMT_extent_cleaned.gpkg carries a Z coordinate (e.g. an elevation value
    # on every vertex); force_2d drops it so this polygon and the swiss_border
    # one below (from swissBOUNDARIES3D, 3D by construction) don't end up
    # mixing 3D and 2D geometry with the flat 2D flow-line data on the same
    # map, which is what was actually breaking the render.
    region = shapely.force_2d(unary_union(boundary.geometry))

    # The extent only cuts off the Swiss side; the foreign side is shown up to
    # foreign_distance from the extent. So a link is kept if it's inside
    # the extent OR it isn't in Switzerland at all - only interior-CH links
    # beyond the extent get dropped.
    swiss_border_full = shapely.force_2d(unary_union(gpd.GeoSeries(swiss_border).to_crs(geometry.crs)))

    links = network.links[["link_id", "freespeed"]].copy()
    links["link_id"] = links["link_id"].astype(str)
    fast_link_ids = set(links.loc[links.freespeed >= MIN_SPEED_KMH_FOR_FLOW_MAP / 3.6, "link_id"])

    # the cheap filters first (fast links with flow, close to the extent), then the exact geometric ones
    geometry = geometry[geometry.link_id.isin(fast_link_ids) & geometry.link_id.isin(set(df_daily.link_id))]
    geometry = geometry[geometry.intersects(region.buffer(foreign_distance))]
    geometry = geometry[geometry.intersects(region) | ~geometry.intersects(swiss_border_full)]
    if geometry.empty:
        logger.info(
            "No roads with freespeed >= %d km/h found inside the scenario extent; skipping cross-border flow map.",
            MIN_SPEED_KMH_FOR_FLOW_MAP,
        )
        return

    flow_columns = ["total_flow", *RIDER_GROUP_FLOW_COLUMNS]
    share_columns = [share_column(c) for c in RIDER_GROUP_FLOW_COLUMNS]
    merged = geometry.merge(df_daily[["link_id", *flow_columns, *share_columns]], on="link_id", how="left")
    merged[flow_columns] = merged[flow_columns].fillna(0.0)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry.crs)

    # A zero-total_flow link renders as an uninformative gray line either
    # way (every metric's color_for treats <=0 as gray) - dropping it here
    # keeps the map's weight down without losing anything visible. This is
    # the majority of the French-side geometry kept above "as far as the
    # network extends" regardless of whether it carries any recorded flow.
    merged = merged[merged.total_flow > 0].copy()
    if merged.empty:
        logger.info("No flow recorded on qualifying roads inside the scenario extent; skipping map.")
        return

    # Compact page (flow_map_page): the data once, the colors/widths/tooltips of the selected metric computed in
    # the browser; only links with flow are written (zero-flow links would render as uninformative gray lines)
    merged = merged.to_crs(epsg=4326)
    flows = {column: merged[column].to_numpy() for column in flow_columns}
    tooltip = [("Link", "link_id", "text"), ("Total car flow", "total_flow", "int")] + [
        (label, key, "pct" if key.endswith("_share_pct") else "int")
        for key, label in ((m["key"], m["label"]) for m in METRICS) if key != "total_flow"
    ]
    flow_layer = flow_map_page.layer(
        "path", [list(g.coords) for g in merged.geometry], flows, texts={"link_id": merged.link_id.tolist()},
        tooltip=tooltip, size_scale=16, size_base=3, metrics=METRICS,
    )

    # Only the part of the national border that actually runs through the extent is relevant here
    swiss_border_geometry = swiss_border_full.intersection(region)
    outlines = [(_outline_rings(region, geometry.crs), [0, 0, 0], 2)]
    if not swiss_border_geometry.is_empty:
        outlines.insert(0, (_outline_rings(swiss_border_geometry, geometry.crs), [178, 24, 43], 1.5))

    output_file = os.path.join(output_path, f"cross_border_flow_map_{name}.html" if name else "cross_border_flow_map.html")
    flow_map_page.write_page(
        output_file, f"Cross-border car flows{' - ' + name if name else ''}", [flow_layer], METRICS, DEFAULT_METRIC,
        outlines=outlines, bounds=_extent_bounds(region, geometry.crs),
        legend_lines=[
            ("Red outline = Swiss national border", "color:#b2182b;"),
            ("Black outline = scenario extent", "color:#222;"),
        ],
    )
    logger.info("Saved cross-border flow map of %s to %s", name, output_file)


def _outline_rings(geometry, crs, tolerance=20.0):
    """Exterior rings (lon, lat) of the polygons of geometry, simplified to tolerance metres"""
    geometry = gpd.GeoSeries([geometry.simplify(tolerance)], crs=crs).to_crs(epsg=4326).iloc[0]
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    return [list(part.exterior.coords) for part in parts if part.geom_type == "Polygon"]


def _extent_bounds(region, crs, margin=0.05):
    """(min lon, min lat, max lon, max lat) of the extent, with a margin"""
    minx, miny, maxx, maxy = gpd.GeoSeries([region], crs=crs).to_crs(epsg=4326).total_bounds
    dx, dy = (maxx - minx) * margin, (maxy - miny) * margin
    return minx - dx, miny - dy, maxx + dx, maxy + dy


def _polygon_outline_layer(geometry, crs, color, width):
    """A stroke-only PolygonLayer for a (Multi)Polygon boundary. Also handles
    a GeometryCollection, e.g. from intersecting a border with an extent
    polygon can leave slivers of LineString/Point alongside the real
    Polygon parts - those non-polygon slivers are dropped since they have no
    exterior to outline."""

    if geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [part for part in geometry.geoms if part.geom_type == "Polygon"]
    else:
        parts = [geometry]
    gdf = gpd.GeoDataFrame(geometry=parts, crs=crs).to_crs(epsg=4326)
    gdf["coordinates"] = gdf.geometry.exterior.apply(lambda x: list(x.coords))
    return pdk.Layer(
        "PolygonLayer",
        gdf,
        get_polygon="coordinates",
        get_fill_color=[0, 0, 0, 0],
        stroked=True,
        get_line_color=color,
        line_width_min_pixels=width,
        pickable=False,
    )


