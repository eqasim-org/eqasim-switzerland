"""
Stage: analysis.counts.cross_border_theoretical_comparison

One counts_on_network_{city}.html-style map per extent (Switzerland/Geneva/
Annemasse, via analysis.counts.runs.ch/.geneva/.annemasse), with
crossborder_share_pct (from analysis.counts.cross_border_flow) added as a
selectable point-color metric alongside pdiff/adiff/geh. "Theoretical" means
the real, non-MATSim count-station data already used in analysis/counts, not
the A+GQPV survey data in data/cross_border.

Written into the shared get_analysis_output_path folder. A city's map is
skipped if its analysis.counts.runs.<city> stage returns no data (e.g.
Geneva needs only_weekday=true, Annemasse needs
include_external_population=true).
"""

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from .matching.plots import Plotter
from .paths import configure_simulation_path, get_analysis_output_path

logger = logging.getLogger("synpp")

# Maps the output label (and file name) to the analysis.counts.runs.<x> stage
# that already matches that city's count stations against the network.
CITY_RUNS = {
    "switzerland": "ch",
    "geneva": "geneva",
    "annemasse": "annemasse",
}

ROAD_TYPES_FOR_MAP = ["motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"]

# matsim_car_flow / matsim_crossborder_flow are computed (attach_crossborder_share)
# but intentionally not surfaced on the map - only their ratio is. Since this
# column is present, Plotter._flow_metric_settings also makes it a selectable
# point-color metric alongside pdiff/adiff/geh, not just a tooltip field.
CROSSBORDER_TOOLTIP_FIELDS = ["crossborder_share_pct"]


def configure(context):
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.cross_border_flow")
    context.stage("data.spatial.swiss_border")
    configure_simulation_path(context)
    context.config("only_weekday", default=False)

    for run_name in CITY_RUNS.values():
        context.stage(f"analysis.counts.runs.{run_name}")


def execute(context):
    output_path = get_analysis_output_path(context)
    os.makedirs(output_path, exist_ok=True)

    network = context.stage("analysis.counts.matching.network")

    daily_csv = context.stage("analysis.counts.cross_border_flow")["daily_csv"]
    df_cb = pd.read_csv(daily_csv, dtype={"link_id": str})
    total_flow_by_link = dict(zip(df_cb.link_id, df_cb.total_flow))
    crossborder_flow_by_link = dict(zip(df_cb.link_id, df_cb.crossborder_flow))

    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))

    saved_maps = {}
    for label, run_name in CITY_RUNS.items():
        path_to_stations = context.stage(f"analysis.counts.runs.{run_name}")

        if path_to_stations is None:
            logger.warning(
                "analysis.counts.runs.%s returned no station data (check its pipeline config); "
                "skipping the %s map.", run_name, label,
            )
            continue

        df_stations = pd.read_pickle(path_to_stations)
        df_stations = attach_crossborder_share(df_stations, network, total_flow_by_link, crossborder_flow_by_link)

        output_file = os.path.join(output_path, f"cross_border_comparison_{label}.html")
        build_comparison_map(df_stations, network, border, output_file)
        saved_maps[label] = output_file
        logger.info("Saved %s comparison map to %s", label, output_file)

    return dict(done=True, path=output_path, maps=saved_maps)


# ---------------------------------------------------------------------------
# Cross-border share per station
# ---------------------------------------------------------------------------

def attach_crossborder_share(df_stations, network, total_flow_by_link, crossborder_flow_by_link):
    """For every station, sum this stage's own car-only daily flow (and its
    cross-border subset) over the simulation links matched to that station -
    the same matched-link-id -> simulation-link-id resolution
    analysis.counts.matching.compare.compare_flow_total_efficient uses."""

    exploded = df_stations[["id", "link_id"]].explode("link_id")
    exploded["link_id"] = exploded["link_id"].astype(str)
    exploded["sim_link"] = network.get_in_simulation_links(exploded["link_id"].tolist())

    exploded["matsim_car_flow"] = exploded["sim_link"].map(total_flow_by_link).fillna(0.0)
    exploded["matsim_crossborder_flow"] = exploded["sim_link"].map(crossborder_flow_by_link).fillna(0.0)

    aggregated = exploded.groupby("id")[["matsim_car_flow", "matsim_crossborder_flow"]].sum().reset_index()
    aggregated["crossborder_share_pct"] = np.where(
        aggregated.matsim_car_flow > 0,
        100.0 * aggregated.matsim_crossborder_flow / aggregated.matsim_car_flow,
        np.nan,
    ).round(1)

    return df_stations.merge(aggregated, on="id", how="left")


# ---------------------------------------------------------------------------
# Map building - same layout as analysis.counts.runs.ch/.geneva/.annemasse
# ---------------------------------------------------------------------------

def build_comparison_map(df_stations, network, border, output_path):
    """Mirrors the counts_on_network_{city}.html maps built by
    analysis.counts.runs.*: same Plotter.create_map call (network ways +
    matched links as PathLayers, count stations as the point layer with the
    pdiff/adiff/geh color switcher), only adding the cross-border columns to
    the point tooltip. matched_links is rebuilt from the matched link ids
    already stored per station (link_id) instead of re-running the matcher,
    since analysis.counts.runs.* only returns the merged results pickle."""

    df_stations = gpd.GeoDataFrame(df_stations, geometry="geometry", crs="EPSG:2056")

    matched_ids = df_stations["link_id"].explode().dropna().astype(str).unique().tolist()
    matched_links = network.get_link_geometries(matched_ids, expand_merged=True)

    point_gdf = Plotter.prepare_flow_map_points(df_stations, df_stations).to_crs(epsg=4326)
    point_gdf = point_gdf.merge(df_stations[["id", *CROSSBORDER_TOOLTIP_FIELDS]], on="id", how="left")

    Plotter.create_map(
        [network.get_ways(road_types=ROAD_TYPES_FOR_MAP).to_crs(epsg=4326), matched_links.to_crs(epsg=4326)],
        data_to_show=["link_id"],
        point_gdf=[point_gdf],
        point_data_to_show=Plotter.FLOW_MAP_TOOLTIP_FIELDS + CROSSBORDER_TOOLTIP_FIELDS,
        border=border,
        path_to_save=output_path,
    )
