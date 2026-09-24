"""Compare Transcality directional counts with MATSim flows."""

import copy
import json
import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..matching.counts import Counts
from ..matching.matcher import TrafficDataMatcher
from ..matching.plots import Plotter
from ..matching.results import save_count_results
from ..paths import configure_simulation_path, get_analysis_output_path, matches_found


logger = logging.getLogger("synpp")


ROAD_TYPES_TO_SHOW = [
    "motorway",
    "trunk",
    "primary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary",
    "secondary_link",
    "tertiary",
]


def configure(context):
    context.stage("analysis.counts.cantons.transcality")
    context.stage("analysis.counts.matching.network")
    context.stage("analysis.counts.matching.compare")
    context.stage("data.spatial.swiss_border")
    context.config("input_downsampling")
    context.config("only_weekday", default=False)
    configure_simulation_path(context)


def execute(context):
    city = "transcality"
    sample_size = context.config("input_downsampling")
    output_directory = get_analysis_output_path(context)
    os.makedirs(output_directory, exist_ok=True)

    counts = Counts(
        file_path=context.stage("analysis.counts.cantons.transcality"),
        id_column="OBJECTID",
        columns_to_keep={
            "flow": "flow",
            "flow_lower": "flow_lower",
            "flow_upper": "flow_upper",
            "osm_id": "osm_id",
            "angle": "angle",
            "source": "source",
            "detector_ids": "detector_ids",
            "profile_json": "profile_json",
        },
        context=context,
    )
    network = context.stage("analysis.counts.matching.network")
    matched = TrafficDataMatcher().match(network=network, counts=counts, mode="directional")
    if not matches_found(matched, city):
        return None

    # Group on the active simulation link, including its detailed replicates.
    audit = matched[["id", "link_id", "distance"]].copy()
    audit["simulation_link_id"] = network.get_in_simulation_links(audit["link_id"])
    audit.merge(
        counts.counts.drop(columns="geometry"), on="id", validate="one_to_one"
    ).to_csv(os.path.join(context.path(), "detector_matches.csv"), index=False)
    counts, matched = aggregate_counts_by_link(counts, matched, network)

    comparison = context.stage("analysis.counts.matching.compare")
    flows = comparison.compare_flow_total_efficient(
        counts,
        matched,
        network,
        sample_size=sample_size,
        get_average=False,
        flow_col="flow",
    )
    if not matches_found(flows, city, source="simulation flow results"):
        return None

    metadata_columns = [
        "id",
        "flow_lower",
        "flow_upper",
        "source",
        "detector_ids",
        "profile_json",
        "station_ids",
        "detector_count",
    ]
    flows = flows.merge(counts.counts[metadata_columns], on="id", how="left")
    flows["bounds_status"] = np.select(
        [
            flows["simulated_flow"] < flows["flow_lower"],
            flows["simulated_flow"] > flows["flow_upper"],
        ],
        ["Below lower bound", "Above upper bound"],
        default="Within bounds",
    )

    plotter = Plotter()
    plotter.plot_network_with_counts(
        counts,
        matched,
        network,
        output=os.path.join(output_directory, f"{city}_network.png"),
        lw=0.7,
        markersize=4,
        figsize=(40, 40),
        road_types="all",
        cut=True,
        highlight_stations=[],
        return_matched_links=False,
    )

    result_path = save_count_results(city, matched, flows, context.path())
    plotter.plot_flow(
        flows=flows,
        counts=counts,
        distance_to_border=2000,
        title="Observed vs Simulated Traffic Flows (Transcality)",
        output_file=os.path.join(output_directory, "flow_comparaison_transcality.png"),
        show_range=True,
    )
    plotter.plot_flow_by_road_type(
        flows,
        network,
        matched,
        counts,
        distance_to_border=0,
        title="Average Observed vs Simulated Flow by Highway Type (Transcality)",
        output_file=os.path.join(output_directory, "flow_by_road_type_transcality.png"),
    )
    plotter.plot_flow_by_source(
        flows,
        output_file=os.path.join(output_directory, "flow_by_source_transcality.png"),
        title="Observed vs Simulated Daily Flow by Counting Source (Transcality)",
    )
    plotter.plot_flow_bounds(
        flows,
        output_file=os.path.join(output_directory, "flow_bounds_transcality.png"),
        title="Transcality simulated flows relative to observed Q25–Q75 bounds",
    )

    points = Plotter.prepare_flow_map_points(counts.counts, flows).merge(
        flows[
            [
                "id",
                "flow_lower",
                "flow_upper",
                "bounds_status",
                "source",
                "detector_ids",
                "profile_json",
            ]
        ],
        on="id",
        how="left",
    )
    points["daily_profile"] = points["profile_json"].map(Plotter.flow_profile_svg)
    points = gpd.GeoDataFrame(points, geometry="geometry", crs=counts.counts.crs).to_crs(
        epsg=4326
    )

    colored_links, legend = prepare_flow_colored_links(
        network,
        comparison.get_link_stats(),
        sample_size,
        ROAD_TYPES_TO_SHOW,
    )
    border = gpd.GeoDataFrame(
        context.stage("data.spatial.swiss_border").to_crs(epsg=4326)
    )
    Plotter.create_map(
        colored_links.to_crs(epsg=4326),
        data_to_show=["link_id", "simulated_flow"],
        point_gdf=[points],
        point_data_to_show=Plotter.TRANSCALITY_FLOW_MAP_TOOLTIP_FIELDS,
        border=border,
        cut_network=True,
        path_color_column="_flow_color",
        path_color_legend=legend,
        path_to_save=os.path.join(output_directory, "counts_on_network_transcality.html"),
    )
    return result_path


def prepare_flow_colored_links(network, link_stats, sample_size, road_types):
    """Attach full-scale simulated flows and viridis colors to real link shapes."""
    roads = network.get_ways(road_types=road_types).copy()
    roads["geometry_link_id"] = roads["link_id"].astype(str)
    roads["link_id"] = network.get_in_simulation_links(roads["geometry_link_id"])
    roads = roads.dropna(subset=["link_id"])
    roads["link_id"] = roads["link_id"].astype(str)

    simulated = link_stats[["link_id", "flow"]].copy()
    simulated["link_id"] = simulated["link_id"].astype(str)
    simulated["simulated_flow"] = pd.to_numeric(simulated["flow"], errors="coerce")
    if sample_size:
        simulated["simulated_flow"] /= sample_size
    simulated = simulated.groupby("link_id", as_index=False)["simulated_flow"].mean()
    roads = roads.merge(simulated, on="link_id", how="inner")
    roads = roads.dropna(subset=["simulated_flow"])
    if roads.empty:
        raise ValueError("No simulated link flows could be joined to network geometries.")

    upper = float(roads["simulated_flow"].quantile(0.95))
    if not np.isfinite(upper) or upper <= 0:
        upper = max(float(roads["simulated_flow"].max()), 1.0)
    colormap = plt.get_cmap("viridis")
    normalized = np.clip(roads["simulated_flow"].to_numpy() / upper, 0, 1)
    roads["_flow_color"] = [
        [int(red * 255), int(green * 255), int(blue * 255), 230]
        for red, green, blue, _ in colormap(normalized)
    ]
    roads["simulated_flow"] = roads["simulated_flow"].round(0).astype(int)
    legend = {
        "label": "Simulated link flow (vehicles/day; colors clipped at P95)",
        "lower": 0,
        "upper": upper,
        "colors": ["#440154", "#31688e", "#35b779", "#fde725"],
    }
    return gpd.GeoDataFrame(roads, geometry="geometry", crs=network.crs), legend


def aggregate_counts_by_link(counts, matched, network):
    """Sum detector observations once per directed simulation link.

    Bounds are summed envelopes, not quantiles of the daily total. A group
    remains directional regardless of how many detectors contributed to it.
    """
    if not counts.counts["id"].is_unique or not matched["id"].is_unique:
        raise ValueError("Transcality requires one observation and one directional match per detector.")
    observations = matched.merge(
        counts.counts.drop(columns="geometry"), on="id", validate="one_to_one"
    )
    observations["simulation_link_id"] = network.get_in_simulation_links(
        observations["link_id"]
    )
    if observations["simulation_link_id"].isna().any():
        raise ValueError("A Transcality match has no corresponding simulation link.")

    rows, matches = [], []
    for link_id, group in observations.groupby("simulation_link_id", sort=True):
        first = group.iloc[0]
        observation_id = f"transcality:link:{link_id}"
        profile = sum_profiles(group["profile_json"])
        rows.append({
            "id": observation_id,
            "geometry": first["geometry"],
            "flow": group["flow"].sum(),
            "flow_lower": group["flow_lower"].sum(),
            "flow_upper": group["flow_upper"].sum(),
            "source": " + ".join(sorted(group["source"].unique())),
            "detector_ids": ", ".join(group["detector_ids"].astype(str)),
            "station_ids": group["id"].tolist(),
            "detector_count": len(group),
            "profile_json": profile,
        })
        # Retain a representative detailed segment for geometry/road metadata;
        # Compare resolves it to the simulation link once, not once per detector.
        matches.append({
            "id": observation_id,
            "geometry": first["geometry"],
            "link_id": first["link_id"],
            "road_geometry": first["road_geometry"],
            "distance": first["distance"],
        })

    grouped_counts = copy.copy(counts)
    grouped_counts.count_data = gpd.GeoDataFrame(rows, geometry="geometry", crs=counts.counts.crs)
    grouped_counts.count_stations = grouped_counts.count_data[["id", "geometry"]].copy()
    grouped_matches = gpd.GeoDataFrame(matches, geometry="geometry", crs=matched.crs)
    logger.info("Aggregated %d Transcality detectors into %d directed simulation links.",
                len(observations), len(rows))
    return grouped_counts, grouped_matches


def sum_profiles(profiles):
    """Sum hourly profiles by hour, preserving their lower/upper envelopes."""
    totals = np.zeros((3, 24), dtype=float)
    for value in profiles:
        profile = json.loads(value)
        hours = np.asarray(profile["hour"])
        if sorted(hours.tolist()) != list(range(24)):
            raise ValueError("Each Transcality detector profile must contain hours 0–23 exactly once.")
        for index, field in enumerate(("flow", "lower", "upper")):
            values = np.asarray(profile[field], dtype=float)
            if values.shape != (24,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid Transcality hourly {field} profile.")
            totals[index, hours.astype(int)] += values
    return json.dumps({"hour": list(range(24)), **{
        field: totals[index].round(2).tolist()
        for index, field in enumerate(("flow", "lower", "upper"))
    }}, separators=(",", ":"))
