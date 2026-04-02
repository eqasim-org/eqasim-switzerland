import os
import logging
import pandas as pd

from .matching.network import Network
from .matching.counts import Counts
from .matching.matcher import TrafficDataMatcher
from .run_utils import filter_data, save_as_target

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("matsim.simulation.prepare")

    context.stage("analysis.counts.cantons.aargau")
    context.stage("analysis.counts.cantons.bern")
    context.stage("analysis.counts.cantons.ch")
    context.stage("analysis.counts.cantons.geneva")
    context.stage("analysis.counts.cantons.luzern")
    context.stage("analysis.counts.cantons.saint_gallen")
    context.stage("analysis.counts.cantons.vaud")
    context.stage("analysis.counts.cantons.zurich")
    context.stage("data.spatial.swiss_border")

    context.config("only_weekday", default=False)
    context.config("output_prefix", "switzerland_")


def _load_counts_and_match(context, network, city):
    if city == "aargau":
        file_path = context.stage("analysis.counts.cantons.aargau")
        counts = Counts(
            file_path=file_path,
            id_column="objectid",
            columns_to_keep={"flow": "flow", "flow_w": "flow_w"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=10,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    elif city == "bern":
        if context.config("only_weekday"):
            return None
        file_path = context.stage("analysis.counts.cantons.bern")
        counts = Counts(
            file_path=file_path,
            id_column="objectid",
            columns_to_keep={"flow": "flow"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=15,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=False,
        )
    elif city == "ch":
        count_stations_file, counts_data_file, year = context.stage("analysis.counts.cantons.ch")
        counts = Counts(
            counts_data_file,
            count_stations_file,
            include_incomplete_data=True,
            minimum_months=6,
            context=context,
            year=year,
        )
        match_kwargs = dict(
            search_radius=80,
            get_pairs=True,
            by_highway_order=True,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    elif city == "geneva":
        file_path = context.stage("analysis.counts.cantons.geneva")
        counts = Counts(
            file_path=file_path,
            columns_to_keep={
                "flow": "flow",
                "flow_w": "flow_w",
                "ANGLE": "angle",
                "quantile_lower_flow_weekday": "flow_lower",
                "quantile_upper_flow_weekday": "flow_upper",
                "max_flow": "flow_max",
            },
            context=context,
        )
        match_kwargs = dict(
            search_radius=20,
            get_pairs=False,
            by_highway_order=False,
            direction_from_osm=False,
        )
    elif city == "luzern":
        if context.config("only_weekday"):
            return None
        file_path = context.stage("analysis.counts.cantons.luzern")
        counts = Counts(
            file_path=file_path,
            id_column="objectid",
            columns_to_keep={"flow": "flow"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=10,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    elif city == "saint_gallen":
        if context.config("only_weekday"):
            return None
        file_path = context.stage("analysis.counts.cantons.saint_gallen")
        counts = Counts(
            file_path=file_path,
            id_column="objectid",
            columns_to_keep={"flow": "flow"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=2,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    elif city == "vaud":
        file_path = context.stage("analysis.counts.cantons.vaud")
        counts = Counts(
            file_path=file_path,
            id_column="id",
            columns_to_keep={"TJM": "flow", "TJOM": "flow_w"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=10,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    elif city == "zurich":
        file_path = context.stage("analysis.counts.cantons.zurich")
        counts = Counts(
            file_path=file_path,
            id_column="id",
            columns_to_keep={"flow": "flow", "flow_w": "flow_w"},
            context=context,
        )
        match_kwargs = dict(
            search_radius=10,
            get_pairs=True,
            by_highway_order=False,
            direction_from_osm=False,
            only_two_link_ids=True,
        )
    else:
        raise ValueError(f"Unsupported city {city}")

    matcher = TrafficDataMatcher(city, cache=context.path())
    matched = matcher.match(network=network, counts=counts, **match_kwargs)

    flow_col = "flow_w" if context.config("only_weekday") and "flow_w" in counts.counts.columns else "flow"
    station_flows = counts.counts[["id", flow_col]].rename(columns={flow_col: "flow"})

    grouped = matched.groupby("id", as_index=False).agg({
        "link_id": list,
        "distance": list,
    })
    grouped["city"] = city

    return station_flows.merge(grouped, on="id", how="inner")


def execute(context):
    logger.info("Preparing calibration target from authority counts before simulation...")

    network_file = os.path.join(
        context.path("matsim.simulation.prepare"),
        f"{context.config('output_prefix')}network.xml.gz",
    )
    assert os.path.exists(network_file), f"Prepared network not found at {network_file}"

    network = Network(network_file=network_file, geometry_file=None, overwrite=True, cache_dir=context.path())

    city_order = [
        "aargau",
        "bern",
        "ch",
        "geneva",
        "luzern",
        "saint_gallen",
        "vaud",
        "zurich",
    ]

    dfs = []
    for city in city_order:
        try:
            city_df = _load_counts_and_match(context, network, city)
            if city_df is None or city_df.empty:
                logger.info(f"\t - {city}: No usable records, skipping.")
                continue
            logger.info(f"\t - {city}: {len(city_df)} matched stations")
            dfs.append(city_df)
        except Exception as e:
            logger.warning(f"\t - {city}: Failed to build pre-simulation matches ({e}), skipping.")

    if not dfs:
        raise RuntimeError("No counts could be matched to the prepared network.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined matched dataset for calibration: {len(df)} records")

    df = filter_data(df, network, require_simulated=False)
    if df is None or df.empty:
        raise RuntimeError("No records left after filtering pre-simulation matched counts.")

    save_as_target(network, df, context.path())

    target_file = os.path.join(context.path(), "target_flow.csv")
    assert os.path.exists(target_file), f"Target file was not created at {target_file}"
    return target_file