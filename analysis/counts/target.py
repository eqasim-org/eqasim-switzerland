import os
import logging
import pandas as pd

from .matching.network import RoadNetwork
from .matching.counts import Counts
from .matching.matcher import TrafficDataMatcher
from .run_utils import filter_data, save_as_target

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.counts.cantons.aargau")
    context.stage("analysis.counts.cantons.bern")
    context.stage("analysis.counts.cantons.ch")
    context.stage("analysis.counts.cantons.geneva")
    context.stage("analysis.counts.cantons.luzern")
    context.stage("analysis.counts.cantons.saint_gallen")
    context.stage("analysis.counts.cantons.vaud")
    context.stage("analysis.counts.cantons.zurich")
    context.stage("analysis.counts.cantons.annemasse")
    context.stage("analysis.counts.matching.network_from_prepare")
    context.stage("data.spatial.swiss_border")

    context.config("only_weekday", default=False)
    context.config("input_downsampling")
    context.config("include_external_population", default = False)


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
            mode="bidirectional",
            search_radius=10,
        )
    elif city == "bern":
        file_path = context.stage("analysis.counts.cantons.bern")
        counts = Counts(
            file_path=file_path,
            id_column="objectid",
            columns_to_keep={'flow':"flow", "flow_w":"flow_w"},
            context=context,
        )    
        match_kwargs = dict(
            mode="bidirectional",
            search_radius=15,
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
            mode="bidirectional",
            search_radius=80,
            prioritize_road_types=True,
        )
    elif city == "geneva":
        file_path = context.stage("analysis.counts.cantons.geneva")
        counts = Counts(file_path=file_path,  
                        id_column="OBJECTID",
                        columns_to_keep={'mean_flow_2025':"flow", "median_flow_2025":"median_flow",
                                         "osm_id":"osm_id", "angle":"angle"},
                        context = context)
        # TrafficDataMatcher detects osm_id/angle and uses the manual match.
        match_kwargs = {}
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
            mode="bidirectional",
            search_radius=10,
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
            mode="bidirectional",
            search_radius=2,
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
            mode="bidirectional",
            search_radius=10,
        )
    elif city == "zurich":
        file_path = context.stage("analysis.counts.cantons.zurich")
        counts = Counts(
            file_path=file_path,
            id_column="id",
            columns_to_keep={"flow": "flow", "flow_w": "flow_w",
                             "osm_id": "osm_id", "angle": "angle"},
            context=context,
        )
        # TrafficDataMatcher detects osm_id/angle and uses the manual match.
        match_kwargs = {}
    elif city == "annemasse":
        file_path = context.stage("analysis.counts.cantons.annemasse")
        counts = Counts(
            file_path=file_path,
            id_column="OBJECTID",
            columns_to_keep={"TJM": "flow"}, # we do not have the flow_w option and we still need to calibrate it, so we use the TJM as a proxy for flow_w
            context=context,
        )
        match_kwargs = dict(
            mode="bidirectional",
            search_radius=20,
        )
    else:
        raise ValueError(f"Unsupported city {city}")

    matcher = TrafficDataMatcher()
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
    
    network = context.stage("analysis.counts.matching.network_from_prepare")

    # Get prepared count data
    city_order = ["aargau", "bern", "ch", "geneva", "luzern", "saint_gallen", "vaud", "zurich"]
    weights = {"aargau": 1, "bern": 1, "ch": 1, "geneva": 1.2, "luzern": 1, "saint_gallen": 1, "vaud": 0.3, "zurich": 1.2}

    if context.config("include_external_population"):
        city_order.append("annemasse")

    dfs = []
    for city in city_order:
        city_df = _load_counts_and_match(context, network, city)
        if city_df is None or city_df.empty:
            logger.info(f"\t - {city}: No usable records, skipping.")
            continue
        logger.info(f"\t - {city}: {len(city_df)} matched stations")
        city_df["weight"] = weights.get(city, 1)
        dfs.append(city_df)
    if not dfs:
        raise RuntimeError("No counts could be matched to the prepared network.")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined matched dataset for calibration: {len(df)} records")

    # This would filter the data and remove the outliers and not correctly matched stations/complex intersections, this last part is done manually, only for ch data.
    df = filter_data(df, network, require_simulated=False)
    if df is None or df.empty:
        raise RuntimeError("No records left after filtering pre-simulation matched counts.")

    save_as_target(network, df, context.path())

    target_file = os.path.join(context.path(), "target_flow.csv")
    assert os.path.exists(target_file), f"Target file was not created at {target_file}"
    return target_file
