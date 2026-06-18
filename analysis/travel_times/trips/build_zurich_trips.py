import geopandas as gpd
from shapely import vectorized
import numpy as np
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.spatial.cantons")
    
    context.config("num_zurich_trips", default=15_000)
    context.config("zurich_trip_distance_min_m", default=4_000)
    context.config("zurich_trip_distance_max_m", default=20_000)
    context.config("random_seed")
    
    # here we check if any of the stages that require the network from prepare are requested, to avoid unnecessary dependencies and re-running prepare when not needed
    stages_to_check = [ "matsim.output", "matsim.simulation.run", "matsim.simulation.prepare"]
    get_network_from_prepare = any(requested_stage.__name__ in stages_to_check for requested_stage in context.config_requested_stages)

    if get_network_from_prepare:
        context.stage("analysis.counts.matching.network_from_prepare", alias="network")
    else:
        context.stage("analysis.counts.matching.network", alias="network")
    

def execute(context):
    # load network (do not create dependency on the network stage to avoid running the prepare (maybe the whole pipeline) again)
    net = context.stage("network")
    base_seed = int(context.config("random_seed") or 0)
    rng = np.random.default_rng(base_seed)
    num_trips = int(context.config("num_zurich_trips"))
    min_trip_distance_m = int(context.config("zurich_trip_distance_min_m"))
    max_trip_distance_m = int(context.config("zurich_trip_distance_max_m"))

    if num_trips <= 0:
        raise ValueError("num_zurich_trips must be > 0")
    if min_trip_distance_m < 0:
        raise ValueError("zurich_trip_distance_min_m must be >= 0")
    if max_trip_distance_m <= min_trip_distance_m:
        raise ValueError("zurich_trip_distance_max_m must be strictly greater than zurich_trip_distance_min_m")

    # geom canton zurich
    df_cantons = context.stage("data.spatial.cantons")
    zurich_rows = df_cantons.loc[
        df_cantons["canton_name_en"].astype(str).str.casefold() == "zurich",
        "geometry"
    ]
    if zurich_rows.empty:
        raise ValueError("Could not find canton 'zurich' in data.spatial.cantons")
    zurich_geom = zurich_rows.buffer(2_000).simplify(500).iloc[0]

    # nodes
    x_values = net.nodes["x"].values
    y_values = net.nodes["y"].values

    # nodes in canton Zurich
    zurich_nodes_mask = vectorized.contains(zurich_geom, x_values, y_values)
    zurich_nodes = net.nodes.loc[zurich_nodes_mask, ["node_id", "x", "y"]].reset_index(drop=True)
    if len(zurich_nodes) < 2:
        raise ValueError("Need at least two network nodes in Zurich to build OD trips")

    node_ids = zurich_nodes["node_id"].astype(str).values
    node_x = zurich_nodes["x"].values
    node_y = zurich_nodes["y"].values

    # Draw random origin/destination pairs and keep only valid pairs.
    sampled_origin_idx = []
    sampled_destination_idx = []
    attempts = 0
    max_attempts = max(num_trips * 50, 10_000)

    while len(sampled_origin_idx) < num_trips and attempts < max_attempts:
        attempts += 1
        origin_idx = int(rng.integers(0, len(zurich_nodes)))
        origin_id = node_ids[origin_idx]

        distances = np.hypot(node_x - node_x[origin_idx], node_y - node_y[origin_idx])
        eligible_mask = (
            (distances >= min_trip_distance_m)
            & (distances <= max_trip_distance_m)
            & (node_ids != origin_id)
        )
        eligible_idx = np.flatnonzero(eligible_mask)

        if eligible_idx.size == 0:
            continue

        destination_idx = int(rng.choice(eligible_idx))
        sampled_origin_idx.append(origin_idx)
        sampled_destination_idx.append(destination_idx)

    if len(sampled_origin_idx) < num_trips:
        raise ValueError(
            "Unable to generate the requested number of Zurich trips with the configured distance "
            f"constraints after {attempts} attempts. Generated {len(sampled_origin_idx)} / {num_trips} "
            "trips; consider relaxing distance constraints."
        )

    sampled_origins = zurich_nodes.iloc[sampled_origin_idx].reset_index(drop=True)
    sampled_destinations = zurich_nodes.iloc[sampled_destination_idx].reset_index(drop=True)
    
    # sample departure times with peak-heavy demand pattern
    time_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))

    # 70% of trips in peak periods (06:30-10:00 and 16:00-19:00), 30% spread across 05:00-23:00
    num_peak = int(num_trips * 0.7)
    num_off_peak = num_trips - num_peak
    departure_times = np.concatenate([
        time_rng.integers(int(6.5 * 3600), 10 * 3600, size=num_peak // 2),
        time_rng.integers(16 * 3600, 19 * 3600, size=num_peak - num_peak // 2),
        time_rng.integers(5 * 3600, 23 * 3600, size=num_off_peak)
    ])
    time_rng.shuffle(departure_times)

    # build trips dataframe
    trips = gpd.GeoDataFrame({
        'identifier': [f'zurich_trip_{i+1}' for i in range(num_trips)],
        'origin_x': sampled_origins['x'],
        'origin_y': sampled_origins['y'],
        'destination_x': sampled_destinations['x'],
        'destination_y': sampled_destinations['y'],
        'departure_time': departure_times
    })

    logger.info(
        "Generated %s Zurich trips between distinct nodes with crow-fly distance in [%sm, %sm]",
        num_trips,
        min_trip_distance_m,
        max_trip_distance_m,
    )

    return trips[['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']]