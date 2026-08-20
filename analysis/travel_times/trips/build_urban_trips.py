import geopandas as gpd
from shapely import contains_xy
import numpy as np
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("data.spatial.swiss_border")
    context.stage("data.spatial.municipality_types")
    context.stage("data.spatial.municipalities")

    context.config("num_urban_trips", default=20000)
    context.config("urban_near_buffer_m", default=3000)
    context.config("urban_trip_distance_min_m", default=3000)
    context.config("urban_trip_distance_max_m", default=15000)
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
    near_buffer_m = int(context.config("urban_near_buffer_m"))
    min_trip_distance_m = int(context.config("urban_trip_distance_min_m"))
    max_trip_distance_m = int(context.config("urban_trip_distance_max_m"))

    if near_buffer_m < 0:
        raise ValueError("urban_near_buffer_m must be >= 0")
    if min_trip_distance_m < 0:
        raise ValueError("urban_trip_distance_min_m must be >= 0")
    if max_trip_distance_m <= min_trip_distance_m:
        raise ValueError("urban_trip_distance_max_m must be strictly greater than urban_trip_distance_min_m")

    # municipality types:
    df_mt = context.stage("data.spatial.municipality_types")
    df_mu = context.stage("data.spatial.municipalities")[0]
    df_mu = df_mu.merge(df_mt, on="municipality_id")
    urban_municipalities = df_mu.loc[df_mu['municipality_type'].isin(['urban']),["municipality_type","geometry"]]
    urbancore_municipalities = df_mu.loc[df_mu['municipality_type'].isin(['urbancore']),["municipality_type","geometry"]]
    urban_or_urbancore = df_mu.loc[df_mu['municipality_type'].isin(['urban', 'urbancore']), ["municipality_type", "geometry"]]

    if urban_or_urbancore.empty:
        raise ValueError("No municipalities with type urban or urbancore were found")

    x_values = net.nodes["x"].values
    y_values = net.nodes["y"].values

    # nodes in urban municipalities
    urban_nodes = contains_xy(urban_municipalities.geometry.union_all().simplify(500), x_values, y_values)
    urbancore_nodes = contains_xy(urbancore_municipalities.geometry.union_all().simplify(500), x_values, y_values)

    # include nodes that are close to urban municipalities using a positive outward buffer
    near_urban_geometry = urban_or_urbancore.geometry.union_all().simplify(500).buffer(near_buffer_m)
    near_urban_nodes = contains_xy(near_urban_geometry, x_values, y_values)

    # get swiss border
    border = context.stage("data.spatial.swiss_border").geometry.simplify(2000).iloc[0]

    # filter nodes inside a 2km *inward-buffered* Swiss border (single-step filter)
    # EPSG:2056 is metric (meters), so 2km == 2_000m.
    buffered_border = border.buffer(-2_000)
    if buffered_border.is_empty:
        raise ValueError("Buffered border is empty; cannot filter nodes using a -2km border buffer")

    nodes_within_buffered_border = contains_xy(buffered_border, x_values, y_values)
    urban_nodes = urban_nodes & nodes_within_buffered_border
    urbancore_nodes = urbancore_nodes & nodes_within_buffered_border
    near_urban_nodes = (near_urban_nodes | urban_nodes | urbancore_nodes) & nodes_within_buffered_border

    if not np.any(near_urban_nodes):
        raise ValueError("No network nodes found in or near urban municipalities after border filtering")
    
    #  Create trips
    num_trips = context.config("num_urban_trips")
    if num_trips <= 0:
        raise ValueError("num_urban_trips must be > 0")

    def sample_nodes(nodes_mask, n_samples):
        if n_samples <= 0:
            return net.nodes.iloc[0:0].copy()

        candidates = net.nodes.loc[nodes_mask]
        if candidates.empty:
            return candidates

        replace = len(candidates) < n_samples
        rs = int(rng.integers(0, 1_000_000_000 - 1))
        return candidates.sample(n=n_samples, replace=replace, random_state=rs)

    # sample origin nodes from near-urban OR urban OR urbancore nodes
    origin_nodes_mask = near_urban_nodes | urban_nodes | urbancore_nodes
    sampled_origins = sample_nodes(origin_nodes_mask, num_trips).reset_index(drop=True)

    if len(sampled_origins) < num_trips:
        raise ValueError(
            f"Not enough eligible origin nodes ({len(sampled_origins)}) to sample {num_trips} urban trips"
        )

    # sample destination nodes from urban or urbancore areas only
    destination_nodes_mask = urban_nodes | urbancore_nodes
    destination_nodes = net.nodes.loc[destination_nodes_mask].copy()
    if destination_nodes.empty:
        raise ValueError("No eligible destination nodes found in urban or urbancore areas")

    urban_node_ids = set(net.nodes.loc[urban_nodes, "node_id"].astype(str))
    urbancore_node_ids = set(net.nodes.loc[urbancore_nodes, "node_id"].astype(str))
    destination_node_ids = destination_nodes["node_id"].astype(str)

    destination_nodes["weight"] = 1.0
    destination_nodes.loc[destination_node_ids.isin(urban_node_ids), "weight"] = 4.0
    destination_nodes.loc[destination_node_ids.isin(urbancore_node_ids), "weight"] = 8.0

    sampled_destinations = []
    destination_x = destination_nodes["x"].values
    destination_y = destination_nodes["y"].values

    for _, origin_row in sampled_origins.iterrows():
        distances = np.hypot(destination_x - origin_row.x, destination_y - origin_row.y)
        eligible_destinations = destination_nodes[(distances >= min_trip_distance_m) & (distances <= max_trip_distance_m)]
        if eligible_destinations.empty:
            raise ValueError(
                f"No eligible destination nodes found with a distance between {min_trip_distance_m}m and "
                f"{max_trip_distance_m}m for a sampled origin; "
                "consider adjusting constraints or expanding the destination node set"
            )
        destination_row = eligible_destinations.sample(
            n=1,
            random_state=int(rng.integers(0, 1_000_000_000 - 1)),
            weights="weight"
        ).iloc[0]
        sampled_destinations.append(destination_row)
    
    sampled_destinations = gpd.GeoDataFrame(sampled_destinations).reset_index(drop=True)
    
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
        'identifier': [f'urban_trip_{i+1}' for i in range(num_trips)],
        'origin_x': sampled_origins['x'],
        'origin_y': sampled_origins['y'],
        'destination_x': sampled_destinations['x'],
        'destination_y': sampled_destinations['y'],
        'departure_time': departure_times
    })

    return trips[['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']]
