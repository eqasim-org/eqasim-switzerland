import geopandas as gpd
from shapely import vectorized
import numpy as np

def configure(context):
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")

    context.config("num_highway_trips", default=2000)    
    context.config("random_seed")

def execute(context):
    # load network
    net = context.stage("analysis.counts.matching.network")

    # get highway links    
    highway_links = net.links[net.links["highway"].isin(('motorway', 'trunk', 'motorway_link', 'trunk_link'))]
    
    # get highway origin/destination nodes (origins from from_node, destinations from to_node)
    origin_node_ids = highway_links["from_node"].astype(str).unique()
    destination_node_ids = highway_links["to_node"].astype(str).unique()

    origin_nodes = net.nodes[net.nodes["node_id"].astype(str).isin(origin_node_ids)]
    destination_nodes = net.nodes[net.nodes["node_id"].astype(str).isin(destination_node_ids)]

    origin_nodes_geometry = gpd.points_from_xy(origin_nodes["x"], origin_nodes["y"])
    destination_nodes_geometry = gpd.points_from_xy(destination_nodes["x"], destination_nodes["y"])

    origin_nodes_gdf = gpd.GeoDataFrame(origin_nodes, geometry=origin_nodes_geometry, crs="EPSG:2056")
    destination_nodes_gdf = gpd.GeoDataFrame(destination_nodes, geometry=destination_nodes_geometry, crs="EPSG:2056")

    # get swiss border
    border = context.stage("data.spatial.swiss_border").geometry.simplify(2000).iloc[0]

    # filter nodes inside a 10km *inward-buffered* Swiss border (single-step filter)
    # EPSG:2056 is metric (meters), so 10km == 10_000m.
    buffered_border = border.buffer(-10_000)
    if buffered_border.is_empty:
        raise ValueError("Buffered border is empty; cannot filter nodes using a -10km border buffer")

    within_buffered_border_origins = vectorized.contains(
        buffered_border, origin_nodes_gdf.geometry.x.values, origin_nodes_gdf.geometry.y.values
    )
    within_buffered_border_destinations = vectorized.contains(
        buffered_border, destination_nodes_gdf.geometry.x.values, destination_nodes_gdf.geometry.y.values
    )

    swiss_origin_nodes = origin_nodes_gdf[within_buffered_border_origins]
    swiss_destination_nodes = destination_nodes_gdf[within_buffered_border_destinations]

    # sample origin nodes
    num_trips = context.config("num_highway_trips")
    seed = context.config("random_seed")
    if len(swiss_origin_nodes) < num_trips:
        raise ValueError(
            f"Not enough eligible origin nodes ({len(swiss_origin_nodes)}) to sample {num_trips} trips"
        )
    sampled_origins = swiss_origin_nodes.sample(n=num_trips, replace=False, random_state=seed).reset_index(drop=True)

    # sample destination nodes (have a distance between 10km and 100km)
    sampled_destinations = []
    for idx, origin_row in sampled_origins.iterrows():
        origin_point = origin_row.geometry
        distances = swiss_destination_nodes.geometry.distance(origin_point)
        eligible_destinations = swiss_destination_nodes[(distances >= 10000) & (distances <= 100000)]
        if eligible_destinations.empty:
            raise ValueError(
                "No eligible destination nodes found within [10km, 100km] for a sampled origin; "
                "consider adjusting constraints or expanding the destination node set"
            )
        destination_row = eligible_destinations.sample(n=1, random_state=(seed or 0) + idx).iloc[0]
        sampled_destinations.append(destination_row)
    
    sampled_destinations = gpd.GeoDataFrame(sampled_destinations).reset_index(drop=True)
    
    # sample departure times (between 7 AM and 8 PM)
    rng = np.random.default_rng(seed)
    departure_times = rng.integers(7 * 3600, 20 * 3600, size=num_trips)

    # build trips dataframe
    trips = gpd.GeoDataFrame({
        'identifier': [f'highway_trip_{i+1}' for i in range(num_trips)],
        'origin_x': sampled_origins['x'],
        'origin_y': sampled_origins['y'],
        'destination_x': sampled_destinations['x'],
        'destination_y': sampled_destinations['y'],
        'departure_time': departure_times
    })

    return trips[['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']]