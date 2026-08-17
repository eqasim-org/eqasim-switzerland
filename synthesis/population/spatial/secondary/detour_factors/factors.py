from matsim.readers import read_network
import pandas as pd
from .tools import build_h3_grid, build_kdtree, compute_detour_factors, adapt_h3_centers_to_nearest_nodes, deduplicate_h3_centers, plot_long_detours_html, DetourFactorMatrix
import logging

logger = logging.getLogger("synpp")

EPSILON = 500 # meters -> detour_factor = shortest_distances / (euclidean_distances + epsilon)
MAX_DISTANCE_THRESHOLD_M = 40_000  # route only OD pairs closer than this euclidean distance (set None to disable)
MIN_DISTANCE_THRESHOLD_M = 900  # route only OD pairs further than this euclidean distance (set None to disable)
FAR_DETOUR_FACTOR = 1.4  # assigned when euclidean distance is above DISTANCE_THRESHOLD_M
NEAR_DETOUR_FACTOR = 1.6  # assigned when euclidean distance is below MIN_DISTANCE_THRESHOLD_M
ADAPTIVE_H3_RESOLUTION = True
RESOLUTIONS = {"dense": 8,"sparse": 7,"out": 5}
DENSE_CELL_MIN_NODES = 40
ORIGIN_BATCH_SIZE = 128
LONG_DETOUR_PLOT_THRESHOLD = 4.0

def configure(context):
    context.stage("matsim.scenario.network.convert_osm")
    context.stage("data.spatial.swiss_border")
    
    context.config("cross_border_exclude_shapefiles", default=None)
    context.config("include_external_population", default=False)


def execute(context):
    logger.info("\t Read the road network ...")
    network_file = context.stage("matsim.scenario.network.convert_osm")
    network = read_network(network_file, only_car_links = True)

    # build a grid of h3
    logger.info("\t Build h3 grid ...")
    matrix, h3_to_index, node_to_index, h3_centers = build_h3_grid(
        context,
        network.nodes,
        adaptive_resolution=ADAPTIVE_H3_RESOLUTION,
        resolutions = RESOLUTIONS,
        dense_cell_min_nodes=DENSE_CELL_MIN_NODES,
        border_buffer = 10_000
    )

    # build kdtree of nodes
    logger.info("\t Build kdtree of network nodes ...")
    nodes_tree = build_kdtree(network.nodes)

    # snap each h3 center to the nearest network node coordinates
    logger.info("\t Snap each h3 center to the nearest network node coordinates ...")
    h3_centers = adapt_h3_centers_to_nearest_nodes(h3_centers, nodes_tree, network)

    # remove duplicated snapped centers and remap matrix/indexes consistently
    matrix, h3_to_index, node_to_index, h3_centers = deduplicate_h3_centers(
        matrix,
        h3_to_index,
        node_to_index,
        h3_centers,
    )

    # build kdtree of h3 centers
    logger.info("\t Build kdtree of h3 centers ...")
    h3_tree = build_kdtree(
        pd.DataFrame({"x": [h3_centers[h3_id][0] for h3_id in h3_to_index.keys()],
                      "y": [h3_centers[h3_id][1] for h3_id in h3_to_index.keys()],
                      "h3_id": list(h3_to_index.keys())}),
        x="x", y="y", object_id="h3_id"
    )

    # build pandana graph
    logger.info("\t Build pandana graph ...")
    # Keep only_car_links=False here to preserve node indexing with nodes_tree/network.nodes.
    graph = network.as_pandana(only_car_links=False, directed=False)

    # compute detour factors (routing is batched per origin, one pandana call per row)
    logger.info("\t Compute detour factors ...")
    matrix = compute_detour_factors(
        matrix,
        graph,
        h3_to_index,
        node_to_index,
        h3_centers,
        nodes_tree,
        imp_name="length",
        epsilon=EPSILON,
        max_distance_threshold_m=MAX_DISTANCE_THRESHOLD_M,
        min_distance_threshold_m=MIN_DISTANCE_THRESHOLD_M,
        far_detour_factor=FAR_DETOUR_FACTOR,
        near_detour_factor=NEAR_DETOUR_FACTOR,
        origin_batch_size=ORIGIN_BATCH_SIZE,
    )

    # export map of long detours
    plot_long_detours_html(
        matrix,
        h3_to_index,
        h3_centers,
        f"{context.path()}/long_detours.html",
        threshold=LONG_DETOUR_PLOT_THRESHOLD,
    )

    # wrap everything needed to query a detour factor from arbitrary coordinates
    return DetourFactorMatrix(matrix, h3_to_index, h3_centers, h3_tree)