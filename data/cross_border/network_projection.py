import logging
import os

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely.ops
from shapely.geometry import LineString, Point
from sklearn.neighbors import KDTree

from matsim.readers import read_network

logger = logging.getLogger("synpp")

_TILE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.pixelkarte-grau/default/current/3857/{z}/{x}/{y}.jpeg"

CANDIDATE_COUNT = 8   # nearest eligible-road network nodes considered per ideal crossing point
MAP_SAMPLE_SIZE = 5000  # cap the number of records drawn on the map


MIN_ENTRY_FREESPEED_THRESHOLD = 50 / 3.6


MOTORWAY_FREESPEED_THRESHOLD = 75 / 3.6

MOTORWAY_SEARCH_RADIUS = 30000

MOTORWAY_CANDIDATE_COUNT = 3

RESULT_COLUMNS = [
    "real_x", "real_y", "interview_x", "interview_y", "occurrences",
    "ideal_crossing_x", "ideal_crossing_y",
    "refined_x", "refined_y", "refined_node_id",
    "travel_time_to_interview_place",
]


def configure(context):
    context.stage("data.cross_border.generate_od")
    context.stage("data.spatial.swiss_border")

    context.config("border_offset", default = 20000)
    context.config("random_seed")

    context.config("cached_network_path", default = None)
    if not context.config("cached_network_path"):
        context.stage("matsim.scenario.network.convert_osm")


def execute(context):
    df            = context.stage("data.cross_border.generate_od")
    ch_border     = context.stage("data.spatial.swiss_border").copy()[0]
    border_offset = context.config("border_offset")

    cached_network_path = context.config("cached_network_path")
    if cached_network_path:
        network_path = cached_network_path
        logger.info(
            "Using cached network file at %s instead of running "
            "matsim.scenario.network.convert_osm.", network_path,
        )
    else:
        network_path = context.stage("matsim.scenario.network.convert_osm")

    requests = []

    # PT trips are excluded here - see the module docstring for why refining
    # onto the car network wouldn't be meaningful for them.
    is_car_trip = df["trip_mode"].isin(["car", "car_passenger"])

    df_origin_projected = df[(df["origin_is_projected"] == True) & is_car_trip]
    if len(df_origin_projected) > 0:
        requests.append(df_origin_projected[["residence_x", "residence_y", "origin_x", "origin_y"]].rename(
            columns = {
                "residence_x": "real_x", "residence_y": "real_y",
                "origin_x": "interview_x", "origin_y": "interview_y",
            },
        ))

    if "destination_is_projected" in df.columns:
        df_destination_projected = df[(df["destination_is_projected"] == True) & is_car_trip]
        if len(df_destination_projected) > 0:
            requests.append(df_destination_projected[
                ["destination_residence_x", "destination_residence_y", "destination_x", "destination_y"]
            ].rename(columns = {
                "destination_residence_x": "real_x", "destination_residence_y": "real_y",
                "destination_x": "interview_x", "destination_y": "interview_y",
            }))

    if len(requests) == 0:
        logger.info("No projected origins/destinations in data.cross_border.generate_od; nothing to refine.")
        return pd.DataFrame(columns = RESULT_COLUMNS)

    df_requests = pd.concat(requests, ignore_index = True)

    unique = df_requests.groupby(
        ["real_x", "real_y", "interview_x", "interview_y"], as_index = False
    ).size().rename(columns = {"size": "occurrences"})

    logger.info(
        "%d origin/destination-projected records collapse to %d unique (real point, assigned interview "
        "place) pairs.", len(df_requests), len(unique),
    )

    logger.info("Reading the car network...")
    network   = read_network(network_path, only_car_links = True)
    graph     = network.as_pandana(only_car_links = False, directed = True)
    nodes     = network.nodes.reset_index(drop = True)
    node_tree = KDTree(nodes[["x", "y"]].values)

    node_id_to_index = pd.Series(nodes.index.values, index = nodes["node_id"].values)

    def node_indices_touching(min_freespeed):
        links = network.links[network.links["freespeed"] >= min_freespeed]
        return pd.unique(node_id_to_index.loc[
            pd.concat([links["from_node"], links["to_node"]]).unique()
        ].values)

    entry_node_indices = node_indices_touching(MIN_ENTRY_FREESPEED_THRESHOLD)
    logger.info(
        "%d/%d car-network nodes touch a link with freespeed >= %.0f km/h; entry-point candidates "
        "(other than the interview place's own nearest node) are restricted to these.",
        len(entry_node_indices), len(nodes), MIN_ENTRY_FREESPEED_THRESHOLD * 3.6,
    )
    entry_node_tree = KDTree(nodes.loc[entry_node_indices, ["x", "y"]].values)

    motorway_node_indices = node_indices_touching(MOTORWAY_FREESPEED_THRESHOLD)
    logger.info(
        "%d/%d car-network nodes touch a motorway-class link (freespeed >= %.0f km/h).",
        len(motorway_node_indices), len(nodes), MOTORWAY_FREESPEED_THRESHOLD * 3.6,
    )
    motorway_tree = KDTree(nodes.loc[motorway_node_indices, ["x", "y"]].values) if len(motorway_node_indices) > 0 else None

    ch_border_simple = ch_border.simplify(50)
    outer_boundary   = ch_border_simple.buffer(border_offset).boundary

    logger.info("Computing the ideal (geometry-only) crossing point per pair...")
    crossings = [
        find_ideal_crossing(
            (row.real_x, row.real_y),
            (row.interview_x, row.interview_y),
            outer_boundary,
        )
        for row in unique.itertuples()
    ]
    unique["ideal_crossing_x"] = [p.x for p in crossings]
    unique["ideal_crossing_y"] = [p.y for p in crossings]

    logger.info("Snapping onto the %d nearest eligible-road network nodes per pair...", CANDIDATE_COUNT)
    crossing_xy = unique[["ideal_crossing_x", "ideal_crossing_y"]].values
    candidate_k = min(CANDIDATE_COUNT, len(entry_node_indices))
    _, candidate_positions = entry_node_tree.query(crossing_xy, k = candidate_k)
    candidate_node_ids = entry_node_indices[candidate_positions.reshape(len(unique), candidate_k)]

    if motorway_tree is not None:
        motorway_k = min(MOTORWAY_CANDIDATE_COUNT, len(motorway_node_indices))
        logger.info(
            "Adding the %d nearest motorway-class node(s) within %.0fm of the ideal crossing point, "
            "where any exist, to each pair's candidate pool...", motorway_k, MOTORWAY_SEARCH_RADIUS,
        )
        motorway_distances, motorway_positions = motorway_tree.query(crossing_xy, k = motorway_k)
        motorway_distances = motorway_distances.reshape(len(unique), motorway_k)
        motorway_candidates = motorway_node_indices[motorway_positions.reshape(len(unique), motorway_k)]

        has_nearby_motorway = motorway_distances <= MOTORWAY_SEARCH_RADIUS
        # Slots with no motorway node in range fall back to duplicating the
        # plain-nearest candidate - a harmless no-op extra column, since it's
        # already in the pool and argmin just sees it twice.
        motorway_candidates = np.where(has_nearby_motorway, motorway_candidates, candidate_node_ids[:, :1])

        logger.info(
            "%d/%d pairs had at least one motorway-class node within range.",
            has_nearby_motorway.any(axis = 1).sum(), len(unique),
        )
        candidate_node_ids = np.hstack([candidate_node_ids, motorway_candidates])

    interview_xy = unique[["interview_x", "interview_y"]].values
    _, target_node_ids = node_tree.query(interview_xy, k = 1)
    target_node_ids = target_node_ids.flatten()

    candidate_count = candidate_node_ids.shape[1]
    logger.info("Routing %d candidate -> interview-place pairs on the car network...", candidate_node_ids.size)
    flat_candidates = candidate_node_ids.flatten()
    flat_targets = np.repeat(target_node_ids, candidate_count)

    travel_times = np.array(graph.shortest_path_lengths(
        flat_candidates.tolist(), flat_targets.tolist(), imp_name = "travel_time",
    )).reshape(candidate_node_ids.shape)

    best_k = np.argmin(travel_times, axis = 1)
    row_range = np.arange(len(unique))
    best_node_ids = candidate_node_ids[row_range, best_k]
    best_travel_times = travel_times[row_range, best_k]

    unreachable = ~np.isfinite(best_travel_times)
    if unreachable.any():
        logger.warning(
            "%d pairs had no reachable candidate node among the %d considered for the ideal crossing "
            "point; their refined point falls back to the closest candidate by distance instead of time.",
            unreachable.sum(), candidate_count,
        )
        unreachable_xy = nodes.loc[candidate_node_ids[unreachable].flatten(), ["x", "y"]].values.reshape(
            unreachable.sum(), candidate_count, 2,
        )
        distances = np.hypot(
            unreachable_xy[:, :, 0] - crossing_xy[unreachable, 0][:, None],
            unreachable_xy[:, :, 1] - crossing_xy[unreachable, 1][:, None],
        )
        best_k[unreachable] = np.argmin(distances, axis = 1)
        best_node_ids = candidate_node_ids[row_range, best_k]
        best_travel_times = travel_times[row_range, best_k]

    unique["refined_node_id"]                = nodes.loc[best_node_ids, "node_id"].values
    unique["refined_x"]                      = nodes.loc[best_node_ids, "x"].values
    unique["refined_y"]                      = nodes.loc[best_node_ids, "y"].values
    unique["travel_time_to_interview_place"] = best_travel_times

    moved_distance = np.hypot(
        unique["refined_x"] - unique["interview_x"],
        unique["refined_y"] - unique["interview_y"],
    )
    logger.info(
        "Refined entry point vs. treating the interview place itself as the origin/destination: "
        "median %.0fs network travel time still separates them, median %.0fm straight-line distance moved.",
        np.nanmedian(best_travel_times[np.isfinite(best_travel_times)]) if np.isfinite(best_travel_times).any() else float("nan"),
        np.median(moved_distance),
    )

    build_map(unique, context.config("random_seed"), os.path.join(context.path(), "network_projection_map.html"))

    return unique[RESULT_COLUMNS]


def find_ideal_crossing(real_xy, interview_xy, outer_boundary):
    """
    Where the straight line from a real (far) point to its assigned
    interview place crosses the outer edge of the border buffer - the
    purely-geometric "as close as possible to the buffer's outer boundary,
    on the line to the interview place" point this stage then snaps onto a
    real network node.
    """
    real      = Point(real_xy)
    interview = Point(interview_xy)
    line = LineString([real, interview])

    intersection = line.intersection(outer_boundary)

    if intersection.is_empty:
        # The line never crosses the buffer's edge (e.g. it grazes past a
        # concave part of the boundary) - fall back to the closest point on
        # the boundary to the real point.
        return shapely.ops.nearest_points(outer_boundary, real)[0]

    if intersection.geom_type == "Point":
        return intersection

    if intersection.geom_type == "MultiPoint":
        return min(intersection.geoms, key = real.distance)

    # A LineString/MultiLineString means the line runs along the boundary
    # for a stretch (near-tangent) - fall back to the closest point on the
    # boundary to the intersection's centroid.
    return shapely.ops.nearest_points(outer_boundary, intersection.centroid)[0]


def build_map(df, random_seed, output_path):
    """
    One folium map: for a sample of up to MAP_SAMPLE_SIZE refined requests,
    the real point (gray dot), the interview place (red dot, i.e. what
    data.cross_border.generate_od actually uses today), the refined network
    entry point (blue dot), and a line connecting all three - so the
    refinement can be checked visually against real roads.
    """
    df_map = df if len(df) <= MAP_SAMPLE_SIZE else df.sample(MAP_SAMPLE_SIZE, random_state = random_seed)

    def to_wgs84(x, y):
        return gpd.GeoSeries(gpd.points_from_xy(x, y), crs = "EPSG:2056").to_crs("EPSG:4326")

    real_wgs84      = to_wgs84(df_map["real_x"], df_map["real_y"]).reset_index(drop = True)
    interview_wgs84 = to_wgs84(df_map["interview_x"], df_map["interview_y"]).reset_index(drop = True)
    refined_wgs84   = to_wgs84(df_map["refined_x"], df_map["refined_y"]).reset_index(drop = True)
    df_map          = df_map.reset_index(drop = True)

    center = [interview_wgs84.y.mean(), interview_wgs84.x.mean()]
    m = folium.Map(location = center, zoom_start = 9, tiles = None)
    folium.TileLayer(
        tiles = _TILE_URL, attr = "© swisstopo", name = "swisstopo (grayscale)", opacity = 0.6, control = False,
    ).add_to(m)

    layer = folium.FeatureGroup(name = "Refined border entry points", show = True)

    for i, row in df_map.iterrows():
        r, v, n = real_wgs84.iloc[i], interview_wgs84.iloc[i], refined_wgs84.iloc[i]

        popup = (
            f"Occurrences: {row['occurrences']:.0f}<br>"
            f"Travel time, refined entry -&gt; interview place: {row['travel_time_to_interview_place']:.0f}s<br>"
            f"Refined node: {row['refined_node_id']}"
        )

        folium.PolyLine(
            locations = [(r.y, r.x), (n.y, n.x), (v.y, v.x)],
            color = "#666666", weight = 1, opacity = 0.6,
        ).add_to(layer)

        folium.CircleMarker(
            location = (r.y, r.x), radius = 3,
            color = "#7f7f7f", fill = True, fill_opacity = 0.8,
            tooltip = "Real point",
        ).add_to(layer)
        folium.CircleMarker(
            location = (v.y, v.x), radius = 4,
            color = "#d62728", fill = True, fill_opacity = 0.8,
            tooltip = "Interview place (data.cross_border.generate_od's unrefined point)",
        ).add_to(layer)
        folium.CircleMarker(
            location = (n.y, n.x), radius = 4,
            color = "#1f77b4", fill = True, fill_opacity = 0.8,
            popup = folium.Popup(popup, max_width = 280),
            tooltip = "Refined network entry point",
        ).add_to(layer)

    layer.add_to(m)
    folium.LayerControl(collapsed = False).add_to(m)

    m.save(output_path)
