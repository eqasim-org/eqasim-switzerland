import geopandas as gpd
import h3
import logging
import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree
from typing import List, Tuple, Union
from tqdm import tqdm
import plotly.graph_objects as go
from shapely.ops import unary_union
from shapely import contains_xy
from data.osm.clean import read_outside_region

logger = logging.getLogger("synpp")


def _remove_h3_children(cell_ids):
    selected = set(cell_ids)
    kept = set()

    for cell in selected:
        keep = True
        parent = cell
        while h3.get_resolution(parent) > 0:
            parent = h3.cell_to_parent(parent)
            if parent in selected:
                keep = False
                break
        if keep:
            kept.add(cell)

    return kept

########################### H3 GRID #########################
def build_h3_grid(
        context,
        df,
        x="x",
        y="y",
        resolution=8,
        crs="EPSG:2056",
        object_id="node_id",
        adaptive_resolution=False,
        resolutions={"dense": 8, "sparse": 7, "out": 4},
        dense_cell_min_nodes=20,
        border_buffer = 0,
    ):
    assert df[x].notnull().all() and df[y].notnull().all(), "x and y columns must not contain null values"
    df = df.copy()

    # build geodataframe
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[x], df[y]), crs=crs)

    # transform to 'EPSG:4326'
    x_coord = df.geometry.x.values
    y_coord = df.geometry.y.values
    df = df.to_crs("EPSG:4326")

    if adaptive_resolution:
        # where to use Dense resolutions
        df["h3_sparse"] = [
            h3.latlng_to_cell(lat, lon, resolutions["sparse"])
            for lat, lon in zip(df.geometry.y, df.geometry.x)
        ]
        sparse_counts = df.groupby("h3_sparse").size()
        dense_sparse_cells = set(sparse_counts[sparse_counts >= dense_cell_min_nodes].index)
        use_dense_resolution = df["h3_sparse"].isin(dense_sparse_cells).to_numpy()
        
        # where to use out resolutions
        modeled_region = get_modeled_region(context, border_buffer=border_buffer)
        use_out_resolution = ~contains_xy(modeled_region, x_coord, y_coord) # these coords are in epsg:2056

        # adaptive h3 resolution
        res = [resolutions["dense"], resolutions["sparse"], resolutions["out"]]
        res_index = np.where(use_out_resolution, 2, np.where(use_dense_resolution, 0, 1))
         
        df["h3"] = [
            h3.latlng_to_cell(lat, lon, res[r])
            for lat, lon, r in zip(df.geometry.y, df.geometry.x, res_index)
        ]

    else:
        # single fixed h3 resolution
        df["h3"] = [h3.latlng_to_cell(lat, lon, resolution) for lat, lon in zip(df.geometry.y, df.geometry.x)]

    # If a parent cell is present, remove all of its children by remapping children to the nearest present ancestor.
    initial_unique_h3 = set(df["h3"].unique())
    kept_h3 = _remove_h3_children(initial_unique_h3)
    if len(kept_h3) != len(initial_unique_h3):
        def _to_kept_h3(cell):
            while cell not in kept_h3:
                cell = h3.cell_to_parent(cell)
            return cell

        df["h3"] = df["h3"].map(_to_kept_h3)
        logger.info(
            "Removed parent/child h3 overlap: %s -> %s cells",
            len(initial_unique_h3),
            len(kept_h3),
        )

    # unique ids
    unique_h3 = df["h3"].unique()

    # matrix
    matrix = np.zeros((len(unique_h3), len(unique_h3)), dtype=float)
    h3_to_index = {h3_id: idx for idx, h3_id in enumerate(unique_h3)}
    node_to_index = {node_id: h3_to_index[h3_id] for node_id, h3_id in zip(df[object_id], df["h3"])}

    # centers of h3 (Point of the center of the polygone)
    h3_centers = {h3_id: h3.cell_to_latlng(h3_id) for h3_id in unique_h3}

    # transform the centers to the original crs
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    h3_centers = {h3_id: transformer.transform(lon, lat) for h3_id, (lat, lon) in h3_centers.items()}

    return matrix, h3_to_index, node_to_index, h3_centers


def get_modeled_region(context, border_buffer=0):
    # Bounding Area
    border = context.stage("data.spatial.swiss_border")
    border = border.reset_index()[["geometry"]].to_crs(epsg=2056)
    
    # Collect all geometries to combine
    geometries = [unary_union(border.geometry)]
    
    # outside CH region
    out_region_file = context.config("cross_border_exclude_shapefiles")
    include_external_population = context.config("include_external_population")
    if out_region_file is not None and include_external_population:
        out_region = read_outside_region(out_region_file)
        geometries.append(unary_union(out_region.geometry))
    
    # Combine border + outside regions into one geometry
    combined = unary_union(geometries)

    # add the buffer to the final geomatry
    if border_buffer > 0:
        combined = combined.buffer(border_buffer)

    return combined

########################### Adapte centers coordinates #########################
def adapt_h3_centers_to_nearest_nodes(h3_centers, nodes_tree, network, restricted_highway_types=("motorway_link", "motorway", "trunk", "trunk_link"),
                                      max_non_restricted_distance_m=200.0,
                                      max_non_restricted_distance_ratio=1.5):
    
    highway_type = network.links["attributes"].apply(
        lambda x: x.get("osm:way:highway", "") if isinstance(x, dict) else ""
    )

    restricted_links = network.links[highway_type.isin(set(restricted_highway_types))]
    restricted_node_ids = set(restricted_links["from_node"]).union(set(restricted_links["to_node"]))

    items = list(h3_centers.items())
    h3_ids = [k for k, _ in items]
    centers_x = [v[0] for _, v in items]
    centers_y = [v[1] for _, v in items]

    query_points = np.column_stack([centers_x, centers_y])
    nearest_distances, _, nearest_node_ids = nodes_tree.nearest_node(centers_x, centers_y)
    nearest_distances = np.asarray(nearest_distances, dtype=float)
    nearest_node_ids = np.asarray(nearest_node_ids)

    final_node_ids = nearest_node_ids.copy()
    nearest_is_restricted = np.isin(nearest_node_ids, list(restricted_node_ids))

    allowed_nodes = network.nodes.loc[~network.nodes["node_id"].isin(restricted_node_ids), ["node_id", "x", "y"]]
    if np.any(nearest_is_restricted) and not allowed_nodes.empty:
        allowed_coords = allowed_nodes[["x", "y"]].to_numpy()
        allowed_node_ids = allowed_nodes["node_id"].to_numpy()
        allowed_tree = cKDTree(allowed_coords)

        allowed_distances, allowed_indices = allowed_tree.query(query_points, k=1)
        allowed_distances = np.asarray(allowed_distances)
        allowed_indices = np.asarray(allowed_indices)
        allowed_nearest_node_ids = allowed_node_ids[allowed_indices]

        # Reassign only if the non-restricted alternative is close enough in absolute
        # distance and not much farther than the original nearest node.
        use_allowed = (
            nearest_is_restricted
            & (allowed_distances <= float(max_non_restricted_distance_m))
            & (allowed_distances <= np.maximum(1.0, nearest_distances) * float(max_non_restricted_distance_ratio))
        )
        final_node_ids[use_allowed] = allowed_nearest_node_ids[use_allowed]

        logger.info(
            "H3 center snapping: %s restricted nearest nodes, %s reassigned (<= %.1fm and ratio <= %.2f)",
            int(np.sum(nearest_is_restricted)),
            int(np.sum(use_allowed)),
            float(max_non_restricted_distance_m),
            float(max_non_restricted_distance_ratio),
        )

    node_coordinates = network.nodes.set_index("node_id")[["x", "y"]]

    return {
        h3_id: (
            float(node_coordinates.at[node_id, "x"]),
            float(node_coordinates.at[node_id, "y"]),
        )
        for h3_id, node_id in zip(h3_ids, final_node_ids)
    }


def deduplicate_h3_centers(matrix, h3_to_index, node_to_index, h3_centers):
    ordered_h3_ids = [h3_id for h3_id, _ in sorted(h3_to_index.items(), key=lambda x: x[1])]
    index_to_h3 = {idx: h3_id for h3_id, idx in h3_to_index.items()}

    representative_by_center = {}
    alias_h3 = {}
    unique_h3_ids = []

    for h3_id in ordered_h3_ids:
        center = (float(h3_centers[h3_id][0]), float(h3_centers[h3_id][1]))
        representative = representative_by_center.get(center)
        if representative is None:
            representative_by_center[center] = h3_id
            alias_h3[h3_id] = h3_id
            unique_h3_ids.append(h3_id)
        else:
            alias_h3[h3_id] = representative

    if len(unique_h3_ids) == len(ordered_h3_ids):
        return matrix, h3_to_index, node_to_index, h3_centers

    new_h3_to_index = {h3_id: idx for idx, h3_id in enumerate(unique_h3_ids)}
    new_h3_centers = {h3_id: h3_centers[h3_id] for h3_id in unique_h3_ids}
    new_matrix = np.zeros((len(unique_h3_ids), len(unique_h3_ids)), dtype=matrix.dtype)

    new_node_to_index = {}
    for node_id, old_idx in node_to_index.items():
        old_h3 = index_to_h3[old_idx]
        representative = alias_h3[old_h3]
        new_node_to_index[node_id] = new_h3_to_index[representative]

    logger.info(
        "Deduplicated snapped h3 centers: %s -> %s cells",
        len(ordered_h3_ids),
        len(unique_h3_ids),
    )

    return new_matrix, new_h3_to_index, new_node_to_index, new_h3_centers

########################### KD TREE #########################
class Tree:
    def __init__(self, kdtree, id_lookup):
        self.kdtree = kdtree
        self.id_lookup = id_lookup

    def nearest_node(self, x: Union[List[float], float], y: Union[List[float], float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        query_point = np.atleast_2d(np.array([x, y]).T)
        distances, indices = self.kdtree.query(query_point, k=1)
        indices = np.atleast_1d(indices)
        distances = np.atleast_1d(distances)
        return distances, indices, self.id_lookup[indices]


def build_kdtree(df, x="x", y="y", crs="EPSG:2056", object_id="node_id"):
    # build kdtree
    coords = np.array(list(zip(df[x], df[y])))
    kdtree = cKDTree(coords)
    id_lookup = np.array(df[object_id])
    return Tree(kdtree, id_lookup)


########################### DETOUR FACTORS #########################
def compute_detour_factors(matrix, graph, h3_to_index, node_to_index, h3_centers, nodes_tree,
                            imp_name="length", epsilon=0.0, chunk_size=None,
                            max_distance_threshold_m=None, min_distance_threshold_m=None,
                            far_detour_factor=1.0, near_detour_factor=1.0,
                            origin_batch_size=16):

    if (
        min_distance_threshold_m is not None
        and max_distance_threshold_m is not None
        and min_distance_threshold_m >= max_distance_threshold_m
    ):
        raise ValueError("min_distance_threshold_m must be smaller than max_distance_threshold_m")

    h3_ids = list(h3_to_index.keys())
    n = len(h3_ids)
 
    # precompute the nearest network node for every h3 center at once (vectorized kdtree query)
    centers_x = [h3_centers[h3_id][0] for h3_id in h3_ids]
    centers_y = [h3_centers[h3_id][1] for h3_id in h3_ids]
    _, nearest_nodes_idx, _ = nodes_tree.nearest_node(centers_x, centers_y)
    nearest_nodes_idx = np.asarray(nearest_nodes_idx)
 
    centers_xy = np.array([h3_centers[h3_id] for h3_id in h3_ids], dtype=float)  # shape (n, 2), row order == h3_ids order
    all_destinations = np.arange(n, dtype=np.int32)

    # Initialize defaults for non-routed OD pairs.
    if max_distance_threshold_m is not None:
        matrix.fill(float(far_detour_factor))
    centers_tree = cKDTree(centers_xy) if (
        max_distance_threshold_m is not None or min_distance_threshold_m is not None
    ) else None
 
    if origin_batch_size is None or origin_batch_size < 1:
        origin_batch_size = 1

    for origin_start in tqdm(range(0, n, origin_batch_size), desc="Computing detour factors"):
        origin_stop = min(origin_start + origin_batch_size, n)
        batched_origin_idx = []
        batched_destination_idx = []

        for origin_idx in range(origin_start, origin_stop):
            if max_distance_threshold_m is None:
                destination_idx = all_destinations[all_destinations != origin_idx]
            else:
                destination_idx = np.asarray(
                    centers_tree.query_ball_point(centers_xy[origin_idx], r=max_distance_threshold_m),
                    dtype=np.int32
                )
                destination_idx = destination_idx[destination_idx != origin_idx]

            if min_distance_threshold_m is not None:
                near_destination_idx = np.asarray(
                    centers_tree.query_ball_point(centers_xy[origin_idx], r=min_distance_threshold_m),
                    dtype=np.int32
                )
                near_destination_idx = near_destination_idx[near_destination_idx != origin_idx]
                if near_destination_idx.size > 0:
                    matrix[origin_idx, near_destination_idx] = float(near_detour_factor)

                if destination_idx.size > 0:
                    destination_distances = np.hypot(
                        centers_xy[origin_idx, 0] - centers_xy[destination_idx, 0],
                        centers_xy[origin_idx, 1] - centers_xy[destination_idx, 1],
                    )
                    destination_idx = destination_idx[destination_distances > float(min_distance_threshold_m)]

            if destination_idx.size == 0:
                continue

            batched_origin_idx.append(np.full(destination_idx.shape, origin_idx, dtype=np.int32))
            batched_destination_idx.append(destination_idx)

        if not batched_origin_idx:
            continue

        origin_idx = np.concatenate(batched_origin_idx)
        destination_idx = np.concatenate(batched_destination_idx)

        euclidean_distances = np.hypot(
            centers_xy[origin_idx, 0] - centers_xy[destination_idx, 0],
            centers_xy[origin_idx, 1] - centers_xy[destination_idx, 1],
        )

        if np.any(euclidean_distances == 0):
            bad_pos = int(np.argmax(euclidean_distances == 0))
            raise ValueError(
                f"Euclidean distance between h3 {h3_ids[int(origin_idx[bad_pos])]} and "
                f"{h3_ids[int(destination_idx[bad_pos])]} is zero."
            )

        destination_nodes = nearest_nodes_idx[destination_idx]
        origin_nodes = nearest_nodes_idx[origin_idx]

        if chunk_size is None:
            shortest_distances = np.asarray(
                graph.shortest_path_lengths(origin_nodes, destination_nodes, imp_name=imp_name),
                dtype=float
            )
        else:
            shortest_distances = np.empty(len(destination_idx), dtype=float)
            for start in range(0, len(destination_idx), chunk_size):
                end = start + chunk_size
                shortest_distances[start:end] = graph.shortest_path_lengths(
                    origin_nodes[start:end], destination_nodes[start:end], imp_name=imp_name
                )

        if np.any(np.isnan(shortest_distances)):
            bad_idx = np.where(np.isnan(shortest_distances))[0]
            bad_pairs = [
                (h3_ids[int(origin_idx[i])], h3_ids[int(destination_idx[i])]) for i in bad_idx[:5]
            ]
            raise ValueError(
                f"No shortest path found for {len(bad_idx)} OD pair(s), e.g. {bad_pairs}."
            )

        matrix[origin_idx, destination_idx] = shortest_distances / (euclidean_distances + epsilon)

        if origin_start % 1000 == 0 and origin_start > 0:
            logger.info("Computed detour factors for %s / %s origins", origin_start, n)

    np.fill_diagonal(matrix, 0)
 
    return matrix


class DetourFactorMatrix:
    """
    Small wrapper around the h3 x h3 detour factor matrix.

    Given arbitrary (x, y) origin / destination coordinates, it snaps them to their
    nearest h3 cell center (using a kdtree built on the h3 centers) and returns the
    precomputed detour factor for that cell pair.
    """

    def __init__(self, matrix, h3_to_index, h3_centers, h3_tree):
        self.matrix = matrix
        self.h3_to_index = h3_to_index
        self.h3_centers = h3_centers
        self.h3_tree = h3_tree
        # Cache the KD-tree row -> matrix row mapping once. Coordinate lookups
        # can then stay fully vectorized without Python dictionary work.
        self._tree_cell_indices = np.fromiter(
            (h3_to_index[h3_id] for h3_id in h3_tree.id_lookup),
            dtype=np.int64,
            count=len(h3_tree.id_lookup),
        )

    def get_cell_indices(self, x, y):
        """Snap coordinates to matrix cells and return their integer indices."""
        # Older cached stage objects predate this derived lookup; initialize it
        # lazily so they remain usable after the code upgrade.
        if not hasattr(self, "_tree_cell_indices"):
            self._tree_cell_indices = np.fromiter(
                (self.h3_to_index[h3_id] for h3_id in self.h3_tree.id_lookup),
                dtype=np.int64,
                count=len(self.h3_tree.id_lookup),
            )
        _, tree_indices, _ = self.h3_tree.nearest_node(x, y)
        return self._tree_cell_indices[np.asarray(tree_indices, dtype=np.int64)]

    def get_detour_factor_by_index(self, origin_index, destination_index):
        """Return clipped factors for already-snapped matrix indices.

        This hot-path API lets callers cache candidate cell indices instead of
        repeating a KD-tree lookup for every prediction.
        """
        factors = np.asarray(self.matrix[origin_index, destination_index], dtype=float)
        factors = np.where(np.isfinite(factors), factors, 10.0)
        return np.clip(factors, 1.0, 10.0)

    def get_detour_factor(self, x_origin, y_origin, x_destination, y_destination):
        """
        Accepts either scalars (single OD pair) or equal-length array-likes
        (batch of OD pairs) and returns the corresponding detour factor(s).
        """
        idx_origin = self.get_cell_indices(x_origin, y_origin)
        idx_destination = self.get_cell_indices(x_destination, y_destination)
        factors = self.matrix[idx_origin, idx_destination]

        if np.isscalar(x_origin) or np.ndim(x_origin) == 0:
            return float(factors[0])
        return np.clip(factors, 1.0, 10.0)


def plot_long_detours_html(
    matrix,
    h3_to_index,
    h3_centers,
    output_path,
    threshold=4.0,
):
    if matrix.size == 0:
        logger.info("Skip long-detour plot: empty matrix")
        return

    n = matrix.shape[0]
    mask = (matrix > float(threshold)) & (~np.eye(n, dtype=bool))
    origin_idx, destination_idx = np.where(mask)

    if origin_idx.size == 0:
        logger.info("Skip long-detour plot: no OD pair above threshold=%s", threshold)
        return

    ordered_h3_ids = [h3_id for h3_id, _ in sorted(h3_to_index.items(), key=lambda x: x[1])]
    centers_xy = np.array([h3_centers[h3_id] for h3_id in ordered_h3_ids], dtype=float)

    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    start_lon, start_lat = transformer.transform(
        centers_xy[origin_idx, 0],
        centers_xy[origin_idx, 1],
    )
    end_lon, end_lat = transformer.transform(
        centers_xy[destination_idx, 0],
        centers_xy[destination_idx, 1],
    )

    detour_values = matrix[origin_idx, destination_idx]

    fig = go.Figure()
    scatter_cls = getattr(go, "Scattermap", None)
    use_mapbox = scatter_cls is None
    if use_mapbox:
        scatter_cls = go.Scattermapbox

    line_count = len(origin_idx)
    lons = np.empty(3 * line_count, dtype=object)
    lats = np.empty(3 * line_count, dtype=object)

    lons[0::3] = start_lon
    lons[1::3] = end_lon
    lons[2::3] = None

    lats[0::3] = start_lat
    lats[1::3] = end_lat
    lats[2::3] = None

    fig.add_trace(
        scatter_cls(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=1, color="gray"),
            opacity=0.4,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        scatter_cls(
            lon=start_lon,
            lat=start_lat,
            mode="markers",
            marker=dict(size=7, color="blue"),
            name="Start",
            hovertemplate="Start<br>Detour factor: %{customdata:.2f}<extra></extra>",
            customdata=detour_values,
        )
    )

    fig.add_trace(
        scatter_cls(
            lon=end_lon,
            lat=end_lat,
            mode="markers",
            marker=dict(size=7, color="red"),
            name="End",
            hovertemplate="End<br>Detour factor: %{customdata:.2f}<extra></extra>",
            customdata=detour_values,
        )
    )

    layout_kwargs = dict(
        title=f"OD pairs with Detour Factor > {threshold}",
        width=1200,
        height=850,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if use_mapbox:
        layout_kwargs["mapbox"] = dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(start_lat)), lon=float(np.mean(start_lon))),
            zoom=9,
        )
    else:
        layout_kwargs["map"] = dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(start_lat)), lon=float(np.mean(start_lon))),
            zoom=9,
        )

    fig.update_layout(**layout_kwargs)

    fig.write_html(output_path)
    logger.info("Saved long-detour plot with %s OD pairs to %s", line_count, output_path)
