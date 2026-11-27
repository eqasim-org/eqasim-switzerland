"""Cache directed road shapes from the exact extended routing network."""

import polars as pl
import shapely


def attach_road_endpoints(requests, path):
    """Use permitted endpoint links, including for same-node short road trips.

    Node-based Java routing exports no itinerary when both points snap to one
    node. Explicit link requests retain a reusable single-link route instead.
    Resolve coincident directed links by their alignment with the OD direction.
    """
    import numpy as np

    links = pl.read_ipc(path).sort("link_id")
    shapes = shapely.from_wkb(links["geometry"].to_numpy())
    valid = shapely.length(shapes) > 1e-9
    direction = requests.select(
        pl.col("destination_x") - pl.col("origin_x"),
        pl.col("destination_y") - pl.col("origin_y"),
    ).to_numpy()
    output = {
        name: np.empty(requests.height, dtype=object)
        for name in ("origin_link_id", "destination_link_id")
    }
    for mode in ("car", "truck"):
        rows = np.flatnonzero(requests["router_mode"].to_numpy() == mode)
        if not len(rows):
            continue
        eligible = np.flatnonzero(valid & links["modes"].list.contains(mode).to_numpy())
        if not len(eligible):
            raise ValueError(
                f"Extended routing network contains no usable {mode} links"
            )
        tree = shapely.STRtree(shapes[eligible])
        for prefix in ("origin", "destination"):
            xy = requests.select(prefix + "_x", prefix + "_y").to_numpy()[rows]
            points = shapely.points(xy)
            matches = tree.query_nearest(points, all_matches=True)
            query, local = matches[0], matches[1]
            indices = eligible[local]
            fractions = shapely.line_locate_point(
                shapes[indices], points[query], normalized=True
            )
            before = shapely.get_coordinates(
                shapely.line_interpolate_point(
                    shapes[indices], np.maximum(0.0, fractions - 1e-5), normalized=True
                )
            )
            after = shapely.get_coordinates(
                shapely.line_interpolate_point(
                    shapes[indices], np.minimum(1.0, fractions + 1e-5), normalized=True
                )
            )
            heading = after - before
            heading /= np.maximum(np.linalg.norm(heading, axis=1)[:, None], 1e-12)
            score = -np.einsum("ij,ij->i", heading, direction[rows[query]])
            order = np.lexsort((indices, score, query))
            sorted_query = query[order]
            first = np.r_[True, sorted_query[1:] != sorted_query[:-1]]
            chosen = order[first]
            if len(chosen) != len(rows):
                raise ValueError(
                    f"Cannot resolve every {mode} endpoint on the extended network"
                )
            ids = links["link_id"].to_numpy()
            output[prefix + "_link_id"][rows[query[chosen]]] = ids[indices[chosen]]
    return requests.with_columns(
        pl.Series(name, values, dtype=pl.String) for name, values in output.items()
    )
