"""Find true directed country transitions on detailed network link geometries."""

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import intersects, prepare
from shapely.geometry import Point

COLUMNS = ["link_id", "from_node", "to_node", "crossing_index", "crossing_x", "crossing_y",
           "direction", "border_x", "border_y", "right_offset_m", "foreign_x", "foreign_y", "highway", "modes", "ch_border_crossing", "geometry", "crossing_point", "border_point"]


def intersection_points(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "Point":
        return [geometry]
    if geometry.geom_type in {"LineString", "LinearRing"}:
        return [Point(geometry.coords[0]), Point(geometry.coords[-1])]
    if hasattr(geometry, "geoms"):
        return [point for part in geometry.geoms for point in intersection_points(part)]
    return []


def crossing_links(links, geometries, country):
    """Return one record per directed link/country transition.

    Touches and links wholly along the boundary are excluded. A border node
    counts when incident link interiors exist on both sides; this includes
    networks split precisely at the country boundary. Curves that enter and
    leave the country retain both transitions even with same-country endpoints.
    """
    boundary = country.boundary
    prepare(boundary)
    candidates = np.flatnonzero(intersects(boundary, geometries.array))
    records = []
    node_sides = defaultdict(set)
    for position in candidates:
        link = links.iloc[position]
        shape = geometries.iloc[position]
        if shape.length == 0:
            continue
        cuts = sorted({0., shape.length, *(shape.project(point) for point in
                         intersection_points(shape.intersection(boundary)))})
        states = []
        samples = []
        for start, end in zip(cuts[:-1], cuts[1:]):
            point = shape.interpolate((start + end) / 2)
            samples.append(point)
            states.append(None if boundary.distance(point) < 1e-7 else bool(country.contains(point)))
        interior = [state for state in states if state is not None]
        if not interior:
            continue
        if boundary.distance(Point(shape.coords[0])) < 1e-7:
            node_sides[str(link.from_node)].add(interior[0])
        if boundary.distance(Point(shape.coords[-1])) < 1e-7:
            node_sides[str(link.to_node)].add(interior[-1])
        records.append((link, shape, cuts, states, samples))
    output = []
    for link, shape, cuts, states, samples in records:
        crossing_index = 0
        attributes = link.get("attributes", {})
        attributes = attributes if isinstance(attributes, dict) else {}
        highway = link.get("highway")
        highway = highway if isinstance(highway, str) else str(attributes.get("osm:way:highway", "unknown"))
        for i, distance in enumerate(cuts):
            point = shape.interpolate(distance)
            if boundary.distance(point) > 1e-7:
                continue
            left = next((j for j in range(i - 1, -1, -1) if states[j] is not None), None)
            right = next((j for j in range(i, len(states)) if states[j] is not None), None)
            before = states[left] if left is not None else None
            after = states[right] if right is not None else None
            if i == 0 and node_sides[str(link.from_node)] == {False, True}:
                before = not after
            if i == len(cuts) - 1 and node_sides[str(link.to_node)] == {False, True}:
                after = not before
            if before is None or after is None or before == after:
                continue
            # A boundary-following segment has one transition at its final end,
            # rather than duplicate pseudo-crossings along the shared boundary.
            if i < len(states) and states[i] is None:
                continue
            if before is False and left is not None:
                foreign = shape.interpolate(cuts[left + 1] - min(1., (cuts[left + 1] - cuts[left]) / 2))
            elif after is False and right is not None:
                foreign = shape.interpolate(cuts[right] + min(1., (cuts[right + 1] - cuts[right]) / 2))
            else:
                foreign = point
            shifted = shift_right(shape, distance)
            output.append({"link_id": str(link.link_id), "from_node": str(link.from_node),
                "to_node": str(link.to_node), "crossing_index": crossing_index,
                "crossing_x": shifted.x, "crossing_y": shifted.y, "direction": "in" if after else "out",
                "border_x": point.x, "border_y": point.y, "right_offset_m": 1.0,
                "foreign_x": foreign.x, "foreign_y": foreign.y, "highway": highway,
                "modes": link_modes(link.get("modes", "")), "ch_border_crossing": True,
                "geometry": shape.wkt, "crossing_point": shifted.wkt, "border_point": point.wkt})
            crossing_index += 1
    return pd.DataFrame(output, columns=COLUMNS)


def write_crossings(frame, path):
    """Write LV95 crossing points and full directed curves as compressed GeoParquet."""
    scalars = frame.drop(columns=["geometry", "crossing_point", "border_point"]).copy()
    scalars["link_id"] = scalars.link_id.astype(str)
    result = gpd.GeoDataFrame(scalars, geometry=gpd.GeoSeries.from_wkt(
        frame.crossing_point, crs=2056), crs=2056)
    result["border_geometry"] = gpd.GeoSeries.from_wkt(frame.border_point, crs=2056)
    result["link_geometry"] = gpd.GeoSeries.from_wkt(frame.geometry, crs=2056)
    result.to_parquet(path, index=False, compression="zstd")


def shift_right(shape, distance):
    """Offset one metre to the right of the local from-node → to-node tangent.

    Use the forward segment at a vertex, or the last segment at the line end.
    Repeated vertices are ignored. LV95 coordinates make the offset metric.
    """
    vectors = np.diff(np.asarray(shape.coords)[:, :2], axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    vectors, lengths = vectors[lengths > 0], lengths[lengths > 0]
    if not len(lengths):
        raise ValueError("Cannot shift a zero-length border link")
    index = min(np.searchsorted(np.cumsum(lengths), distance, side="right"), len(lengths) - 1)
    dx, dy = vectors[index] / lengths[index]
    point = shape.interpolate(distance)
    return Point(point.x + dy, point.y - dx)


def link_modes(value):
    """Keep every allowed mode, using comma-separated text in the network export."""
    if isinstance(value, (set, list, tuple, np.ndarray)):
        return ",".join(sorted(str(mode) for mode in value))
    return "" if value is None or pd.isna(value) else str(value)
