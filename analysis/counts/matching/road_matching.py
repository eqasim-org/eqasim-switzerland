"""Geometry helpers used to match traffic-count records to MATSim links."""

from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString


ROAD_TYPE_PRIORITY = (
    "motorway", "trunk", "primary", "motorway_link", "trunk_link",
    "primary_link", "secondary", "secondary_link", "tertiary",
    "tertiary_link", "unclassified", "residential", "living_street",
    "service", "track",
)


def geometry_angle(geometry) -> float:
    """Return a directed line angle in degrees, measured counter-clockwise from east."""
    if geometry is None or geometry.is_empty:
        return np.nan
    if isinstance(geometry, MultiLineString):
        start = geometry.geoms[0].coords[0]
        end = geometry.geoms[-1].coords[-1]
    elif isinstance(geometry, LineString):
        start = geometry.coords[0]
        end = geometry.coords[-1]
    else:
        raise TypeError(f"Expected a line geometry, got {geometry.geom_type}")
    return np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))


def angular_difference(first, second):
    """Return the smallest absolute difference between angles in degrees."""
    return np.abs((first - second + 180) % 360 - 180)


def lines_are_opposite(first_angle, second_angle, tolerance=15.0) -> bool:
    return angular_difference(first_angle, second_angle) >= 180.0 - tolerance


def select_opposite_pair(candidates: pd.DataFrame, tolerance=15.0) -> pd.DataFrame:
    """Select the closest pair of candidates travelling in opposite directions.

    An empty frame is returned when no valid pair exists. This makes the
    bidirectional contract strict: a station never silently receives one link.
    """
    valid_pairs = []
    rows = list(candidates.index)
    for first, second in combinations(rows, 2):
        if lines_are_opposite(
            candidates.at[first, "road_angle"],
            candidates.at[second, "road_angle"],
            tolerance,
        ):
            mean_distance = (
                candidates.at[first, "distance"] + candidates.at[second, "distance"]
            ) / 2.0
            valid_pairs.append((mean_distance, str(candidates.at[first, "link_id"]),
                                str(candidates.at[second, "link_id"]), first, second))

    if not valid_pairs:
        return candidates.iloc[0:0]
    _, _, _, first, second = min(valid_pairs)
    return candidates.loc[[first, second]]


def match_points_to_opposite_links(counts, roads, search_radius, tolerance=15.0):
    """Match each point count to exactly two nearby, opposite MATSim links."""
    if counts.empty or roads.empty:
        return _empty_matches(counts.crs)

    # `id` is the station key, so duplicate ids must not generate a non-unique
    # pandas index when we map geometries back to the matched rows.
    counts = counts.drop_duplicates(subset=["id"], keep="first").copy()

    buffered = counts[["id", "geometry"]].copy()
    buffered = buffered.set_geometry(buffered.geometry.buffer(search_radius))
    joined = gpd.sjoin(buffered, roads[["link_id", "geometry"]],
                       how="inner", predicate="intersects")
    if joined.empty:
        return _empty_matches(counts.crs)

    joined = joined.drop(columns=["index_right"], errors="ignore")
    station_geometry = counts.set_index("id").geometry
    joined["geometry"] = joined["id"].map(station_geometry)
    joined = joined.merge(
        roads[["link_id", "geometry"]].rename(columns={"geometry": "road_geometry"}),
        on="link_id",
        how="left",
    )
    joined["distance"] = joined.apply(
        lambda row: row["geometry"].distance(row["road_geometry"]), axis=1
    )
    joined["road_angle"] = joined["road_geometry"].map(geometry_angle)
    selected_pairs = [
        select_opposite_pair(group, tolerance=tolerance)
        for _, group in joined.groupby("id", sort=False)
    ]
    selected_pairs = [pair for pair in selected_pairs if not pair.empty]
    if not selected_pairs:
        return _empty_matches(counts.crs)
    matched = pd.concat(selected_pairs, ignore_index=True)
    return gpd.GeoDataFrame(
        matched[["id", "geometry", "link_id", "road_geometry", "distance"]],
        geometry="geometry",
        crs=counts.crs,
    )


def line_direction(reference, candidate, tolerance=20.0):
    difference = angular_difference(geometry_angle(reference), geometry_angle(candidate))
    if difference <= tolerance:
        return "same"
    if difference >= 180.0 - tolerance:
        return "opposite"
    return None


def sampled_line_distance(first, second, samples=10) -> float:
    """Symmetric sampled distance suitable for overlapping road geometries."""
    def distances(source, target):
        lines = list(source.geoms) if isinstance(source, MultiLineString) else [source]
        points = [line.interpolate(i / samples, normalized=True)
                  for line in lines if line.length > 0 for i in range(samples)]
        return np.array([target.distance(point) for point in points])

    first_to_second = distances(first, second)
    second_to_first = distances(second, first)
    if not len(first_to_second) or not len(second_to_first):
        return np.inf
    return min(first_to_second.mean(), second_to_first.mean())


def match_line_to_opposite_links(reference, roads, search_radius,
                                 orientation_tolerance=20.0,
                                 maximum_distance=10.0):
    """Return the closest same/opposite road pair for a line count geometry."""
    candidate_indices = list(roads.sindex.intersection(reference.buffer(search_radius).bounds))
    candidates = roads.iloc[candidate_indices].copy()
    if candidates.empty:
        return candidates
    candidates["direction"] = candidates.geometry.map(
        lambda road: line_direction(reference, road, orientation_tolerance)
    )
    candidates = candidates.dropna(subset=["direction"])
    candidates["distance"] = candidates.geometry.map(
        lambda road: sampled_line_distance(reference, road)
    )

    selected = []
    for direction in ("same", "opposite"):
        directional = candidates[candidates["direction"] == direction]
        if directional.empty:
            return candidates.iloc[0:0]
        best = directional.sort_values(["distance", "link_id"], kind="stable").iloc[[0]]
        if best.iloc[0].distance > maximum_distance:
            return candidates.iloc[0:0]
        selected.append(best)
    return pd.concat(selected, ignore_index=True)


def _empty_matches(crs):
    return gpd.GeoDataFrame(
        columns=["id", "geometry", "link_id", "road_geometry", "distance"],
        geometry="geometry",
        crs=crs,
    )
