import gzip
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point
try:
    from lxml import etree as ET
except ImportError:
    import xml.etree.ElementTree as ET

from data.cross_border.destinations import make_entry_border_facility_id, make_exit_border_facility_id


logger = logging.getLogger("synpp")

BORDER_METRIC_CANDIDATE_LIMIT = 500
BORDER_CROSSING_SEARCH_RADII = [500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
LOCAL_FALLBACK_SEARCH_RADII = [250.0, 500.0, 1000.0, 2000.0, 3000.0]
POINT_CLASSIFICATION_PARALLEL_THRESHOLD = 100000
POINT_CLASSIFICATION_CHUNK_SIZE = 100000
POINT_CLASSIFICATION_MAX_WORKERS = 8


def _local_tag(tag):
    # MATSim files can be namespace-free or namespace-qualified depending on writer.
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def _read_network_nodes_links(network_path):
    nodes = []
    links = []

    logger.info("Reading network nodes and links from %s ...", network_path)
    with gzip.open(network_path, "rb") as stream:
        tree = ET.iterparse(stream, events=["start"])

        for _, elem in tree:
            tag = _local_tag(elem.tag)
            if tag == "node":
                nodes.append({
                    "node_id": str(elem.attrib["id"]),
                    "x": float(elem.attrib["x"]),
                    "y": float(elem.attrib["y"]),
                })
                # Clearing as we stream keeps memory bounded on full Switzerland networks.
                elem.clear()
            elif tag == "link":
                links.append({
                    "link_id": str(elem.attrib["id"]),
                    "from_node": str(elem.attrib["from"]),
                    "to_node": str(elem.attrib["to"]),
                    "modes": str(elem.attrib.get("modes", "")),
                })
                # Clearing as we stream keeps memory bounded on full Switzerland networks.
                elem.clear()

    logger.info("Read %d network nodes and %d network links.", len(nodes), len(links))
    return pd.DataFrame.from_records(nodes), pd.DataFrame.from_records(links)


def _as_foreign_resident_border_facilities(df_cross_border_destinations):
    if df_cross_border_destinations is None:
        return pd.DataFrame(columns=["facility_id", "geometry", "direction"])

    # The cross-border destination stage keeps one row per person, so flatten its
    # entry and exit columns into the facility rows that need directional links.
    entry = df_cross_border_destinations[[
        "entry_interview_point_id", "entry_interview_geometry_point"
    ]].copy()
    entry = entry.rename(columns={
        "entry_interview_point_id": "facility_id",
        "entry_interview_geometry_point": "geometry",
    })
    entry["direction"] = "entry"

    exit = df_cross_border_destinations[[
        "exit_interview_point_id", "exit_interview_geometry_point"
    ]].copy()
    exit = exit.rename(columns={
        "exit_interview_point_id": "facility_id",
        "exit_interview_geometry_point": "geometry",
    })
    exit["direction"] = "exit"

    return pd.concat([entry, exit], ignore_index=True, sort=False)


def _as_swiss_resident_border_facilities(df_swiss_residents_od):
    if df_swiss_residents_od is None:
        return pd.DataFrame(columns=["facility_id", "geometry", "direction"])

    # Swiss residents can be sampled as either home -> border or border -> home,
    # so every matched border-crossing record needs both directional facilities.
    entry = df_swiss_residents_od[["cross_border_person_id", "border_crossing_point"]].copy()
    entry["facility_id"] = entry["cross_border_person_id"].astype(str).apply(make_entry_border_facility_id)
    entry = entry.rename(columns={"border_crossing_point": "geometry"})
    entry["direction"] = "entry"

    exit = df_swiss_residents_od[["cross_border_person_id", "border_crossing_point"]].copy()
    exit["facility_id"] = exit["cross_border_person_id"].astype(str).apply(make_exit_border_facility_id)
    exit = exit.rename(columns={"border_crossing_point": "geometry"})
    exit["direction"] = "exit"

    return pd.concat([
        entry[["facility_id", "geometry", "direction"]],
        exit[["facility_id", "geometry", "direction"]],
    ], ignore_index=True, sort=False)


def _as_border_facilities(df_cross_border_destinations, df_swiss_residents_od):
    facilities = gpd.GeoDataFrame(
        pd.concat([
            _as_foreign_resident_border_facilities(df_cross_border_destinations),
            _as_swiss_resident_border_facilities(df_swiss_residents_od),
        ], ignore_index=True, sort=False),
        geometry="geometry",
        crs="EPSG:2056",
    )

    # Shared interview crossings appear many times, but one facility ID only needs
    # one link assignment.
    facilities = facilities.dropna(subset=["facility_id", "geometry"])
    facilities["facility_id"] = facilities["facility_id"].astype(str)
    return facilities.drop_duplicates("facility_id").reset_index(drop=True)


def _is_car_link(modes):
    # MATSim stores modes as comma-separated strings in the network XML.
    return "car" in str(modes).split(",")


def _filter_links_near_facilities(nodes, links, facilities, search_radius):
    if facilities is None or len(facilities) == 0:
        return links

    # Numeric bounding-box prefiltering avoids buffering/testing the full Swiss
    # border geometry, which is very expensive for hundreds of thousands of nodes.
    node_lookup = nodes.set_index("node_id")[["x", "y"]]
    link_nodes = (
        links[["from_node", "to_node"]]
        .merge(node_lookup, left_on="from_node", right_index=True)
        .merge(node_lookup, left_on="to_node", right_index=True, suffixes=("_from", "_to"))
    )

    keep = np.zeros(len(links), dtype=bool)
    x_from = link_nodes["x_from"].to_numpy()
    y_from = link_nodes["y_from"].to_numpy()
    x_to = link_nodes["x_to"].to_numpy()
    y_to = link_nodes["y_to"].to_numpy()

    for point in facilities["geometry"].drop_duplicates():
        minx = point.x - search_radius
        maxx = point.x + search_radius
        miny = point.y - search_radius
        maxy = point.y + search_radius
        keep |= (
            ((x_from >= minx) & (x_from <= maxx) & (y_from >= miny) & (y_from <= maxy)) |
            ((x_to >= minx) & (x_to <= maxx) & (y_to >= miny) & (y_to <= maxy))
        )

    if keep.any():
        return links.loc[keep].copy()

    logger.warning("No car links found near border facilities; falling back to all car links.")
    return links


def _point_chunks(count, chunk_size):
    for start in range(0, count, chunk_size):
        yield start, min(start + chunk_size, count)


def _points_inside_ch(swiss_border, x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    count = len(x)

    # Shapely 2 provides a vectorized point-in-polygon path that avoids building
    # GeoSeries objects for every endpoint. intersects_xy counts boundary points
    # as inside, matching the previous within() | touches() behavior.
    if hasattr(shapely, "intersects_xy"):
        workers = min(
            POINT_CLASSIFICATION_MAX_WORKERS,
            os.cpu_count() or 1,
            max(1, math.ceil(count / POINT_CLASSIFICATION_CHUNK_SIZE)),
        )
        if count < POINT_CLASSIFICATION_PARALLEL_THRESHOLD or workers <= 1:
            return shapely.intersects_xy(swiss_border, x, y)

        chunks = list(_point_chunks(count, POINT_CLASSIFICATION_CHUNK_SIZE))
        inside = np.empty(count, dtype=bool)

        logger.info(
            "Classifying %d points against the Swiss border in %d chunks using %d workers ...",
            count,
            len(chunks),
            workers,
        )

        def classify_chunk(start, end):
            # Shapely's vectorized GEOS predicate is the expensive part here; the
            # chunk wrapper lets large network-node batches run concurrently.
            values = shapely.intersects_xy(swiss_border, x[start:end], y[start:end])
            return start, end, np.asarray(values, dtype=bool)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(classify_chunk, start, end) for start, end in chunks]

            for chunk_index, future in enumerate(as_completed(futures), start=1):
                start, end, values = future.result()
                inside[start:end] = values

                if chunk_index == 1 or chunk_index % 2 == 0 or chunk_index == len(futures):
                    logger.info(
                        "Classified point chunk %d/%d against the Swiss border.",
                        chunk_index,
                        len(futures),
                    )

        return inside

    points = gpd.GeoSeries(gpd.points_from_xy(x, y), crs="EPSG:2056")
    return points.within(swiss_border) | points.touches(swiss_border)


def _attach_inside_flags(candidates, swiss_border, inside_cache):
    # Classify only the endpoints that are actually considered for a facility.
    # The same network nodes appear repeatedly, so cache results by node_id.
    endpoints = pd.concat([
        candidates[["from_node", "x_from", "y_from"]].rename(columns={
            "from_node": "node_id",
            "x_from": "x",
            "y_from": "y",
        }),
        candidates[["to_node", "x_to", "y_to"]].rename(columns={
            "to_node": "node_id",
            "x_to": "x",
            "y_to": "y",
        }),
    ], ignore_index=True).drop_duplicates("node_id")

    missing = endpoints[~endpoints["node_id"].isin(inside_cache)]
    if len(missing) > 0:
        inside = _points_inside_ch(
            swiss_border,
            missing["x"].to_numpy(),
            missing["y"].to_numpy(),
        )
        inside_cache.update(dict(zip(missing["node_id"], inside.astype(bool))))

    candidates = candidates.copy()
    candidates["from_inside_ch"] = candidates["from_node"].map(inside_cache).astype(bool)
    candidates["to_inside_ch"] = candidates["to_node"].map(inside_cache).astype(bool)
    return candidates


def _merge_link_endpoint_coordinates(nodes, links):
    return (
        links.merge(nodes, left_on="from_node", right_on="node_id")
        .merge(nodes, left_on="to_node", right_on="node_id", suffixes=("_from", "_to"))
    )[[
        "link_id", "from_node", "to_node", "x_from", "y_from", "x_to", "y_to",
    ]]


def _build_candidate_link_sets(network_path, swiss_border, facilities=None, facility_search_radius=3000.0):
    # The patcher has a tiny network reader so it does not depend on optional reader
    # dependencies while matsim.simulation.prepare is imported.
    nodes, links = _read_network_nodes_links(network_path)
    links = links[links["modes"].apply(_is_car_link)].copy().reset_index(drop=True)
    if len(links) == 0:
        raise RuntimeError("Cannot assign directional cross-border facilities: network has no car links.")
    logger.info("Found %d car links before border-near filtering.", len(links))

    all_links = _merge_link_endpoint_coordinates(nodes, links)
    inside_cache = {}

    logger.info("Classifying endpoints of all car links against the Swiss border ...")
    all_links = _attach_inside_flags(all_links, swiss_border, inside_cache)
    logger.info("Classified %d unique car-link endpoints against the Swiss border.", len(inside_cache))

    # These are the primary candidates: one-way car links that really connect the
    # two sides of the Swiss polygon according to their directed endpoints.
    crossing_links = all_links[all_links["from_inside_ch"] != all_links["to_inside_ch"]].copy()
    logger.info("Found %d directed car links whose endpoints cross the Swiss border.", len(crossing_links))
    if len(crossing_links) == 0:
        logger.warning("No endpoint-crossing car links found; all border facilities will use local fallback matching.")

    # Keep a local Swiss-side candidate set for networks that are clipped at the
    # border or have geometry precision issues around the official boundary.
    logger.info("Filtering car links to endpoints within %.0f m of target border facilities ...", facility_search_radius)
    local_links = _filter_links_near_facilities(nodes, links, facilities, facility_search_radius)
    logger.info("Kept %d car links near target border facilities.", len(local_links))

    local_links = _merge_link_endpoint_coordinates(nodes, local_links)
    local_links = _attach_inside_flags(local_links, swiss_border, inside_cache)
    logger.info("Prepared local fallback endpoint table for %d candidate links.", len(local_links))

    return all_links, crossing_links, local_links, inside_cache


def _point_to_segment_distances(point, links):
    px = point.x
    py = point.y
    x1 = links["x_from"].to_numpy()
    y1 = links["y_from"].to_numpy()
    x2 = links["x_to"].to_numpy()
    y2 = links["y_to"].to_numpy()

    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy

    with np.errstate(divide="ignore", invalid="ignore"):
        t = ((px - x1) * dx + (py - y1) * dy) / length_squared

    t = np.nan_to_num(t, nan=0.0)
    t = np.clip(t, 0.0, 1.0)

    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return np.hypot(px - nearest_x, py - nearest_y)


def _candidate_links(point, links, radius, allow_global=False):
    # Directional border matching must not stop at the first nearby inner link:
    # the true crossing link can be slightly farther from the interview point.
    minx = point.x - radius
    maxx = point.x + radius
    miny = point.y - radius
    maxy = point.y + radius
    segment_bbox = (
        (np.minimum(links["x_from"], links["x_to"]) <= maxx) &
        (np.maximum(links["x_from"], links["x_to"]) >= minx) &
        (np.minimum(links["y_from"], links["y_to"]) <= maxy) &
        (np.maximum(links["y_from"], links["y_to"]) >= miny)
    )

    if segment_bbox.any():
        candidates = links[segment_bbox].copy()
        candidates["distance"] = _point_to_segment_distances(point, candidates)
        candidates = candidates[candidates["distance"] <= radius]

        if len(candidates) > 0:
            return candidates.sort_values("distance")

    if not allow_global:
        return links.head(0).assign(distance=pd.Series(dtype=float))

    # Fall back to a global nearest scan only for unusual network/border gaps.
    candidates = links.copy()
    candidates["distance"] = _point_to_segment_distances(point, candidates)
    return candidates.sort_values("distance").head(100)


def _endpoint_distances_to_border(candidates, swiss_border):
    boundary = swiss_border.boundary

    # The local candidate set is small by this point, so plain Shapely distance
    # calls keep the code robust across Shapely 1 and 2 installations.
    from_distances = []
    to_distances = []
    crosses_border = []
    for link in candidates.itertuples(index=False):
        from_point = Point(link.x_from, link.y_from)
        to_point = Point(link.x_to, link.y_to)
        line = LineString([(link.x_from, link.y_from), (link.x_to, link.y_to)])
        from_distances.append(from_point.distance(boundary))
        to_distances.append(to_point.distance(boundary))
        crosses_border.append(line.intersects(boundary))

    candidates = candidates.copy()
    candidates["from_border_distance"] = from_distances
    candidates["to_border_distance"] = to_distances
    candidates["crosses_swiss_border"] = crosses_border
    return candidates


def _attach_match_metadata(link, match_type):
    link = link.copy()
    link["match_type"] = match_type
    if "from_border_distance" not in link:
        link["from_border_distance"] = np.nan
    if "to_border_distance" not in link:
        link["to_border_distance"] = np.nan
    if "crosses_swiss_border" not in link:
        link["crosses_swiss_border"] = bool(link["from_inside_ch"] != link["to_inside_ch"])
    return link


def _fallback_direction_score(candidates, point, swiss_border):
    # If no endpoint clearly crosses the border, use the vector toward the Swiss
    # interior to pick the closest link that points approximately in the right way.
    interior = swiss_border.representative_point()
    target_x = interior.x - point.x
    target_y = interior.y - point.y
    target_norm = math.hypot(target_x, target_y)

    link_x = candidates["x_to"] - candidates["x_from"]
    link_y = candidates["y_to"] - candidates["y_from"]
    link_norm = np.hypot(link_x, link_y)

    if target_norm == 0:
        return np.zeros(len(candidates))

    with np.errstate(divide="ignore", invalid="ignore"):
        score = (link_x * target_x + link_y * target_y) / (link_norm * target_norm)

    return np.nan_to_num(score, nan=0.0)


def _find_endpoint_crossing_link(facility, crossing_links):
    for radius in BORDER_CROSSING_SEARCH_RADII:
        candidates = _candidate_links(facility.geometry, crossing_links, radius)
        if len(candidates) == 0:
            continue

        if facility.direction == "entry":
            directional = candidates[~candidates["from_inside_ch"] & candidates["to_inside_ch"]]
        else:
            directional = candidates[candidates["from_inside_ch"] & ~candidates["to_inside_ch"]]

        if len(directional) > 0:
            return directional.iloc[0]

    return None


def _choose_directional_link(facility, crossing_links, local_links, swiss_border, inside_cache):
    point = facility.geometry
    best_border_distance = None
    best_fallback = None

    # First try the purpose-built crossing set. This avoids accepting a nearby
    # Swiss-side access road while the true directed border link is a bit farther.
    endpoint_crossing = _find_endpoint_crossing_link(facility, crossing_links)
    if endpoint_crossing is not None:
        endpoint_crossing = _endpoint_distances_to_border(endpoint_crossing.to_frame().T, swiss_border)
        return _attach_match_metadata(endpoint_crossing.iloc[0], "endpoint_crossing")

    for radius in LOCAL_FALLBACK_SEARCH_RADII:
        candidates = _candidate_links(point, local_links, radius)
        if len(candidates) == 0:
            continue

        candidates = _endpoint_distances_to_border(
            candidates.head(BORDER_METRIC_CANDIDATE_LIMIT),
            swiss_border,
        )
        if facility.direction == "entry":
            # When both endpoints are classified on the same side of the border,
            # still require the directed link to move from the border inward.
            directional = candidates[candidates["from_border_distance"] < candidates["to_border_distance"]]
        else:
            # Exit links are the opposite movement: from the interior back toward
            # the border-crossing location.
            directional = candidates[candidates["to_border_distance"] < candidates["from_border_distance"]]

        if len(directional) > 0:
            directional = directional.sort_values(["crosses_swiss_border", "distance"], ascending=[False, True])
            if bool(directional.iloc[0]["crosses_swiss_border"]):
                return _attach_match_metadata(directional.iloc[0], "border_intersection_direction")
            if best_border_distance is None:
                best_border_distance = directional.iloc[0]

        # Keep a last-resort candidate for networks where the crossing is missing
        # or geometry precision makes both endpoint and border-distance tests flat.
        local = candidates.head(25).copy()
        local["direction_score"] = _fallback_direction_score(local, point, swiss_border)
        if facility.direction == "exit":
            local["direction_score"] *= -1.0

        local = local[local["direction_score"] > 0.0]
        if best_fallback is None and len(local) > 0:
            best_fallback = local.sort_values(["direction_score", "distance"], ascending=[False, True]).iloc[0]

    if best_border_distance is not None:
        return _attach_match_metadata(best_border_distance, "border_distance_direction")

    if best_fallback is not None:
        logger.warning(
            "Using non-border directional fallback for facility %s (%s). Inspect this crossing in analysis output.",
            facility.facility_id,
            facility.direction,
        )
        return _attach_match_metadata(best_fallback, "fallback_no_border_direction")

    candidates = _candidate_links(point, local_links, 3000.0, allow_global=True)
    if len(candidates) == 0:
        raise RuntimeError(f"Cannot find any network link candidate for border facility {facility.facility_id}.")

    candidates = _attach_inside_flags(candidates, swiss_border, inside_cache)
    candidates = _endpoint_distances_to_border(candidates, swiss_border)
    logger.warning(
        "Using nearest-link fallback for facility %s (%s). No directional border candidate was found.",
        facility.facility_id,
        facility.direction,
    )
    return _attach_match_metadata(candidates.iloc[0], "fallback_nearest_link")


def assign_directional_border_links(df_cross_border_destinations, df_swiss_residents_od, network_path, swiss_border):
    assignments = build_directional_border_link_table(
        df_cross_border_destinations,
        df_swiss_residents_od,
        network_path,
        swiss_border,
    )
    return dict(zip(assignments["facility_id"], assignments["link_id"]))


def build_directional_border_link_table(df_cross_border_destinations, df_swiss_residents_od, network_path, swiss_border):
    facilities = _as_border_facilities(df_cross_border_destinations, df_swiss_residents_od)
    if len(facilities) == 0:
        return facilities.assign(link_id=None, link_geometry=None, distance=None)

    border_geometry = swiss_border.unary_union if hasattr(swiss_border, "unary_union") else swiss_border
    all_links, crossing_links, local_links, inside_cache = _build_candidate_link_sets(
        network_path,
        border_geometry,
        facilities=facilities,
    )

    # Build a tabular assignment so preparation can patch XML and analysis stages
    # can export the exact same directional matching result.
    logger.info("Assigning %d directional border facilities to links ...", len(facilities))
    rows = []
    link_lookup = all_links.set_index("link_id")
    for facility_index, facility in enumerate(facilities.itertuples(index=False), start=1):
        if facility_index == 1 or facility_index % 100 == 0 or facility_index == len(facilities):
            logger.info(
                "Assigning directional border facility %d/%d ...",
                facility_index,
                len(facilities),
            )

        matched_link = _choose_directional_link(facility, crossing_links, local_links, border_geometry, inside_cache)
        link_id = str(matched_link["link_id"])
        link = link_lookup.loc[link_id]
        link_geometry = LineString([(link.x_from, link.y_from), (link.x_to, link.y_to)])
        rows.append({
            "facility_id": facility.facility_id,
            "direction": facility.direction,
            "geometry": facility.geometry,
            "link_id": link_id,
            "link_geometry": link_geometry,
            "distance": facility.geometry.distance(link_geometry),
            "from_inside_ch": bool(matched_link["from_inside_ch"]),
            "to_inside_ch": bool(matched_link["to_inside_ch"]),
            "from_border_distance": float(matched_link["from_border_distance"]),
            "to_border_distance": float(matched_link["to_border_distance"]),
            "crosses_swiss_border": bool(matched_link["crosses_swiss_border"]),
            "match_type": matched_link["match_type"],
        })

    logger.info("Classified %d unique candidate link endpoints against the Swiss border.", len(inside_cache))

    assignments = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:2056")
    match_types = assignments["match_type"].value_counts().to_dict()
    logger.info("Directional border link match types: %s", match_types)

    base_facility_id = assignments["facility_id"].str.replace(r"_(entry|exit)$", "", regex=True)
    paired_links = assignments.assign(base_facility_id=base_facility_id).pivot_table(
        index="base_facility_id",
        columns="direction",
        values="link_id",
        aggfunc="first",
    )
    if {"entry", "exit"}.issubset(paired_links.columns):
        same_link_pairs = paired_links[paired_links["entry"] == paired_links["exit"]]
        if len(same_link_pairs) > 0:
            logger.warning(
                "%d border facilities still use the same link for entry and exit. Check match_type in the analysis CSV.",
                len(same_link_pairs),
            )

    return assignments


def _set_xml_attribute(line, attribute, value):
    # The prepared MATSim files place each activity/facility on one line; keeping a
    # streaming text patch avoids loading the full population XML in memory.
    pattern = rf'(\s{attribute}=")[^"]*(")'
    replacement = rf'\g<1>{value}\2'
    if re.search(pattern, line):
        return re.sub(pattern, replacement, line, count=1)

    end = "/>" if "/>" in line else ">"
    return line.replace(end, f' {attribute}="{value}"{end}', 1)


def _patch_facilities_xml(path, facility_links):
    tmp_path = f"{path}.tmp"
    patched = 0

    with gzip.open(path, "rt", encoding="utf-8") as source:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as target:
            for line in source:
                if "<facility " in line:
                    match = re.search(r'\sid="([^"]+)"', line)
                    if match and match.group(1) in facility_links:
                        line = _set_xml_attribute(line, "linkId", facility_links[match.group(1)])
                        patched += 1
                target.write(line)

    os.replace(tmp_path, path)
    return patched


def _patch_population_xml(path, facility_links):
    tmp_path = f"{path}.tmp"
    patched = 0

    with gzip.open(path, "rt", encoding="utf-8") as source:
        with gzip.open(tmp_path, "wt", encoding="utf-8") as target:
            for line in source:
                if "<activity " in line and ' facility="' in line:
                    match = re.search(r'\sfacility="([^"]+)"', line)
                    if match and match.group(1) in facility_links:
                        line = _set_xml_attribute(line, "link", facility_links[match.group(1)])
                        patched += 1
                target.write(line)

    os.replace(tmp_path, path)
    return patched


def patch_directional_border_links(
    df_cross_border_destinations,
    df_swiss_residents_od,
    swiss_border,
    network_path,
    facilities_path,
    population_path,
):
    facility_links = assign_directional_border_links(
        df_cross_border_destinations,
        df_swiss_residents_od,
        network_path,
        swiss_border,
    )

    if len(facility_links) == 0:
        logger.info("No directional cross-border facilities found to patch.")
        return

    patched_facilities = _patch_facilities_xml(facilities_path, facility_links)
    patched_activities = _patch_population_xml(population_path, facility_links)

    logger.info(
        "Patched %d cross-border facilities and %d cross-border activities with directional links.",
        patched_facilities,
        patched_activities,
    )
