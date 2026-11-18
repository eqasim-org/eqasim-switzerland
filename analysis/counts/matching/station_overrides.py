"""Resolve explicit station measurement groups to current directed network links."""
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .road_matching import angular_difference, local_geometry_angle

logger = logging.getLogger("synpp")


def station_key(value):
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return str(int(value)) if value.isdigit() else value


def match_station_overrides(matcher, network, count_data, overrides, max_distance=50, max_angle=45):
    if not np.isfinite(max_distance) or max_distance <= 0 or not 0 <= max_angle < 90:
        raise ValueError("Overrides require a positive search distance and an angle tolerance below 90 degrees")
    if network.crs != count_data.crs or not network.crs.is_projected or network.crs.to_epsg() != 2056:
        raise ValueError("ASTRA station overrides require counts and network in EPSG:2056")
    definitions = {station_key(key): groups for key, groups in overrides.items()}
    roads = network.get_geometry()
    rows = []
    for station in count_data.itertuples():
        groups = definitions.get(station_key(station.id))
        if not groups:
            raise ValueError(f"Station {station.id} has an empty override")
        if len({group['group'] for group in groups}) != len(groups):
            raise ValueError(f"Station {station.id} has duplicate override group names")
        resolved = []
        for group in groups:
            if group["direction"] not in (1, 2):
                raise ValueError(f"ASTRA direction must be 1 or 2 for station {station.id}")
            reference = Point(float(group["x"]), float(group["y"]))
            angle = float(group["angle"])
            if not np.isfinite([reference.x, reference.y, angle]).all():
                raise ValueError(f"Invalid override location/direction for station {station.id}")
            osm_id = matcher._normalise_osm_id(group["osm_id"])
            indices = roads.sindex.query(reference.buffer(max_distance), predicate="intersects")
            candidates = roads.iloc[indices][["link_id", "geometry"]].copy()
            candidates["link_id"] = candidates.link_id.astype(str)
            attributes = network.links.loc[network.links.link_id.astype(str).isin(candidates.link_id)].copy()
            matches = []
            endpoints = {}
            for record in attributes.itertuples():
                values = getattr(record, "attributes", {})
                values = values if isinstance(values, dict) else {}
                current = matcher._normalise_osm_id(getattr(record, "osm_id", None)) or matcher._normalise_osm_id(values.get("osm:way:id"))
                old = matcher._old_osm_ids(values.get("old_osm_id"))
                old += matcher._old_osm_ids(getattr(record, "old_osm_id", None))
                if current == osm_id or osm_id in old:
                    matches.append((str(record.link_id), "osm_id" if current == osm_id else "old_osm_id"))
                    endpoints[str(record.link_id)] = (
                        getattr(record, "from_node", getattr(record, "from_id", None)),
                        getattr(record, "to_node", getattr(record, "to_id", None)))
            candidates = candidates.merge(pd.DataFrame(matches, columns=["link_id", "override_match_method"]), on="link_id")
            if candidates.empty:
                break
            candidates["distance"] = candidates.geometry.distance(reference)
            candidates["angle_difference"] = candidates.geometry.map(
                lambda shape: angular_difference(angle, local_geometry_angle(shape, reference)))
            candidates = candidates[(candidates["distance"] <= max_distance) & (candidates.angle_difference <= max_angle)]
            if candidates.empty:
                break
            # Current OSM IDs win only among geographically/directionally valid
            # candidates; historical IDs still resolve merged and split versions.
            candidates["historical"] = candidates.override_match_method.eq("old_osm_id")
            selected = candidates.sort_values(["historical", "distance", "angle_difference", "link_id"], kind="stable").iloc[0]
            resolved.append(dict(id=station.id, geometry=station.geometry, link_id=selected.link_id,
                                 road_geometry=selected.geometry, distance=selected.distance,
                                 match_method="station_override", is_station_override=True,
                                 measurement_group=group["group"], reference_osm_id=osm_id,
                                 measurement_geometry=reference, direction_number=group["direction"],
                                 expected_links=len(groups), override_match_method=selected.override_match_method,
                                 matched_from_node=endpoints[selected.link_id][0], matched_to_node=endpoints[selected.link_id][1],
                                 angle_difference=selected.angle_difference))
        serial = any(a["direction_number"] == b["direction_number"]
                     and a["matched_to_node"] is not None
                     and a["matched_to_node"] == b["matched_from_node"]
                     for a in resolved for b in resolved if a is not b)
        if len(resolved) != len(groups) or len({row["link_id"] for row in resolved}) != len(groups) or serial:
            logger.warning("Station %s override incomplete, duplicated, or serial; leaving station unmatched (%d/%d groups).",
                           station.id, len(resolved), len(groups))
            continue
        rows.extend(resolved)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=count_data.crs) if rows else matcher._empty_matches(count_data.crs)
