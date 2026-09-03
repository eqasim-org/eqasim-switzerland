"""Public traffic-count matching API."""

import logging
from enum import Enum

import geopandas as gpd
import pandas as pd

from .road_matching import (
    ROAD_TYPE_PRIORITY,
    angular_difference,
    geometry_angle,
    match_line_to_opposite_links,
    match_points_to_opposite_links,
)

logger = logging.getLogger("synpp")


class MatchMode(str, Enum):
    """Number and direction of links represented by one observed flow value."""

    DIRECTIONAL = "directional"
    BIDIRECTIONAL = "bidirectional"


class TrafficDataMatcher:
    """Match count observations to MATSim links.

    Directional observations require manual ``osm_id`` and ``angle`` columns
    and produce exactly one directed link. Bidirectional observations represent
    an aggregate across both directions and produce exactly two opposite links.
    """

    def match(self, network, counts, mode=None, search_radius=10,
              prioritize_road_types=False):
        has_manual_direction = {"osm_id", "angle"}.issubset(counts.counts.columns)
        if mode is None:
            if not has_manual_direction:
                raise ValueError(
                    "Matching mode is required for counts without osm_id and angle. "
                    "Use mode='bidirectional' for flows aggregated across directions."
                )
            mode = MatchMode.DIRECTIONAL
        else:
            mode = MatchMode(mode)

        if mode is MatchMode.DIRECTIONAL:
            if not has_manual_direction:
                raise ValueError(
                    "Directional matching requires osm_id and angle in the count data."
                )
            matched = self._match_directional(network, counts)
        else:
            matched = self._match_bidirectional(
                network, counts, search_radius, prioritize_road_types
            )

        expected_links = 1 if mode is MatchMode.DIRECTIONAL else 2
        cardinality = matched.groupby("id").size()
        if not cardinality.empty and not cardinality.eq(expected_links).all():
            raise RuntimeError(
                f"{mode.value} matching must return exactly {expected_links} link(s) per station."
            )

        total = counts.counts["id"].nunique()
        found = matched["id"].nunique()
        logger.info("Matched %d of %d %s count stations.", found, total, mode.value)
        return matched.reset_index(drop=True)

    def _match_directional(self, network, counts):
        count_data = counts.counts[["id", "geometry", "osm_id", "angle"]].copy()
        count_data["_osm_key"] = count_data["osm_id"].map(self._normalise_osm_id)
        count_data["angle"] = pd.to_numeric(count_data["angle"], errors="coerce")

        links = network.links[["link_id"]].copy()
        if "osm_id" in network.links.columns:
            links["osm_id"] = network.links["osm_id"]
        else:
            links["osm_id"] = network.links["attributes"].map(
                lambda attributes: attributes.get("osm:way:id")
                if isinstance(attributes, dict) else None
            )
        links["link_id"] = links["link_id"].astype(str)
        links["_osm_key"] = links["osm_id"].map(self._normalise_osm_id)
        links = links[links["_osm_key"].isin(count_data["_osm_key"].dropna())]

        roads = self._network_roads(network)
        roads = roads[roads["link_id"].isin(links["link_id"])]
        roads = roads.merge(links[["link_id", "_osm_key"]], on="link_id", how="inner")
        roads["road_angle"] = roads["geometry"].map(geometry_angle)
        roads = roads.rename(columns={"geometry": "road_geometry"})

        candidates = count_data.merge(
            roads[["link_id", "road_geometry", "_osm_key", "road_angle"]],
            on="_osm_key",
            how="left",
        ).dropna(subset=["link_id", "angle", "road_angle"])
        if candidates.empty:
            return self._empty_matches(counts.counts.crs)
        candidates["angle_difference"] = angular_difference(
            candidates["angle"], candidates["road_angle"]
        )
        candidates = candidates.sort_values(
            ["id", "angle_difference", "link_id"], kind="stable"
        ).drop_duplicates("id")
        candidates["distance"] = candidates.apply(
            lambda row: row["geometry"].distance(row["road_geometry"]), axis=1
        )
        return gpd.GeoDataFrame(
            candidates[["id", "geometry", "link_id", "road_geometry", "distance"]],
            geometry="geometry",
            crs=counts.counts.crs,
        )

    def _match_bidirectional(self, network, counts, search_radius, prioritize_road_types):
        count_data = gpd.GeoDataFrame(
            counts.counts[["id", "geometry"]].copy(),
            geometry="geometry",
            crs=counts.counts.crs,
        )
        if count_data.empty:
            return self._empty_matches(count_data.crs)
        if network.crs != count_data.crs:
            raise ValueError("Count and MATSim network CRS do not match.")

        roads = self._local_roads(network, count_data, search_radius)
        if count_data.geom_type.eq("Point").all():
            if prioritize_road_types:
                return self._match_points_by_road_priority(
                    network, count_data, search_radius
                )
            return match_points_to_opposite_links(count_data, roads, search_radius)
        if count_data.geom_type.isin(["LineString", "MultiLineString"]).all():
            return self._match_lines(count_data, roads, search_radius)
        raise ValueError("A count dataset must contain only points or only lines.")

    def _match_points_by_road_priority(self, network, counts, search_radius):
        remaining = counts.copy()
        _search_radius = search_radius
        results = []
        for road_type in ROAD_TYPE_PRIORITY:
            roads = network.get_ways(road_types=[road_type])
            roads = self._restrict_to_network_links(network, roads)
            matched = match_points_to_opposite_links(remaining, roads, _search_radius)
            if not matched.empty:
                results.append(matched)
                remaining = remaining[~remaining["id"].isin(matched["id"])]
            if remaining.empty or road_type=='secondary':
                break
            _search_radius = max(search_radius/4,_search_radius*0.7)

        return pd.concat(results, ignore_index=True) if results else self._empty_matches(counts.crs)

    def _match_lines(self, counts, roads, search_radius):
        rows = []
        for station in counts.itertuples(index=False):
            links = match_line_to_opposite_links(station.geometry, roads, search_radius)
            for link in links.itertuples(index=False):
                rows.append({
                    "id": station.id,
                    "geometry": station.geometry,
                    "link_id": link.link_id,
                    "road_geometry": link.geometry,
                    "distance": link.distance,
                })
        return gpd.GeoDataFrame(rows, columns=self._match_columns(),
                                geometry="geometry", crs=counts.crs)

    def _local_roads(self, network, counts, search_radius):
        west, south, east, north = counts.total_bounds
        buffer = 100 + search_radius
        roads = network.get_geometry().cx[
            west - buffer:east + buffer, south - buffer:north + buffer
        ]
        return self._restrict_to_network_links(network, roads)

    def _network_roads(self, network):
        return self._restrict_to_network_links(network, network.get_geometry())

    @staticmethod
    def _restrict_to_network_links(network, roads):
        roads = roads[["link_id", "geometry"]].copy()
        roads["link_id"] = roads["link_id"].astype(str)
        network_link_ids = network.links["link_id"].astype(str)
        return roads[roads["link_id"].isin(network_link_ids)].reset_index(drop=True)

    @staticmethod
    def _normalise_osm_id(value):
        if pd.isna(value):
            return None
        value = str(value).strip()
        return value[:-2] if value.endswith(".0") else value

    @classmethod
    def _empty_matches(cls, crs):
        return gpd.GeoDataFrame(columns=cls._match_columns(), geometry="geometry", crs=crs)

    @staticmethod
    def _match_columns():
        return ["id", "geometry", "link_id", "road_geometry", "distance"]
