"""Performance pre-aggregates for the synthetic build, keyed on H3 hex IDs.
Run after Phase 5: requires spider_link_index, network_links, persons, and trips.
"""

from __future__ import annotations

import logging

import duckdb

log = logging.getLogger(__name__)


def build_spider_link_volumes_by_hex_res6(db: duckdb.DuckDBPyConnection) -> int:
    """Pre-aggregate link traversals by the traveller's home res-6 hex."""
    db.execute("""
        INSERT INTO spider_link_volumes_by_hex_res6 (home_h3_index, link_id, n_traversals)
        SELECT p.home_h3_res6 AS home_h3_index,
               sli.link_id,
               COUNT(*)::INTEGER AS n_traversals
        FROM spider_link_index sli
        JOIN persons p ON p.person_id = sli.person_id
        WHERE p.home_h3_res6 IS NOT NULL
        GROUP BY p.home_h3_res6, sli.link_id
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM spider_link_volumes_by_hex_res6"
    ).fetchone()[0]
    log.info("spider_link_volumes_by_hex_res6: %d rows", n)
    return n


def build_zone_flow_link_volumes_hex_res6(db: duckdb.DuckDBPyConnection) -> int:
    """Pre-aggregate link traversals by the trip's (origin, dest) res-6 hex pair."""
    db.execute("""
        INSERT INTO zone_flow_link_volumes_hex_res6
            (origin_h3_index, dest_h3_index, link_id, n_trips)
        SELECT t.origin_h3_res6 AS origin_h3_index,
               t.dest_h3_res6   AS dest_h3_index,
               sli.link_id,
               COUNT(*)::INTEGER AS n_trips
        FROM spider_link_index sli
        JOIN trips t
          ON t.person_id = sli.person_id AND t.trip_index = sli.trip_index
        WHERE t.origin_h3_res6 IS NOT NULL
          AND t.dest_h3_res6 IS NOT NULL
        GROUP BY t.origin_h3_res6, t.dest_h3_res6, sli.link_id
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM zone_flow_link_volumes_hex_res6"
    ).fetchone()[0]
    log.info("zone_flow_link_volumes_hex_res6: %d rows", n)
    return n


def build_node_flow_matrix(db: duckdb.DuckDBPyConnection) -> int:
    """Turning-movement counts per network node (node = to_node of the first link
    in each consecutive route pair, which equals from_node of the second)."""
    db.execute("""
        INSERT INTO node_flow_matrix (node_id, from_link, to_link, n_trips)
        WITH consecutive AS (
            SELECT
                person_id, trip_index, position,
                link_id AS from_link,
                LEAD(link_id) OVER (
                    PARTITION BY person_id, trip_index
                    ORDER BY position
                ) AS to_link
            FROM spider_link_index
        )
        SELECT nl.to_node AS node_id,
               c.from_link,
               c.to_link,
               COUNT(*)::INTEGER AS n_trips
        FROM consecutive c
        JOIN network_links nl ON nl.link_id = c.from_link
        WHERE c.to_link IS NOT NULL
        GROUP BY nl.to_node, c.from_link, c.to_link
    """)
    n = db.execute("SELECT COUNT(*) FROM node_flow_matrix").fetchone()[0]
    log.info("node_flow_matrix: %d rows", n)
    return n


def build_all(db: duckdb.DuckDBPyConnection) -> None:
    """Run the three pre-aggregate builds."""
    log.info("=== pre-aggregates START")
    build_spider_link_volumes_by_hex_res6(db)
    build_zone_flow_link_volumes_hex_res6(db)
    build_node_flow_matrix(db)
    log.info("=== pre-aggregates DONE")
