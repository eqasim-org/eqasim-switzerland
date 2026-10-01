"""Canonical canton assignment: one ST_Contains pass against the swisstopo canton
polygons in hot_polygons. Run after hot_polygons and the network tables are loaded."""

from __future__ import annotations

import logging

import duckdb

log = logging.getLogger(__name__)


def _build_canton_poly(db: duckdb.DuckDBPyConnection) -> int:
    """Materialise (canton_id, geom) helper + RTREE. Returns canton count."""
    n = db.execute(
        "SELECT COUNT(*) FROM hot_polygons WHERE polygon_type = 'canton'"
    ).fetchone()[0]
    if n == 0:
        return 0
    db.execute("DROP TABLE IF EXISTS _canton_poly")
    db.execute("""
        CREATE TABLE _canton_poly AS
        SELECT CAST(SPLIT_PART(polygon_id, ':', 2) AS INTEGER) AS canton_id,
               polygon_geom AS geom
        FROM hot_polygons
        WHERE polygon_type = 'canton'
    """)
    db.execute("CREATE INDEX rtree_canton_poly ON _canton_poly USING RTREE (geom)")
    return n


def _drop_canton_poly(db: duckdb.DuckDBPyConnection) -> None:
    db.execute("DROP TABLE IF EXISTS _canton_poly")


def assign_persons_canton(db: duckdb.DuckDBPyConnection) -> int:
    """Overwrite persons.canton_id from the canton polygons (synthetic)."""
    db.execute("""
        UPDATE persons
        SET canton_id = m.canton_id
        FROM (
            SELECT p.person_id, MIN(cp.canton_id) AS canton_id
            FROM persons p
            JOIN _canton_poly cp ON ST_Contains(cp.geom, p.home_pt)
            WHERE p.home_pt IS NOT NULL
            GROUP BY p.person_id
        ) m
        WHERE persons.person_id = m.person_id
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM persons WHERE canton_id IS NOT NULL"
    ).fetchone()[0]
    log.info("canton: persons with canton_id = %d", n)
    return n


def assign_trips_canton(db: duckdb.DuckDBPyConnection) -> int:
    """Populate trips.origin_canton_id + dest_canton_id."""
    db.execute("""
        UPDATE trips
        SET origin_canton_id = m.origin_canton_id,
            dest_canton_id   = m.dest_canton_id
        FROM (
            SELECT t.person_id, t.trip_index,
                   MIN(oc.canton_id) AS origin_canton_id,
                   MIN(dc.canton_id) AS dest_canton_id
            FROM trips t
            LEFT JOIN _canton_poly oc ON ST_Contains(oc.geom, t.origin_pt)
            LEFT JOIN _canton_poly dc ON ST_Contains(dc.geom, t.dest_pt)
            GROUP BY t.person_id, t.trip_index
        ) m
        WHERE trips.person_id = m.person_id AND trips.trip_index = m.trip_index
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM trips WHERE origin_canton_id IS NOT NULL"
    ).fetchone()[0]
    log.info("canton: trips with origin_canton_id = %d", n)
    return n


def assign_activities_canton(db: duckdb.DuckDBPyConnection) -> int:
    """Populate activities.canton_id from each activity's location point."""
    db.execute("""
        UPDATE activities
        SET canton_id = m.canton_id
        FROM (
            SELECT a.person_id, a.activity_index, MIN(cp.canton_id) AS canton_id
            FROM activities a
            JOIN _canton_poly cp ON ST_Contains(cp.geom, a.location_pt)
            WHERE a.location_pt IS NOT NULL
            GROUP BY a.person_id, a.activity_index
        ) m
        WHERE activities.person_id = m.person_id
          AND activities.activity_index = m.activity_index
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM activities WHERE canton_id IS NOT NULL"
    ).fetchone()[0]
    log.info("canton: activities with canton_id = %d", n)
    return n


def assign_network_nodes_canton(db: duckdb.DuckDBPyConnection) -> int:
    """Populate network_nodes.canton_id from each node's point."""
    db.execute("""
        UPDATE network_nodes
        SET canton_id = m.canton_id
        FROM (
            SELECT nn.node_id, MIN(cp.canton_id) AS canton_id
            FROM network_nodes nn
            JOIN _canton_poly cp ON ST_Contains(cp.geom, nn.geom)
            WHERE nn.geom IS NOT NULL
            GROUP BY nn.node_id
        ) m
        WHERE network_nodes.node_id = m.node_id
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM network_nodes WHERE canton_id IS NOT NULL"
    ).fetchone()[0]
    log.info("canton: network_nodes with canton_id = %d", n)
    return n


def assign_network_links_canton(db: duckdb.DuckDBPyConnection) -> int:
    """Populate network_links.canton_id from each link's centroid."""
    db.execute("""
        UPDATE network_links
        SET canton_id = m.canton_id
        FROM (
            SELECT nl.link_id, MIN(cp.canton_id) AS canton_id
            FROM network_links nl
            JOIN _canton_poly cp ON ST_Contains(cp.geom, ST_Centroid(nl.geom))
            WHERE nl.geom IS NOT NULL
            GROUP BY nl.link_id
        ) m
        WHERE network_links.link_id = m.link_id
    """)
    n = db.execute(
        "SELECT COUNT(*) FROM network_links WHERE canton_id IS NOT NULL"
    ).fetchone()[0]
    log.info("canton: network_links with canton_id = %d", n)
    return n


def assign_canton_ids(
    db: duckdb.DuckDBPyConnection,
    *,
    source_type: str,
    overwrite_persons: bool,
    has_trips: bool,
    has_network: bool,
    has_activities: bool = False,
) -> bool:
    """Run the canonical canton assignment; returns False (no-op) when no canton polygons.

    One boundary source (_canton_poly) feeds all tables, so every canton_id in the
    file is mutually consistent."""
    n_canton = _build_canton_poly(db)
    if n_canton == 0:
        log.warning("canton: no canton polygons loaded - canton_id left as-is")
        return False
    log.info("canton: assigning from %d swisstopo canton polygons (source=%s)",
             n_canton, source_type)
    try:
        if overwrite_persons:
            assign_persons_canton(db)
        if has_trips:
            assign_trips_canton(db)
        if has_activities:
            assign_activities_canton(db)
        if has_network:
            assign_network_links_canton(db)
            assign_network_nodes_canton(db)
    finally:
        _drop_canton_poly(db)
    return True
