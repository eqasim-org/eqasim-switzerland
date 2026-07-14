"""Build static_assets BLOBs (key / content_type / payload); the backend reads them by key.
Input geometry is EPSG:2056 (LV95); output GeoJSON is EPSG:4326 (WGS84)."""

from __future__ import annotations

import json
import logging

import duckdb

log = logging.getLogger(__name__)

GEOJSON_CT = "application/geo+json"
JSON_CT = "application/json"


def put_asset(db: duckdb.DuckDBPyConnection, key: str, content_type: str,
              payload: bytes) -> None:
    """Insert/replace one static_assets row."""
    db.execute("DELETE FROM static_assets WHERE key = ?", [key])
    db.execute(
        "INSERT INTO static_assets (key, content_type, payload) VALUES (?, ?, ?)",
        [key, content_type, payload],
    )


def _feature(geometry_geojson: str, properties: dict) -> dict:
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": json.loads(geometry_geojson),
    }


def build_municipalities_geojson(db: duckdb.DuckDBPyConnection) -> bool:
    """Gemeinde polygons as one WGS84 FeatureCollection with bfs/name/kantonsnum."""
    n = db.execute(
        "SELECT COUNT(*) FROM hot_polygons WHERE polygon_type = 'gemeinde'"
    ).fetchone()[0]
    if n == 0:
        log.info("static_assets: no gemeinde polygons - municipalities.geojson skipped")
        return False
    rows = db.execute("""
        WITH g AS (
            SELECT CAST(SPLIT_PART(polygon_id, ':', 2) AS INTEGER) AS bfs,
                   polygon_name AS name,
                   polygon_geom AS geom
            FROM hot_polygons WHERE polygon_type = 'gemeinde'
        )
        SELECT g.bfs, g.name,
               (SELECT MIN(CAST(SPLIT_PART(c.polygon_id, ':', 2) AS INTEGER))
                  FROM hot_polygons c
                 WHERE c.polygon_type = 'canton'
                   AND ST_Contains(c.polygon_geom, ST_PointOnSurface(g.geom))) AS kantonsnum,
               ST_AsGeoJSON(ST_Transform(g.geom, 'EPSG:2056', 'EPSG:4326', true)) AS gj
        FROM g
        ORDER BY g.bfs
    """).fetchall()
    features = [
        _feature(gj, {"bfs": bfs, "name": name, "kantonsnum": kn})
        for (bfs, name, kn, gj) in rows
    ]
    payload = json.dumps(
        {"type": "FeatureCollection", "features": features},
        ensure_ascii=False,
    ).encode("utf-8")
    put_asset(db, "municipalities", GEOJSON_CT, payload)
    log.info("static_assets: municipalities - %d features", len(features))
    return True


def build_merged_segments(db: duckdb.DuckDBPyConnection) -> int:
    """One GeoJSON FeatureCollection of road links per canton, key merged_segments:<canton_id>.

    One Feature per network_links row; properties carry link_id (also the ids in
    spider_routes.route_links) so the backend joins link volumes by plain id match."""
    n_links = db.execute("SELECT COUNT(*) FROM network_links").fetchone()[0]
    if n_links == 0:
        log.info("static_assets: merged_segments skipped (no network_links)")
        return 0
    rows = db.execute("""
        SELECT canton_id,
               link_id,
               road_type,
               freespeed,
               ST_AsGeoJSON(ST_Transform(geom, 'EPSG:2056', 'EPSG:4326', true)) AS gj
        FROM network_links
        WHERE geom IS NOT NULL AND canton_id IS NOT NULL
        ORDER BY canton_id, link_id
    """).fetchall()
    by_canton: dict[int, list] = {}
    for canton, link_id, road_type, freespeed, gj in rows:
        by_canton.setdefault(canton, []).append(
            _feature(gj, {"link_id": link_id, "road_type": road_type,
                          "freespeed": freespeed})
        )
    for canton, feats in by_canton.items():
        payload = json.dumps(
            {"type": "FeatureCollection", "features": feats},
            ensure_ascii=False,
        ).encode("utf-8")
        put_asset(db, f"merged_segments:{canton}", GEOJSON_CT, payload)
    log.info("static_assets: merged_segments - %d cantons, %d links total",
             len(by_canton), len(rows))
    return len(by_canton)


def build_nodes_by_canton(db: duckdb.DuckDBPyConnection) -> int:
    """network_nodes split into one WGS84 FeatureCollection per canton."""
    n_nodes = db.execute("SELECT COUNT(*) FROM network_nodes").fetchone()[0]
    n_canton = db.execute(
        "SELECT COUNT(*) FROM hot_polygons WHERE polygon_type = 'canton'"
    ).fetchone()[0]
    if n_nodes == 0 or n_canton == 0:
        log.info("static_assets: nodes_by_canton skipped (nodes=%d, cantons=%d)",
                 n_nodes, n_canton)
        return 0
    rows = db.execute("""
        SELECT CAST(SPLIT_PART(c.polygon_id, ':', 2) AS INTEGER) AS canton,
               n.node_id,
               ST_AsGeoJSON(ST_Transform(n.geom, 'EPSG:2056', 'EPSG:4326', true)) AS gj
        FROM network_nodes n
        JOIN hot_polygons c
          ON c.polygon_type = 'canton' AND ST_Contains(c.polygon_geom, n.geom)
        WHERE n.geom IS NOT NULL
        ORDER BY canton, n.node_id
    """).fetchall()
    by_canton: dict[int, list] = {}
    for canton, node_id, gj in rows:
        by_canton.setdefault(canton, []).append(
            _feature(gj, {"node_id": node_id})
        )
    for canton, feats in by_canton.items():
        payload = json.dumps(
            {"type": "FeatureCollection", "features": feats},
            ensure_ascii=False,
        ).encode("utf-8")
        put_asset(db, f"nodes_by_canton/{canton}_nodes.geojson", GEOJSON_CT, payload)
    log.info("static_assets: nodes_by_canton - %d cantons, %d nodes total",
             len(by_canton), len(rows))
    return len(by_canton)


def build_metadata_asset(
    db: duckdb.DuckDBPyConnection, *, sample_rate: float | None, run_name: str,
    scaled_to_full_population: bool,
) -> None:
    """Write static_assets['metadata'] JSON: sample_rate, run_name, scaled_to_full_population."""
    payload = json.dumps({
        "sample_rate": sample_rate,
        "run_name": run_name,
        "scaled_to_full_population": bool(scaled_to_full_population),
    }, ensure_ascii=False).encode("utf-8")
    put_asset(db, "metadata", JSON_CT, payload)
    log.info("static_assets: metadata - sample_rate=%s run_name=%s scaled=%s",
             sample_rate, run_name, bool(scaled_to_full_population))


def build_all(db: duckdb.DuckDBPyConnection) -> None:
    """Build the static assets; each sub-builder skips gracefully if inputs are absent.

    nodes_by_canton is intentionally NOT built - the backend generates node GeoJSON
    itself from network_nodes."""
    log.info("=== static_assets START")
    build_municipalities_geojson(db)
    build_merged_segments(db)
    log.info("=== static_assets DONE")
