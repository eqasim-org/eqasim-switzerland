"""Build static_assets BLOBs (key / content_type / payload); the backend reads them by key.
Input geometry is EPSG:2056 (LV95); output GeoJSON is EPSG:4326 (WGS84)."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict

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


_COORD_DECIMALS = 6  # ~0.1 m - plenty for the map; keeps the payload small


def _round_coords(geom: dict) -> dict:
    """Round LineString/MultiLineString coordinates in place to _COORD_DECIMALS.

    Rounding is deterministic, so a link and its reversed-coordinate twin still
    round to identical values - the geometry-key pairing of forward+reverse links
    is preserved."""
    t = geom.get("type")
    c = geom.get("coordinates")
    if not c:
        return geom
    if t == "LineString":
        geom["coordinates"] = [[round(x, _COORD_DECIMALS), round(y, _COORD_DECIMALS)] for x, y in c]
    elif t == "MultiLineString":
        geom["coordinates"] = [
            [[round(x, _COORD_DECIMALS), round(y, _COORD_DECIMALS)] for x, y in line]
            for line in c
        ]
    return geom


def _flat_coords(geom: dict):
    """Flatten a LineString/MultiLineString geometry to a [[x, y], ...] list."""
    t = geom.get("type")
    c = geom.get("coordinates")
    if t == "LineString":
        return c
    if t == "MultiLineString":
        return [pt for line in c for pt in line]
    return None


def _arrow_for_coords(coords) -> str:
    """Direction glyph for one link from its own coordinates. Westward
    (start lon > end lon) -> left arrow, otherwise right arrow; falls back to
    latitude for (near-)vertical links so a reversed pair still gets opposite
    glyphs."""
    if not coords or len(coords) < 2:
        return "→"
    s_lon, s_lat = coords[0][0], coords[0][1]
    e_lon, e_lat = coords[-1][0], coords[-1][1]
    if s_lon != e_lon:
        return "←" if s_lon > e_lon else "→"
    return "←" if s_lat > e_lat else "→"


def _geometry_key(coords) -> str:
    """Direction-independent geometry key: the smaller of the forward and reversed
    coordinate sequences, so a link and its reversed-coordinate twin hash to one
    bucket. Coords are already rounded deterministically, so the pairing is exact."""
    parts = [f"{x},{y}" for x, y in coords]
    fwd = ";".join(parts)
    rev = ";".join(reversed(parts))
    return fwd if fwd <= rev else rev


def _js_num(v) -> str:
    """Stringify a per-link scalar the way the client merge did (JSON number -> JS
    toString): integral floats lose the trailing '.0'. None -> empty string."""
    if v is None:
        return ""
    f = float(v)
    return str(int(f)) if f == int(f) else repr(f)


def _merged_segments_for_canton(db: duckdb.DuckDBPyConnection, canton_id: int) -> bytes | None:
    """Build the fat, merged FeatureCollection for one canton - a byte-for-byte
    port of the webmap backend's providers/network_geometry.merged_segments_geojson,
    so the stored asset equals what the backend would otherwise rebuild at runtime.

    Non-finite capacity/freespeed/length/permlanes are coerced to NULL so the
    payload is always valid JSON (some PT links carry freespeed = Infinity)."""
    rows = db.execute(
        """
        SELECT link_id, modes,
               CASE WHEN isfinite(capacity)  THEN ROUND(capacity, 1)  END AS capacity,
               CASE WHEN isfinite(freespeed) THEN ROUND(freespeed * 3.6, 2) END AS freespeed,
               CASE WHEN isfinite(length)    THEN ROUND(length, 2)    END AS length,
               CASE WHEN isfinite(permlanes) THEN permlanes            END AS permlanes,
               road_type,
               ST_AsGeoJSON(
                   ST_Transform(geom, 'EPSG:2056', 'EPSG:4326', always_xy := true)
               ) AS gj
        FROM network_links
        WHERE canton_id = ?
        """,
        [canton_id],
    ).fetchall()
    if not rows:
        return None

    # Group directed links by shared 2D geometry (forward + reverse -> one
    # segment). Insertion order is SQL row order, so per_id_* arrays come out in
    # the same order the client merge produced.
    groups: "OrderedDict[str, dict]" = OrderedDict()
    singletons = []  # degenerate geometries that can't merge; appended as-is
    for link_id, modes, capacity, freespeed, length, permlanes, road_type, gj in rows:
        if not gj:
            continue
        geom = _round_coords(json.loads(gj))
        coords = _flat_coords(geom)
        rep = {
            "link_id": link_id,
            "modes": modes,
            "capacity": capacity,
            "freespeed": freespeed,
            "length": length,
            "permlanes": permlanes,
            "road_type": road_type,
        }
        if not coords or len(coords) < 2:
            singletons.append({"type": "Feature", "properties": rep, "geometry": geom})
            continue
        key = _geometry_key(coords)
        grp = groups.get(key)
        if grp is None:
            grp = {"geometry": geom, "rep": rep,
                   "keys": [], "arrows": [], "freespeeds": [],
                   "capacities": [], "lengths": [], "permlanes": []}
            groups[key] = grp
        grp["keys"].append(str(link_id))
        grp["arrows"].append(_arrow_for_coords(coords))
        grp["freespeeds"].append(_js_num(freespeed))
        grp["capacities"].append(_js_num(capacity))
        grp["lengths"].append(_js_num(length))
        grp["permlanes"].append(_js_num(permlanes))

    # Merged segments first so features[0] always carries per_id_keys - the
    # frontend's no-op guard only inspects the first feature.
    features = []
    for grp in groups.values():
        features.append({
            "type": "Feature",
            "properties": {
                **grp["rep"],
                "per_id_keys": "|".join(grp["keys"]),
                "per_id_arrows": "|".join(grp["arrows"]),
                "per_id_freespeeds": "|".join(grp["freespeeds"]),
                "per_id_capacities": "|".join(grp["capacities"]),
                "per_id_lengths": "|".join(grp["lengths"]),
                "per_id_permlanes": "|".join(grp["permlanes"]),
            },
            "geometry": grp["geometry"],
        })
    features.extend(singletons)

    # Default json.dumps (ensure_ascii=True) so arrow glyphs serialize as
    # ←/→, matching the backend byte-for-byte.
    return json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")


def build_merged_segments(db: duckdb.DuckDBPyConnection) -> int:
    """One fat, merged GeoJSON FeatureCollection per canton, key merged_segments:<canton_id>.

    Ports webmap providers/network_geometry.merged_segments_geojson so the webmap
    consumes the stored asset directly instead of rebuilding it from network_links
    on every cold request. Forward+reverse links sharing a 2D geometry are merged
    into one feature carrying index-aligned per_id_* pipe arrays; scalar props
    (link_id, modes, capacity, freespeed[km/h], length, permlanes, road_type) come
    from the representative link."""
    n_links = db.execute("SELECT COUNT(*) FROM network_links").fetchone()[0]
    if n_links == 0:
        log.info("static_assets: merged_segments skipped (no network_links)")
        return 0
    cantons = [r[0] for r in db.execute(
        "SELECT DISTINCT canton_id FROM network_links "
        "WHERE canton_id IS NOT NULL ORDER BY canton_id"
    ).fetchall()]
    n_done = 0
    for canton in cantons:
        payload = _merged_segments_for_canton(db, canton)
        if payload is None:
            continue
        put_asset(db, f"merged_segments:{canton}", GEOJSON_CT, payload)
        n_done += 1
    log.info("static_assets: merged_segments (fat/merged) - %d cantons", n_done)
    return n_done


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
