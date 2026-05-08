"""Phase 4 — load swisstopo polygons + compute polygon-level aggregates.

Polygon types loaded (all from swisstopo TLM):
  canton   – swissBOUNDARIES3D_*_TLM_KANTONSGEBIET.shp
  bezirk   – swissBOUNDARIES3D_*_TLM_BEZIRKSGEBIET.shp
  gemeinde – swissBOUNDARIES3D_*_TLM_HOHEITSGEBIET.shp

Aggregates mirror demo_grid + trip_grid_origin + out_of_home column lists, so
polygon-Y queries are an O(1) lookup on the backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb

log = logging.getLogger(__name__)


_POLY_SPECS = {
    "canton":   ("KANTONSNUM", "NAME", None),
    "bezirk":   ("BEZIRKSNUM", "NAME", "KANTONSNUM"),
    "gemeinde": ("BFS_NUMMER", "NAME", "BEZIRKSNUM"),
}


def load_hot_polygons(
    db: duckdb.DuckDBPyConnection,
    canton_shp: Optional[Path],
    bezirk_shp: Optional[Path],
    gemeinde_shp: Optional[Path],
) -> list[str]:
    """Insert polygons from up to 3 swisstopo layers. Returns loaded types."""
    loaded: list[str] = []
    for ptype, shp in [("canton", canton_shp), ("bezirk", bezirk_shp), ("gemeinde", gemeinde_shp)]:
        if shp is None or not shp.exists():
            log.warning("hot_polygons: %s shapefile missing — skipping", ptype)
            continue
        id_col, name_col, parent_col = _POLY_SPECS[ptype]
        parent_expr = (
            f"'{_parent_type(ptype)}:' || CAST({parent_col} AS VARCHAR)"
            if parent_col else "NULL"
        )

        db.execute(f"""
            INSERT INTO hot_polygons (polygon_id, polygon_type, polygon_name, parent_id, polygon_geom)
            SELECT
                '{ptype}:' || CAST({id_col} AS VARCHAR) AS polygon_id,
                '{ptype}'                                AS polygon_type,
                {name_col}                               AS polygon_name,
                {parent_expr}                            AS parent_id,
                ST_Force2D(CAST(geom AS GEOMETRY))       AS polygon_geom
            FROM ST_Read('{shp}')
        """)
        loaded.append(ptype)
        n = db.execute(
            f"SELECT COUNT(*) FROM hot_polygons WHERE polygon_type = '{ptype}'"
        ).fetchone()[0]
        log.info("hot_polygons: loaded %d %s polygons", n, ptype)
    return loaded


def build_hot_polygon_demo(db: duckdb.DuckDBPyConnection) -> int:
    """Spatial join persons → hot_polygons; aggregate same cols as demo_grid_*."""
    db.execute("""
        INSERT INTO hot_polygon_demo
        WITH joined AS (
            SELECT hp.polygon_id, p.age, p.sex, p.car_availability,
                   p.has_driving_license, p.employed,
                   p.subscriptions_ga, p.subscriptions_halbtax, p.subscriptions_verbund,
                   p.subscriptions_strecke, p.subscriptions_gleis7,
                   p.subscriptions_junior, p.subscriptions_other,
                   h.n_cars_class, p.n_activities
            FROM persons p
            LEFT JOIN households h ON p.household_id = h.household_id
            JOIN hot_polygons hp ON ST_Contains(hp.polygon_geom, p.home_pt)
            WHERE p.home_pt IS NOT NULL
        )
        SELECT polygon_id,
            COUNT(*)::INTEGER AS n_persons,
            COUNT(*) FILTER (WHERE age IS NOT NULL AND age <  6)::INTEGER,
            COUNT(*) FILTER (WHERE age >=  6 AND age < 15)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 15 AND age < 18)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 18 AND age < 24)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 24 AND age < 30)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 30 AND age < 45)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 45 AND age < 65)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 65 AND age < 80)::INTEGER,
            COUNT(*) FILTER (WHERE age >= 80)::INTEGER,
            COUNT(*) FILTER (WHERE sex = 0)::INTEGER,
            COUNT(*) FILTER (WHERE sex = 1)::INTEGER,
            COUNT(*) FILTER (WHERE car_availability = 'always')::INTEGER,
            COUNT(*) FILTER (WHERE car_availability = 'sometimes')::INTEGER,
            COUNT(*) FILTER (WHERE car_availability = 'never')::INTEGER,
            COUNT(*) FILTER (WHERE has_driving_license)::INTEGER,
            COUNT(*) FILTER (WHERE employed)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_ga)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_halbtax)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_verbund)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_strecke)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_gleis7)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_junior)::INTEGER,
            COUNT(*) FILTER (WHERE subscriptions_other)::INTEGER,
            COUNT(*) FILTER (WHERE n_cars_class = '0')::INTEGER,
            COUNT(*) FILTER (WHERE n_cars_class = '1')::INTEGER,
            COUNT(*) FILTER (WHERE n_cars_class = '2')::INTEGER,
            COUNT(*) FILTER (WHERE n_cars_class = '3+')::INTEGER,
            COALESCE(SUM(n_activities), 0)::INTEGER
        FROM joined GROUP BY polygon_id
    """)
    n = db.execute("SELECT COUNT(*) FROM hot_polygon_demo").fetchone()[0]
    log.info("hot_polygon_demo: %d rows", n)
    return n


def build_hot_polygon_trips(db: duckdb.DuckDBPyConnection) -> int:
    time_buckets = ",\n            ".join(
        f"COUNT(*) FILTER (WHERE FLOOR(departure_time/3600) = {h})::INTEGER"
        for h in range(24)
    )
    db.execute(f"""
        INSERT INTO hot_polygon_trips
        WITH joined AS (
            SELECT hp.polygon_id, t.main_mode, t.following_purpose,
                   t.departure_time, t.network_distance, t.crowfly_distance
            FROM trips t
            JOIN hot_polygons hp ON ST_Contains(hp.polygon_geom, t.origin_pt)
            WHERE t.origin_pt IS NOT NULL
        )
        SELECT polygon_id,
            COUNT(*)::INTEGER AS n_trips,
            COUNT(*) FILTER (WHERE main_mode = 'car')::INTEGER,
            COUNT(*) FILTER (WHERE main_mode = 'pt')::INTEGER,
            COUNT(*) FILTER (WHERE main_mode = 'walk')::INTEGER,
            COUNT(*) FILTER (WHERE main_mode = 'bike')::INTEGER,
            COUNT(*) FILTER (WHERE main_mode = 'car_passenger')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'home')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'work')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'education')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'shop')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'leisure')::INTEGER,
            COUNT(*) FILTER (WHERE following_purpose = 'other')::INTEGER,
            COALESCE(SUM(network_distance), 0.0),
            COALESCE(SUM(crowfly_distance), 0.0),
            COUNT(*) FILTER (WHERE network_distance < 1000)::INTEGER,
            COUNT(*) FILTER (WHERE network_distance >= 1000  AND network_distance < 3000)::INTEGER,
            COUNT(*) FILTER (WHERE network_distance >= 3000  AND network_distance < 10000)::INTEGER,
            COUNT(*) FILTER (WHERE network_distance >= 10000 AND network_distance < 30000)::INTEGER,
            COUNT(*) FILTER (WHERE network_distance >= 30000)::INTEGER,
            {time_buckets}
        FROM joined GROUP BY polygon_id
    """)
    n = db.execute("SELECT COUNT(*) FROM hot_polygon_trips").fetchone()[0]
    log.info("hot_polygon_trips: %d rows", n)
    return n


def build_hot_polygon_out_of_home(db: duckdb.DuckDBPyConnection) -> int:
    away_select = ",\n            ".join(
        f"""COUNT(DISTINCT person_id) FILTER (
            WHERE COALESCE(start_time, -1) <= {h * 3600}
              AND COALESCE(end_time, 1e12) >  {h * 3600}
              AND purpose != 'home'
        )::INTEGER AS away_h{h}""" for h in range(24)
    )
    away_coalesce = ",\n               ".join(f"COALESCE(aa.away_h{h}, 0)" for h in range(24))
    db.execute(f"""
        INSERT INTO hot_polygon_out_of_home
        WITH person_poly AS (
            SELECT hp.polygon_id, p.person_id
            FROM persons p
            JOIN hot_polygons hp ON ST_Contains(hp.polygon_geom, p.home_pt)
            WHERE p.home_pt IS NOT NULL
        ),
        cell_pop AS (
            SELECT polygon_id, COUNT(*)::INTEGER AS n_persons
            FROM person_poly GROUP BY polygon_id
        ),
        joined AS (
            SELECT pp.polygon_id, a.person_id, a.purpose, a.start_time, a.end_time
            FROM person_poly pp JOIN activities a USING (person_id)
        ),
        away_agg AS (
            SELECT polygon_id, {away_select} FROM joined GROUP BY polygon_id
        )
        SELECT cp.polygon_id, cp.n_persons,
               {away_coalesce}
        FROM cell_pop cp LEFT JOIN away_agg aa USING (polygon_id)
        WHERE cp.n_persons > 0
    """)
    n = db.execute("SELECT COUNT(*) FROM hot_polygon_out_of_home").fetchone()[0]
    log.info("hot_polygon_out_of_home: %d rows", n)
    return n


def build_hot_polygon_flows(db: duckdb.DuckDBPyConnection) -> int:
    """Trips classified by origin AND dest polygon (canton level if available).

    Uses the smallest polygon_type present (gemeinde > bezirk > canton). For
    >1 polygon types, only one is chosen — backend can flow at finer levels via
    raw R-Tree lookup if needed.
    """
    chosen = db.execute("""
        SELECT polygon_type FROM hot_polygons GROUP BY polygon_type
        ORDER BY CASE polygon_type WHEN 'gemeinde' THEN 1 WHEN 'bezirk' THEN 2 ELSE 3 END
        LIMIT 1
    """).fetchone()
    if chosen is None:
        log.warning("hot_polygon_flows: no polygons loaded — skipping")
        return 0
    ptype = chosen[0]
    db.execute(f"""
        INSERT INTO hot_polygon_flows
        WITH polys AS (
            SELECT polygon_id, polygon_geom FROM hot_polygons WHERE polygon_type = '{ptype}'
        )
        SELECT
            o.polygon_id AS origin_polygon_id,
            d.polygon_id AS dest_polygon_id,
            COUNT(*)::INTEGER AS n_trips,
            COUNT(*) FILTER (WHERE t.main_mode = 'car')::INTEGER,
            COUNT(*) FILTER (WHERE t.main_mode = 'pt')::INTEGER,
            COUNT(*) FILTER (WHERE t.main_mode = 'walk')::INTEGER,
            COUNT(*) FILTER (WHERE t.main_mode = 'bike')::INTEGER,
            COUNT(*) FILTER (WHERE t.main_mode = 'car_passenger')::INTEGER
        FROM trips t
        JOIN polys o ON ST_Contains(o.polygon_geom, t.origin_pt)
        JOIN polys d ON ST_Contains(d.polygon_geom, t.dest_pt)
        WHERE t.origin_pt IS NOT NULL AND t.dest_pt IS NOT NULL
        GROUP BY origin_polygon_id, dest_polygon_id
    """)
    n = db.execute("SELECT COUNT(*) FROM hot_polygon_flows").fetchone()[0]
    log.info("hot_polygon_flows: %d pairs", n)
    return n


def _parent_type(ptype: str) -> str:
    return {"bezirk": "canton", "gemeinde": "bezirk"}.get(ptype, "")
