"""Phase 3 — pre-aggregated grids.

All aggregations are done in pure DuckDB SQL: no Python loop over rows. Cells
are ``(row, col)`` derived from the LV95 origin (xmin, ymin) via FLOOR. Empty
cells are not persisted (sparse storage — typical 10M-points → ~50k cells).

Cell-ID encoding: ``(row << 24) | col``. With 24-bit columns, supports up to
16M columns per row — orders of magnitude beyond CH-bbox needs.
"""

from __future__ import annotations

import logging
from typing import Iterable

import duckdb

from .hilbert import CH_BBOX_LV95

log = logging.getLogger(__name__)

XMIN, YMIN, XMAX, YMAX = CH_BBOX_LV95


def _bbox_clause(point_col: str) -> str:
    """SQL WHERE-fragment to keep only points inside the CH bbox."""
    return (
        f"{point_col} IS NOT NULL "
        f"AND ST_X({point_col}) BETWEEN {XMIN} AND {XMAX} "
        f"AND ST_Y({point_col}) BETWEEN {YMIN} AND {YMAX}"
    )


def _cell_id_expr(x_expr: str, y_expr: str, resolution_m: int) -> str:
    """SQL fragment that computes cell_id from (x, y) coordinates.

    Coordinates must already be inside the CH bbox (callers add WHERE clauses
    via _bbox_clause); the cast to UBIGINT will fail otherwise.
    """
    col = f"CAST(({x_expr} - {XMIN}) / {resolution_m} AS UBIGINT)"
    row = f"CAST(({y_expr} - {YMIN}) / {resolution_m} AS UBIGINT)"
    return f"(({row} << 24) | {col})"


def _cell_geom_expr(cell_id_alias: str, resolution_m: int) -> str:
    """Reconstruct the cell polygon from a cell_id."""
    col = f"({cell_id_alias} & ((1 << 24) - 1))"
    row = f"({cell_id_alias} >> 24)"
    x0 = f"({col} * {resolution_m} + {XMIN})"
    y0 = f"({row} * {resolution_m} + {YMIN})"
    x1 = f"({x0} + {resolution_m})"
    y1 = f"({y0} + {resolution_m})"
    return f"ST_MakeEnvelope({x0}, {y0}, {x1}, {y1})"


def _cell_center_expr(cell_id_alias: str, resolution_m: int) -> str:
    col = f"({cell_id_alias} & ((1 << 24) - 1))"
    row = f"({cell_id_alias} >> 24)"
    cx = f"({col} * {resolution_m} + {XMIN} + {resolution_m / 2})"
    cy = f"({row} * {resolution_m} + {YMIN} + {resolution_m / 2})"
    return f"ST_Point({cx}, {cy})"


def build_demo_grid(db: duckdb.DuckDBPyConnection, resolution_m: int) -> int:
    """Aggregate persons → demo_grid_<r>m. Joins households for n_cars_class."""
    cell_id = _cell_id_expr("ST_X(p.home_pt)", "ST_Y(p.home_pt)", resolution_m)
    cell_geom = _cell_geom_expr("cell_id", resolution_m)
    cell_center = _cell_center_expr("cell_id", resolution_m)

    db.execute(f"""
        INSERT INTO demo_grid_{resolution_m}m
        WITH base AS (
            SELECT
                {cell_id} AS cell_id,
                p.age, p.sex, p.car_availability, p.has_driving_license, p.employed,
                p.subscriptions_ga, p.subscriptions_halbtax, p.subscriptions_verbund,
                p.subscriptions_strecke, p.subscriptions_gleis7,
                p.subscriptions_junior, p.subscriptions_other,
                h.n_cars_class,
                p.n_activities
            FROM persons p
            LEFT JOIN households h ON p.household_id = h.household_id
            WHERE {_bbox_clause('p.home_pt')}
        ),
        agg AS (
            SELECT
                cell_id,
                COUNT(*)::INTEGER AS n_persons,
                COUNT(*) FILTER (WHERE age IS NOT NULL AND age <  6)::INTEGER AS age_0_6,
                COUNT(*) FILTER (WHERE age >=  6 AND age < 15)::INTEGER AS age_6_15,
                COUNT(*) FILTER (WHERE age >= 15 AND age < 18)::INTEGER AS age_15_18,
                COUNT(*) FILTER (WHERE age >= 18 AND age < 24)::INTEGER AS age_18_24,
                COUNT(*) FILTER (WHERE age >= 24 AND age < 30)::INTEGER AS age_24_30,
                COUNT(*) FILTER (WHERE age >= 30 AND age < 45)::INTEGER AS age_30_45,
                COUNT(*) FILTER (WHERE age >= 45 AND age < 65)::INTEGER AS age_45_65,
                COUNT(*) FILTER (WHERE age >= 65 AND age < 80)::INTEGER AS age_65_80,
                COUNT(*) FILTER (WHERE age >= 80)::INTEGER AS age_80_plus,
                COUNT(*) FILTER (WHERE sex = 0)::INTEGER AS sex_male,
                COUNT(*) FILTER (WHERE sex = 1)::INTEGER AS sex_female,
                COUNT(*) FILTER (WHERE car_availability = 'always')::INTEGER AS car_avail_always,
                COUNT(*) FILTER (WHERE car_availability = 'sometimes')::INTEGER AS car_avail_sometimes,
                COUNT(*) FILTER (WHERE car_availability = 'never')::INTEGER AS car_avail_never,
                COUNT(*) FILTER (WHERE has_driving_license)::INTEGER AS has_driving_license,
                COUNT(*) FILTER (WHERE employed)::INTEGER AS employed,
                COUNT(*) FILTER (WHERE subscriptions_ga)::INTEGER AS subs_ga,
                COUNT(*) FILTER (WHERE subscriptions_halbtax)::INTEGER AS subs_halbtax,
                COUNT(*) FILTER (WHERE subscriptions_verbund)::INTEGER AS subs_verbund,
                COUNT(*) FILTER (WHERE subscriptions_strecke)::INTEGER AS subs_strecke,
                COUNT(*) FILTER (WHERE subscriptions_gleis7)::INTEGER AS subs_gleis7,
                COUNT(*) FILTER (WHERE subscriptions_junior)::INTEGER AS subs_junior,
                COUNT(*) FILTER (WHERE subscriptions_other)::INTEGER AS subs_other,
                COUNT(*) FILTER (WHERE n_cars_class = '0')::INTEGER AS cars_0,
                COUNT(*) FILTER (WHERE n_cars_class = '1')::INTEGER AS cars_1,
                COUNT(*) FILTER (WHERE n_cars_class = '2')::INTEGER AS cars_2,
                COUNT(*) FILTER (WHERE n_cars_class = '3+')::INTEGER AS cars_3_plus,
                COALESCE(SUM(n_activities), 0)::INTEGER AS sum_activities
            FROM base
            GROUP BY cell_id
        )
        SELECT
            cell_id,
            {cell_geom} AS cell_geom,
            {cell_center} AS cell_center,
            n_persons,
            age_0_6, age_6_15, age_15_18, age_18_24, age_24_30,
            age_30_45, age_45_65, age_65_80, age_80_plus,
            sex_male, sex_female,
            car_avail_always, car_avail_sometimes, car_avail_never,
            has_driving_license, employed,
            subs_ga, subs_halbtax, subs_verbund, subs_strecke,
            subs_gleis7, subs_junior, subs_other,
            cars_0, cars_1, cars_2, cars_3_plus,
            sum_activities
        FROM agg
        WHERE cell_id IS NOT NULL AND n_persons > 0
    """)
    n = db.execute(f"SELECT COUNT(*) FROM demo_grid_{resolution_m}m").fetchone()[0]
    log.info("demo_grid_%dm: %d cells", resolution_m, n)
    return n


def build_trip_grid_origin(db: duckdb.DuckDBPyConnection, resolution_m: int) -> int:
    cell_id = _cell_id_expr("ST_X(origin_pt)", "ST_Y(origin_pt)", resolution_m)
    cell_geom = _cell_geom_expr("cell_id", resolution_m)
    time_buckets = ",\n                ".join(
        f"COUNT(*) FILTER (WHERE FLOOR(departure_time/3600) = {h})::INTEGER AS time_h{h}"
        for h in range(24)
    )
    db.execute(f"""
        INSERT INTO trip_grid_origin_{resolution_m}m
        WITH base AS (
            SELECT
                {cell_id} AS cell_id,
                main_mode, following_purpose, departure_time,
                network_distance, crowfly_distance
            FROM trips
            WHERE {_bbox_clause('origin_pt')}
        ),
        agg AS (
            SELECT
                cell_id,
                COUNT(*)::INTEGER AS n_trips,
                COUNT(*) FILTER (WHERE main_mode = 'car')::INTEGER AS mode_car,
                COUNT(*) FILTER (WHERE main_mode = 'pt')::INTEGER AS mode_pt,
                COUNT(*) FILTER (WHERE main_mode = 'walk')::INTEGER AS mode_walk,
                COUNT(*) FILTER (WHERE main_mode = 'bike')::INTEGER AS mode_bike,
                COUNT(*) FILTER (WHERE main_mode = 'car_passenger')::INTEGER AS mode_car_passenger,
                COUNT(*) FILTER (WHERE following_purpose = 'home')::INTEGER AS purpose_home,
                COUNT(*) FILTER (WHERE following_purpose = 'work')::INTEGER AS purpose_work,
                COUNT(*) FILTER (WHERE following_purpose = 'education')::INTEGER AS purpose_education,
                COUNT(*) FILTER (WHERE following_purpose = 'shop')::INTEGER AS purpose_shop,
                COUNT(*) FILTER (WHERE following_purpose = 'leisure')::INTEGER AS purpose_leisure,
                COUNT(*) FILTER (WHERE following_purpose = 'other')::INTEGER AS purpose_other,
                COALESCE(SUM(network_distance), 0.0) AS sum_network_distance,
                COALESCE(SUM(crowfly_distance), 0.0) AS sum_crowfly_distance,
                COUNT(*) FILTER (WHERE network_distance < 1000)::INTEGER AS dist_bucket_0_1km,
                COUNT(*) FILTER (WHERE network_distance >= 1000 AND network_distance < 3000)::INTEGER AS dist_bucket_1_3km,
                COUNT(*) FILTER (WHERE network_distance >= 3000 AND network_distance < 10000)::INTEGER AS dist_bucket_3_10km,
                COUNT(*) FILTER (WHERE network_distance >= 10000 AND network_distance < 30000)::INTEGER AS dist_bucket_10_30km,
                COUNT(*) FILTER (WHERE network_distance >= 30000)::INTEGER AS dist_bucket_30_plus_km,
                {time_buckets}
            FROM base
            GROUP BY cell_id
        )
        SELECT cell_id, {cell_geom}, n_trips,
               mode_car, mode_pt, mode_walk, mode_bike, mode_car_passenger,
               purpose_home, purpose_work, purpose_education, purpose_shop,
               purpose_leisure, purpose_other,
               sum_network_distance, sum_crowfly_distance,
               dist_bucket_0_1km, dist_bucket_1_3km, dist_bucket_3_10km,
               dist_bucket_10_30km, dist_bucket_30_plus_km,
               {", ".join(f"time_h{h}" for h in range(24))}
        FROM agg WHERE cell_id IS NOT NULL AND n_trips > 0
    """)
    n = db.execute(f"SELECT COUNT(*) FROM trip_grid_origin_{resolution_m}m").fetchone()[0]
    log.info("trip_grid_origin_%dm: %d cells", resolution_m, n)
    return n


def build_flow_grid(db: duckdb.DuckDBPyConnection, resolution_m: int) -> int:
    o = _cell_id_expr("ST_X(origin_pt)", "ST_Y(origin_pt)", resolution_m)
    d = _cell_id_expr("ST_X(dest_pt)", "ST_Y(dest_pt)", resolution_m)
    o_geom = _cell_geom_expr("origin_cell_id", resolution_m)
    d_geom = _cell_geom_expr("dest_cell_id", resolution_m)
    db.execute(f"""
        INSERT INTO flow_grid_{resolution_m}m
        WITH agg AS (
            SELECT
                {o} AS origin_cell_id,
                {d} AS dest_cell_id,
                COUNT(*)::INTEGER AS n_trips,
                COUNT(*) FILTER (WHERE main_mode = 'car')::INTEGER AS mode_car,
                COUNT(*) FILTER (WHERE main_mode = 'pt')::INTEGER AS mode_pt,
                COUNT(*) FILTER (WHERE main_mode = 'walk')::INTEGER AS mode_walk,
                COUNT(*) FILTER (WHERE main_mode = 'bike')::INTEGER AS mode_bike,
                COUNT(*) FILTER (WHERE main_mode = 'car_passenger')::INTEGER AS mode_car_passenger
            FROM trips
            WHERE {_bbox_clause('origin_pt')} AND {_bbox_clause('dest_pt')}
            GROUP BY origin_cell_id, dest_cell_id
        )
        SELECT origin_cell_id, dest_cell_id,
               {o_geom} AS origin_cell_geom,
               {d_geom} AS dest_cell_geom,
               n_trips, mode_car, mode_pt, mode_walk, mode_bike, mode_car_passenger
        FROM agg WHERE origin_cell_id IS NOT NULL AND dest_cell_id IS NOT NULL AND n_trips > 0
    """)
    n = db.execute(f"SELECT COUNT(*) FROM flow_grid_{resolution_m}m").fetchone()[0]
    log.info("flow_grid_%dm: %d pairs", resolution_m, n)
    return n


def build_out_of_home_grid(db: duckdb.DuckDBPyConnection, resolution_m: int) -> int:
    """Per-person hourly: which hour the person is *away* from home.

    A person is counted as "away" at hour h if their *active* activity at
    second h*3600 has purpose != 'home'. Active = start_time <= t < end_time
    (boundary times are NULL for first/last activities — treat as open-ended).
    """
    home_cell = _cell_id_expr("ST_X(p.home_pt)", "ST_Y(p.home_pt)", resolution_m)
    cell_geom = _cell_geom_expr("cell_id", resolution_m)

    away_buckets_raw = ",\n                ".join(
        f"""COUNT(DISTINCT person_id) FILTER (
            WHERE COALESCE(start_time, -1) <= {h * 3600}
              AND COALESCE(end_time, 1e12) >  {h * 3600}
              AND purpose != 'home'
        )::INTEGER AS away_h{h}""" for h in range(24)
    )

    db.execute(f"""
        INSERT INTO out_of_home_grid_{resolution_m}m
        WITH person_cell AS (
            SELECT p.person_id, {home_cell} AS cell_id
            FROM persons p WHERE {_bbox_clause('p.home_pt')}
        ),
        cell_pop AS (
            SELECT cell_id, COUNT(*)::INTEGER AS n_persons
            FROM person_cell GROUP BY cell_id
        ),
        joined AS (
            SELECT pc.cell_id, a.person_id, a.purpose, a.start_time, a.end_time
            FROM person_cell pc
            JOIN activities a USING (person_id)
        ),
        away_agg AS (
            SELECT cell_id,
                {away_buckets_raw}
            FROM joined
            GROUP BY cell_id
        )
        SELECT cp.cell_id, {cell_geom} AS cell_geom, cp.n_persons,
               COALESCE(aa.away_h0, 0), COALESCE(aa.away_h1, 0), COALESCE(aa.away_h2, 0),
               COALESCE(aa.away_h3, 0), COALESCE(aa.away_h4, 0), COALESCE(aa.away_h5, 0),
               COALESCE(aa.away_h6, 0), COALESCE(aa.away_h7, 0), COALESCE(aa.away_h8, 0),
               COALESCE(aa.away_h9, 0), COALESCE(aa.away_h10, 0), COALESCE(aa.away_h11, 0),
               COALESCE(aa.away_h12, 0), COALESCE(aa.away_h13, 0), COALESCE(aa.away_h14, 0),
               COALESCE(aa.away_h15, 0), COALESCE(aa.away_h16, 0), COALESCE(aa.away_h17, 0),
               COALESCE(aa.away_h18, 0), COALESCE(aa.away_h19, 0), COALESCE(aa.away_h20, 0),
               COALESCE(aa.away_h21, 0), COALESCE(aa.away_h22, 0), COALESCE(aa.away_h23, 0)
        FROM cell_pop cp
        LEFT JOIN away_agg aa USING (cell_id)
        WHERE cp.cell_id IS NOT NULL AND cp.n_persons > 0
    """)
    n = db.execute(f"SELECT COUNT(*) FROM out_of_home_grid_{resolution_m}m").fetchone()[0]
    log.info("out_of_home_grid_%dm: %d cells", resolution_m, n)
    return n
