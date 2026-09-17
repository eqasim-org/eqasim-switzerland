"""H3 hexagon aggregates: demo/trip/flow/out-of-home tables at res 12/9/6.
H3 indices are set by raw_entities at insert time; cell geometry is LV95 via .hex.
"""

from __future__ import annotations

import logging
from typing import Iterable

import duckdb
import pandas as pd

from .hex import (
    H3_RES_COARSE,
    H3_RES_FINE,
    H3_RES_MID,
    hex_geoms_wkt_lv95_bulk,
)

log = logging.getLogger(__name__)


_DEMO_COLS = [
    "n_persons",
    "age_0_6", "age_6_15", "age_15_18", "age_18_24", "age_24_30",
    "age_30_45", "age_45_65", "age_65_80", "age_80_plus",
    "sex_male", "sex_female",
    "car_avail_always", "car_avail_sometimes", "car_avail_never",
    "has_driving_license", "employed",
    "subs_ga", "subs_halbtax", "subs_verbund", "subs_strecke",
    "subs_gleis7", "subs_junior", "subs_other",
    "cars_0", "cars_1", "cars_2", "cars_3_plus",
    "sum_activities",
]

_TRIP_COLS = [
    "n_trips",
    "mode_car", "mode_pt", "mode_walk", "mode_bike", "mode_car_passenger",
    "purpose_home", "purpose_work", "purpose_education",
    "purpose_shop", "purpose_leisure", "purpose_other",
    "sum_network_distance", "sum_crowfly_distance",
    "dist_bucket_0_1km", "dist_bucket_1_3km", "dist_bucket_3_10km",
    "dist_bucket_10_30km", "dist_bucket_30_plus_km",
    *[f"time_h{h}" for h in range(24)],
]

_OH_COLS = ["n_persons", *[f"away_h{h}" for h in range(24)]]


def _ddl_sums(cols: Iterable[str]) -> str:
    return ", ".join(f"SUM({c})::INTEGER AS {c}" if c.startswith(("n_", "age_", "sex_",
                     "car_avail", "has_", "employed", "subs_", "cars_", "sum_activities",
                     "purpose_", "mode_", "dist_bucket_", "time_h", "away_h"))
                     else f"SUM({c}) AS {c}" for c in cols)


def _attach_geoms(db: duckdb.DuckDBPyConnection, df: pd.DataFrame,
                   include_center: bool) -> pd.DataFrame:
    """Add cell_geom_wkt (and optionally cell_center_wkt) columns from h3_index."""
    if df.empty:
        df = df.copy()
        df["cell_geom_wkt"] = pd.Series(dtype=str)
        if include_center:
            df["cell_center_wkt"] = pd.Series(dtype=str)
        return df
    h3_ids = df["h3_index"].astype("int64").tolist()
    geoms, centers = hex_geoms_wkt_lv95_bulk(h3_ids)
    df = df.copy()
    df["cell_geom_wkt"] = geoms
    if include_center:
        df["cell_center_wkt"] = centers
    return df


def build_demo_hex_pyramid(db: duckdb.DuckDBPyConnection) -> None:
    """Build demo_hex_res12 from persons, then rollup to res9 and res6."""

    demo_agg_sql = ",\n            ".join([
        "COUNT(*)::INTEGER AS n_persons",
        "COUNT(*) FILTER (WHERE p.age IS NOT NULL AND p.age <  6)::INTEGER AS age_0_6",
        "COUNT(*) FILTER (WHERE p.age >=  6 AND p.age < 15)::INTEGER AS age_6_15",
        "COUNT(*) FILTER (WHERE p.age >= 15 AND p.age < 18)::INTEGER AS age_15_18",
        "COUNT(*) FILTER (WHERE p.age >= 18 AND p.age < 24)::INTEGER AS age_18_24",
        "COUNT(*) FILTER (WHERE p.age >= 24 AND p.age < 30)::INTEGER AS age_24_30",
        "COUNT(*) FILTER (WHERE p.age >= 30 AND p.age < 45)::INTEGER AS age_30_45",
        "COUNT(*) FILTER (WHERE p.age >= 45 AND p.age < 65)::INTEGER AS age_45_65",
        "COUNT(*) FILTER (WHERE p.age >= 65 AND p.age < 80)::INTEGER AS age_65_80",
        "COUNT(*) FILTER (WHERE p.age >= 80)::INTEGER AS age_80_plus",
        "COUNT(*) FILTER (WHERE p.sex = 0)::INTEGER AS sex_male",
        "COUNT(*) FILTER (WHERE p.sex = 1)::INTEGER AS sex_female",
        "COUNT(*) FILTER (WHERE p.car_availability = 'always')::INTEGER AS car_avail_always",
        "COUNT(*) FILTER (WHERE p.car_availability = 'sometimes')::INTEGER AS car_avail_sometimes",
        "COUNT(*) FILTER (WHERE p.car_availability = 'never')::INTEGER AS car_avail_never",
        "COUNT(*) FILTER (WHERE p.has_driving_license)::INTEGER AS has_driving_license",
        "COUNT(*) FILTER (WHERE p.employed)::INTEGER AS employed",
        "COUNT(*) FILTER (WHERE p.subscriptions_ga)::INTEGER AS subs_ga",
        "COUNT(*) FILTER (WHERE p.subscriptions_halbtax)::INTEGER AS subs_halbtax",
        "COUNT(*) FILTER (WHERE p.subscriptions_verbund)::INTEGER AS subs_verbund",
        "COUNT(*) FILTER (WHERE p.subscriptions_strecke)::INTEGER AS subs_strecke",
        "COUNT(*) FILTER (WHERE p.subscriptions_gleis7)::INTEGER AS subs_gleis7",
        "COUNT(*) FILTER (WHERE p.subscriptions_junior)::INTEGER AS subs_junior",
        "COUNT(*) FILTER (WHERE p.subscriptions_other)::INTEGER AS subs_other",
        "COUNT(*) FILTER (WHERE h.n_cars_class = '0')::INTEGER AS cars_0",
        "COUNT(*) FILTER (WHERE h.n_cars_class = '1')::INTEGER AS cars_1",
        "COUNT(*) FILTER (WHERE h.n_cars_class = '2')::INTEGER AS cars_2",
        "COUNT(*) FILTER (WHERE h.n_cars_class IN ('3', '3+'))::INTEGER AS cars_3_plus",
        "COALESCE(SUM(p.n_activities), 0)::INTEGER AS sum_activities",
    ])

    df12 = db.execute(f"""
        SELECT p.home_h3_res{H3_RES_FINE} AS h3_index,
               ANY_VALUE(p.home_h3_res{H3_RES_MID}) AS h3_parent_res9,
               ANY_VALUE(p.home_h3_res{H3_RES_COARSE}) AS h3_parent_res6,
               {demo_agg_sql}
        FROM persons p
        LEFT JOIN households h ON p.household_id = h.household_id
        WHERE p.home_h3_res{H3_RES_FINE} IS NOT NULL
        GROUP BY p.home_h3_res{H3_RES_FINE}
    """).fetchdf()
    df12 = _attach_geoms(db, df12, include_center=True)

    db.register("_tmp_demo12", df12)
    db.execute(f"""
        INSERT INTO demo_hex_res{H3_RES_FINE}
            (h3_index, h3_parent_res9, h3_parent_res6,
             cell_geom, cell_center,
             {', '.join(_DEMO_COLS)})
        SELECT h3_index, h3_parent_res9, h3_parent_res6,
               ST_GeomFromText(cell_geom_wkt), ST_GeomFromText(cell_center_wkt),
               {', '.join(_DEMO_COLS)}
        FROM _tmp_demo12
    """)
    db.unregister("_tmp_demo12")
    log.info("demo_hex_res%d: %d cells", H3_RES_FINE, len(df12))

    sum_cols = ", ".join(f"SUM({c})::INTEGER AS {c}" for c in _DEMO_COLS)
    df9 = db.execute(f"""
        SELECT h3_parent_res9 AS h3_index,
               ANY_VALUE(h3_parent_res6) AS h3_parent_res6,
               {sum_cols}
        FROM demo_hex_res{H3_RES_FINE}
        GROUP BY h3_parent_res9
    """).fetchdf()
    df9 = _attach_geoms(db, df9, include_center=True)
    db.register("_tmp_demo9", df9)
    db.execute(f"""
        INSERT INTO demo_hex_res{H3_RES_MID}
            (h3_index, h3_parent_res6, cell_geom, cell_center,
             {', '.join(_DEMO_COLS)})
        SELECT h3_index, h3_parent_res6,
               ST_GeomFromText(cell_geom_wkt), ST_GeomFromText(cell_center_wkt),
               {', '.join(_DEMO_COLS)}
        FROM _tmp_demo9
    """)
    db.unregister("_tmp_demo9")
    log.info("demo_hex_res%d: %d cells", H3_RES_MID, len(df9))

    df6 = db.execute(f"""
        SELECT h3_parent_res6 AS h3_index, {sum_cols}
        FROM demo_hex_res{H3_RES_MID}
        GROUP BY h3_parent_res6
    """).fetchdf()
    df6 = _attach_geoms(db, df6, include_center=True)
    db.register("_tmp_demo6", df6)
    db.execute(f"""
        INSERT INTO demo_hex_res{H3_RES_COARSE}
            (h3_index, cell_geom, cell_center,
             {', '.join(_DEMO_COLS)})
        SELECT h3_index,
               ST_GeomFromText(cell_geom_wkt), ST_GeomFromText(cell_center_wkt),
               {', '.join(_DEMO_COLS)}
        FROM _tmp_demo6
    """)
    db.unregister("_tmp_demo6")
    log.info("demo_hex_res%d: %d cells", H3_RES_COARSE, len(df6))


def build_trip_hex_origin(db: duckdb.DuckDBPyConnection) -> None:
    """Aggregate trips by origin_h3_res9."""
    time_buckets = ",\n            ".join(
        f"COUNT(*) FILTER (WHERE FLOOR(departure_time/3600) = {h})::INTEGER AS time_h{h}"
        for h in range(24)
    )
    df = db.execute(f"""
        SELECT origin_h3_res{H3_RES_MID} AS h3_index,
               COUNT(*)::INTEGER AS n_trips,
               ANY_VALUE(NULL) AS h3_parent_res6,
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
        FROM trips
        WHERE origin_h3_res{H3_RES_MID} IS NOT NULL
        GROUP BY origin_h3_res{H3_RES_MID}
    """).fetchdf()
    if df.empty:
        log.info("trip_hex_origin_res%d: 0 cells", H3_RES_MID)
        return

    from . import hex as hexmod
    df["h3_parent_res6"] = hexmod.hex_parent_int(df["h3_index"].astype("int64").to_numpy(),
                                                  H3_RES_COARSE)
    df = _attach_geoms(db, df, include_center=False)

    db.register("_tmp_trip_hex", df)
    db.execute(f"""
        INSERT INTO trip_hex_origin_res{H3_RES_MID}
            (h3_index, h3_parent_res6, cell_geom,
             {', '.join(_TRIP_COLS)})
        SELECT h3_index, h3_parent_res6,
               ST_GeomFromText(cell_geom_wkt),
               {', '.join(_TRIP_COLS)}
        FROM _tmp_trip_hex
    """)
    db.unregister("_tmp_trip_hex")
    log.info("trip_hex_origin_res%d: %d cells", H3_RES_MID, len(df))


def build_flow_hex(db: duckdb.DuckDBPyConnection) -> None:
    """OD-pairs at res9. Synthetic only."""
    df = db.execute(f"""
        SELECT origin_h3_res{H3_RES_MID} AS origin_h3_index,
               dest_h3_res{H3_RES_MID}   AS dest_h3_index,
               COUNT(*)::INTEGER AS n_trips,
               COUNT(*) FILTER (WHERE main_mode = 'car')::INTEGER AS mode_car,
               COUNT(*) FILTER (WHERE main_mode = 'pt')::INTEGER AS mode_pt,
               COUNT(*) FILTER (WHERE main_mode = 'walk')::INTEGER AS mode_walk,
               COUNT(*) FILTER (WHERE main_mode = 'bike')::INTEGER AS mode_bike,
               COUNT(*) FILTER (WHERE main_mode = 'car_passenger')::INTEGER AS mode_car_passenger
        FROM trips
        WHERE origin_h3_res{H3_RES_MID} IS NOT NULL
          AND dest_h3_res{H3_RES_MID}   IS NOT NULL
        GROUP BY origin_h3_res{H3_RES_MID}, dest_h3_res{H3_RES_MID}
    """).fetchdf()
    if df.empty:
        log.info("flow_hex_res%d: 0 pairs", H3_RES_MID)
        return

    o_ids = df["origin_h3_index"].astype("int64").tolist()
    d_ids = df["dest_h3_index"].astype("int64").tolist()
    o_geoms, _ = hex_geoms_wkt_lv95_bulk(o_ids)
    d_geoms, _ = hex_geoms_wkt_lv95_bulk(d_ids)
    df["o_wkt"] = o_geoms
    df["d_wkt"] = d_geoms

    db.register("_tmp_flow", df)
    db.execute(f"""
        INSERT INTO flow_hex_res{H3_RES_MID}
            (origin_h3_index, dest_h3_index,
             origin_cell_geom, dest_cell_geom,
             n_trips, mode_car, mode_pt, mode_walk, mode_bike, mode_car_passenger)
        SELECT origin_h3_index, dest_h3_index,
               ST_GeomFromText(o_wkt), ST_GeomFromText(d_wkt),
               n_trips, mode_car, mode_pt, mode_walk, mode_bike, mode_car_passenger
        FROM _tmp_flow
    """)
    db.unregister("_tmp_flow")
    log.info("flow_hex_res%d: %d pairs", H3_RES_MID, len(df))


def build_oh_hex(db: duckdb.DuckDBPyConnection) -> None:
    """Out-of-home per home-hex at res9."""
    away_per_hour = ",\n            ".join(
        f"""COUNT(DISTINCT p.person_id) FILTER (
            WHERE COALESCE(a.start_time, -1) <= {h * 3600}
              AND COALESCE(a.end_time, 1e12) >  {h * 3600}
              AND a.purpose != 'home'
        )::INTEGER AS away_h{h}""" for h in range(24)
    )

    away_sql_sums = ",\n            ".join(f"SUM(away_h{h})::INTEGER AS away_h{h}" for h in range(24))

    df = db.execute(f"""
        WITH person_cell AS (
            SELECT person_id, home_h3_res{H3_RES_MID} AS h3_index
            FROM persons
            WHERE home_h3_res{H3_RES_MID} IS NOT NULL
        ),
        cell_pop AS (
            SELECT h3_index, COUNT(*)::INTEGER AS n_persons
            FROM person_cell GROUP BY h3_index
        ),
        away_per_person AS (
            SELECT p.person_id, p.h3_index, {away_per_hour}
            FROM person_cell p
            LEFT JOIN activities a ON a.person_id = p.person_id
            GROUP BY p.person_id, p.h3_index
        ),
        away_per_cell AS (
            SELECT h3_index, {away_sql_sums}
            FROM away_per_person GROUP BY h3_index
        )
        SELECT c.h3_index, c.n_persons,
               {', '.join(f'a.away_h{h}' for h in range(24))}
        FROM cell_pop c JOIN away_per_cell a USING (h3_index)
    """).fetchdf()
    if df.empty:
        log.info("oh_hex_res%d: 0 cells", H3_RES_MID)
        return

    from . import hex as hexmod
    df["h3_parent_res6"] = hexmod.hex_parent_int(df["h3_index"].astype("int64").to_numpy(),
                                                  H3_RES_COARSE)
    df = _attach_geoms(db, df, include_center=False)

    db.register("_tmp_oh", df)
    db.execute(f"""
        INSERT INTO oh_hex_res{H3_RES_MID}
            (h3_index, h3_parent_res6, cell_geom,
             {', '.join(_OH_COLS)})
        SELECT h3_index, h3_parent_res6, ST_GeomFromText(cell_geom_wkt),
               {', '.join(_OH_COLS)}
        FROM _tmp_oh
    """)
    db.unregister("_tmp_oh")
    log.info("oh_hex_res%d: %d cells", H3_RES_MID, len(df))
