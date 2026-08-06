"""Post-build validation: validate(db, source_type, full=False) checks schema only;
full=True adds row counts, bbox plausibility and pre-aggregation consistency."""

from __future__ import annotations

from .schema import (
    SCHEMA_VERSION,
    EXPECTED_TABLES_SYNTHETIC,
    EXPECTED_TABLES_MICROCENSUS,
    EXPECTED_COLUMNS,
)


CH_BBOX_LV95 = (2_400_000.0, 1_050_000.0, 2_900_000.0, 1_300_000.0)


def _expected_tables(source_type: str) -> set[str]:
    if source_type == "synthetic":
        return set(EXPECTED_TABLES_SYNTHETIC)
    if source_type == "microcensus":
        return set(EXPECTED_TABLES_MICROCENSUS)
    raise ValueError(f"Unknown source_type: {source_type!r}")


def validate_schema(db, source_type: str) -> None:
    """Schema-only checks - safe on an empty (DDL-only) database."""
    actual = {r[0] for r in db.execute("SHOW TABLES").fetchall()}
    expected = _expected_tables(source_type)
    missing = expected - actual
    if missing:
        raise AssertionError(f"Tables missing for {source_type}: {sorted(missing)}")

    for table, required_cols in EXPECTED_COLUMNS.items():
        if table not in actual:
            if table in expected:
                raise AssertionError(f"Required table {table} missing")
            continue
        actual_cols = {r[0] for r in db.execute(f"DESCRIBE {table}").fetchall()}
        col_missing = required_cols - actual_cols
        if col_missing:
            raise AssertionError(
                f"{table}: columns missing: {sorted(col_missing)}"
            )


def validate_full(db, source_type: str) -> None:
    """Schema + row-level sanity checks; run only after Phase 6."""
    validate_schema(db, source_type)

    sv = db.execute("SELECT schema_version FROM metadata").fetchone()
    if sv is None or sv[0] != SCHEMA_VERSION:
        raise AssertionError(
            f"metadata.schema_version != {SCHEMA_VERSION}: got {sv!r}"
        )

    n_persons = db.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    n_trips = db.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    n_acts = db.execute("SELECT COUNT(*) FROM activities").fetchone()[0]
    if not (n_persons > 0 and n_trips > 0 and n_acts > 0):
        raise AssertionError(
            f"Empty raw entities: persons={n_persons}, trips={n_trips}, activities={n_acts}"
        )

    xmin, ymin, xmax, ymax = CH_BBOX_LV95
    bbox_violations = db.execute(f"""
        SELECT COUNT(*) FROM persons
        WHERE home_pt IS NOT NULL
          AND (ST_X(home_pt) < {xmin} OR ST_X(home_pt) > {xmax}
               OR ST_Y(home_pt) < {ymin} OR ST_Y(home_pt) > {ymax})
    """).fetchone()[0]
    if bbox_violations:
        raise AssertionError(
            f"{bbox_violations} home_pt coords outside CH-bbox - projection bug?"
        )

    _validate_v2_hex_consistency(db, n_persons=n_persons, n_trips=n_trips)
    if source_type == "synthetic":
        _validate_synthetic_enrichment(db)
        _validate_v4_pre_aggregates(db)
        _validate_pt_link_volumes(db)
    if source_type == "microcensus":
        _validate_microcensus_enrichment(db)


def _validate_v2_hex_consistency(db, *, n_persons: int, n_trips: int) -> None:
    """H3 parent/child hierarchy + cross-resolution sum consistency.

    Invariants: every hex has a parent at each coarser resolution; n_persons sums
    match across resolutions and equal persons-with-home; trip hex sum matches trips."""
    from .schema import H3_RES_COARSE, H3_RES_FINE, H3_RES_MID

    n_orphan12 = db.execute(f"""
        SELECT COUNT(*) FROM demo_hex_res{H3_RES_FINE}
        WHERE h3_parent_res9 NOT IN (SELECT h3_index FROM demo_hex_res{H3_RES_MID})
    """).fetchone()[0]
    if n_orphan12 > 0:
        raise AssertionError(
            f"{n_orphan12} res-{H3_RES_FINE} hex without res-{H3_RES_MID} parent"
        )

    n_orphan9 = db.execute(f"""
        SELECT COUNT(*) FROM demo_hex_res{H3_RES_MID}
        WHERE h3_parent_res6 NOT IN (SELECT h3_index FROM demo_hex_res{H3_RES_COARSE})
    """).fetchone()[0]
    if n_orphan9 > 0:
        raise AssertionError(
            f"{n_orphan9} res-{H3_RES_MID} hex without res-{H3_RES_COARSE} parent"
        )

    s12 = db.execute(f"SELECT SUM(n_persons) FROM demo_hex_res{H3_RES_FINE}").fetchone()[0] or 0
    s9 = db.execute(f"SELECT SUM(n_persons) FROM demo_hex_res{H3_RES_MID}").fetchone()[0] or 0
    s6 = db.execute(f"SELECT SUM(n_persons) FROM demo_hex_res{H3_RES_COARSE}").fetchone()[0] or 0
    if not (s12 == s9 == s6):
        raise AssertionError(
            f"demo_hex resolution sums mismatch: r{H3_RES_FINE}={s12}, "
            f"r{H3_RES_MID}={s9}, r{H3_RES_COARSE}={s6}"
        )

    n_persons_with_home = db.execute(
        "SELECT COUNT(*) FROM persons WHERE home_pt IS NOT NULL"
    ).fetchone()[0]
    if s12 != n_persons_with_home:
        raise AssertionError(
            f"demo_hex_res{H3_RES_FINE} sum ({s12}) ≠ persons-with-home ({n_persons_with_home})"
        )

    trip_hex_sum = db.execute(
        f"SELECT SUM(n_trips) FROM trip_hex_origin_res{H3_RES_MID}"
    ).fetchone()[0] or 0
    if abs(trip_hex_sum - n_trips) / max(n_trips, 1) > 0.01:
        raise AssertionError(
            f"trip_hex_origin_res{H3_RES_MID} sum ({trip_hex_sum}) differs >1% from trips ({n_trips})"
        )


def _validate_v4_pre_aggregates(db) -> None:
    """Iteration-4 pre-aggregate sanity checks (synthetic only).

    Skips gracefully if spider_link_index is empty (partial build)."""
    spider_n = db.execute("SELECT COUNT(*) FROM spider_link_index").fetchone()[0]
    if spider_n == 0:
        return

    slv_sum = db.execute(
        "SELECT COALESCE(SUM(n_traversals), 0) FROM spider_link_volumes_by_hex_res6"
    ).fetchone()[0]
    spider_with_home = db.execute("""
        SELECT COUNT(*) FROM spider_link_index sli
        JOIN persons p ON p.person_id = sli.person_id
        WHERE p.home_h3_res6 IS NOT NULL
    """).fetchone()[0]
    if slv_sum != spider_with_home:
        raise AssertionError(
            f"spider_link_volumes_by_hex_res6 sum ({slv_sum}) ≠ spider rows joinable "
            f"to a home_h3_res6 ({spider_with_home})"
        )

    zflv_sum = db.execute(
        "SELECT COALESCE(SUM(n_trips), 0) FROM zone_flow_link_volumes_hex_res6"
    ).fetchone()[0]
    spider_joined = db.execute("""
        SELECT COUNT(*) FROM spider_link_index sli
        JOIN trips t ON t.person_id = sli.person_id AND t.trip_index = sli.trip_index
        WHERE t.origin_h3_res6 IS NOT NULL AND t.dest_h3_res6 IS NOT NULL
    """).fetchone()[0]
    if zflv_sum != spider_joined:
        raise AssertionError(
            f"zone_flow_link_volumes_hex_res6 sum ({zflv_sum}) ≠ spider×trip "
            f"hex-keyed rows ({spider_joined})"
        )

    nfm_n = db.execute("SELECT COUNT(*) FROM node_flow_matrix").fetchone()[0]
    if nfm_n == 0:
        raise AssertionError(
            "node_flow_matrix is empty but spider_link_index has rows"
        )


def _validate_pt_link_volumes(db) -> None:
    """Range sanity for pt_link_volumes; empty table is allowed (no events/schedule)."""
    n = db.execute("SELECT COUNT(*) FROM pt_link_volumes").fetchone()[0]
    if n == 0:
        return
    bad_bins = db.execute(
        "SELECT COUNT(*) FROM pt_link_volumes WHERE time_bin < 0 OR time_bin > 95"
    ).fetchone()[0]
    if bad_bins:
        raise AssertionError(f"pt_link_volumes: {bad_bins} rows with time_bin outside 0..95")
    bad_vol = db.execute(
        "SELECT COUNT(*) FROM pt_link_volumes WHERE volume < 0"
    ).fetchone()[0]
    if bad_vol:
        raise AssertionError(f"pt_link_volumes: {bad_vol} rows with negative volume")
    no_route = db.execute(
        "SELECT COUNT(*) FROM pt_link_volumes WHERE route_id = ''"
    ).fetchone()[0]
    if no_route == n:
        raise AssertionError(
            "pt_link_volumes: every row has an empty route_id - "
            "TransitDriverStarts.transitRouteId not picked up?"
        )


def _validate_synthetic_enrichment(db) -> None:
    """The webmap's PT-subscription and car-availability panels need these columns
    populated. A stale/partial persons parquet loads without error and leaves them
    all-NULL (raw_entities.load_persons_synthetic degrades silently), so assert
    coverage here. The threshold is below 100% because persons unmatched to the
    microcensus (age 0-5) legitimately carry NULL subscriptions."""
    n_persons, n_car, n_subs = db.execute("""
        SELECT COUNT(*),
               COUNT(car_availability),
               COUNT(subscriptions_ga)
        FROM persons
    """).fetchone()
    if n_persons == 0:
        raise AssertionError("synthetic persons table is empty")

    for name, n in (("car_availability", n_car), ("subscriptions_ga", n_subs)):
        if n < 0.9 * n_persons:
            raise AssertionError(
                f"synthetic persons.{name} non-null for only {n}/{n_persons} "
                f"({100.0 * n / n_persons:.1f}%) - expected >=90%; the persons parquet "
                "is probably a stale/partial file missing this column "
                "(check the 'persons parquet ->' line in the build log)"
            )

    # persons and trips/activities must come from the same downsampling: a leftover
    # parquet at a different sample rate leaves most persons with no plan at all.
    n_with_acts = db.execute(
        "SELECT COUNT(DISTINCT person_id) FROM activities"
    ).fetchone()[0]
    n_orphan_acts = db.execute("""
        SELECT COUNT(DISTINCT a.person_id) FROM activities a
        WHERE NOT EXISTS (SELECT 1 FROM persons p WHERE p.person_id = a.person_id)
    """).fetchone()[0]
    if n_orphan_acts:
        raise AssertionError(
            f"{n_orphan_acts} person_ids appear in activities but not in persons - "
            "persons parquet does not match the MATSim run"
        )
    if n_with_acts < 0.5 * n_persons:
        raise AssertionError(
            f"only {n_with_acts}/{n_persons} synthetic persons "
            f"({100.0 * n_with_acts / n_persons:.1f}%) have any activity - expected >=50%; "
            "persons parquet is probably at a different sample rate than the MATSim run"
        )


def _validate_microcensus_enrichment(db) -> None:
    """The webmap's microcensus panels need respondent survey attributes and
    per-day activity counts; all-NULL columns mean the respondents pickle was
    not found/joined (raw_entities.load_persons_microcensus degrades silently)."""
    n_car = db.execute(
        "SELECT COUNT(*) FROM persons WHERE car_availability IS NOT NULL"
    ).fetchone()[0]
    n_subs = db.execute(
        "SELECT COUNT(*) FROM persons WHERE subscriptions_ga IS NOT NULL"
    ).fetchone()[0]
    if n_car == 0 or n_subs == 0:
        raise AssertionError(
            f"microcensus persons lack survey attributes (car_availability non-null: "
            f"{n_car}, subscriptions_ga non-null: {n_subs}) - respondents pickle "
            "missing or person-id join broken"
        )
    n_acts = db.execute(
        "SELECT COUNT(*) FROM persons WHERE n_activities IS NOT NULL"
    ).fetchone()[0]
    if n_acts == 0:
        raise AssertionError(
            "microcensus persons have no n_activities - activities derivation "
            "or backfill_n_activities failed"
        )
    n_income = db.execute("""
        SELECT COUNT(*) FROM persons p
        JOIN households h ON h.household_id = p.household_id
        WHERE h.income_class IS NOT NULL
    """).fetchone()[0]
    if n_income == 0:
        raise AssertionError(
            "no microcensus person joins a household with income_class - "
            "household key convention broken"
        )
    n_act_rows, n_act_canton = db.execute(
        "SELECT COUNT(*), COUNT(canton_id) FROM activities"
    ).fetchone()
    if n_act_rows > 0 and n_act_canton == 0:
        raise AssertionError(
            f"activities.canton_id is NULL for all {n_act_rows} activities - "
            "canton assignment did not run (or failed non-fatally)"
        )
    cars_2, cars_3_plus = db.execute(
        "SELECT COALESCE(SUM(cars_2), 0), COALESCE(SUM(cars_3_plus), 0) "
        "FROM hot_polygon_demo"
    ).fetchone()
    if cars_2 > 0 and cars_3_plus == 0:
        raise AssertionError(
            f"hot_polygon_demo: cars_3_plus sums to 0 while cars_2 sums to {cars_2} - "
            "n_cars_class '3'/'3+' label mismatch in the aggregation filters"
        )


def validate(db, source_type: str, full: bool = False) -> None:
    if full:
        validate_full(db, source_type)
    else:
        validate_schema(db, source_type)
