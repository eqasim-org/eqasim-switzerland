"""Post-build validation per webmap-handoff briefing.

Two modes:
  - ``validate(db, source_type, full=False)``: schema-only checks. Used by the
    skeleton smoke test and after Phase-1 (DDL only) — does not require any
    rows.
  - ``validate(db, source_type, full=True)``: full check including row counts,
    bbox plausibility, and pre-aggregation consistency. Run after the final
    build phase.

Raises ``AssertionError`` with a specific message on the first failure. Caller
catches and decides whether to abort the stage.
"""

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
    """Schema-only checks — safe on an empty (DDL-only) database."""
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
    """Schema + row-level sanity checks. Run only after Phase 6.

    Mirrors the briefing's validate.py spec.
    """
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
            f"{bbox_violations} home_pt coords outside CH-bbox — projection bug?"
        )


    grid_sum = db.execute("SELECT SUM(n_persons) FROM demo_grid_5000m").fetchone()[0] or 0
    if abs(grid_sum - n_persons) / max(n_persons, 1) > 0.01:
        raise AssertionError(
            f"demo_grid_5000m persons sum {grid_sum} differs >1% from persons {n_persons}"
        )

    trip_grid_sum = db.execute(
        "SELECT SUM(n_trips) FROM trip_grid_origin_500m"
    ).fetchone()[0] or 0
    if abs(trip_grid_sum - n_trips) / max(n_trips, 1) > 0.01:
        raise AssertionError(
            f"trip_grid_origin_500m trips sum {trip_grid_sum} differs >1% from trips {n_trips}"
        )

    _validate_grid_consistency(db)


def _validate_grid_consistency(db) -> None:
    """Every 500m demo-grid cell must have at least one overlapping 100m
    subcell. A mismatch indicates the cell_id encoding is inconsistent with
    the cell_geom decoder (e.g. CAST-rounding vs FLOOR).
    """
    n_broken = db.execute("""
        SELECT COUNT(*) FROM demo_grid_500m g500
        WHERE NOT EXISTS (
            SELECT 1 FROM demo_grid_100m g100
            WHERE ST_Intersects(g100.cell_geom, g500.cell_geom)
        )
    """).fetchone()[0]
    if n_broken > 0:
        raise AssertionError(
            f"{n_broken} 500m cells have no overlapping 100m subcell "
            f"— grid encoding inconsistent"
        )


def validate(db, source_type: str, full: bool = False) -> None:
    if full:
        validate_full(db, source_type)
    else:
        validate_schema(db, source_type)
