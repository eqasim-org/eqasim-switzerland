"""eqasim post-process stage: webmap DuckDB export.

Produces up to two files in <matsim_dir>/simulation_output/webmap/:
    synthetic.duckdb
    microcensus.duckdb

Selected via the single config option ``webmap_export``:
    "both"        → build both files
    "synthetic"   → build only synthetic.duckdb
    "microcensus" → build only microcensus.duckdb
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import grids, hot_polygons, network, raw_entities, spider
from .schema import (
    GRID_RESOLUTIONS_M,
    SCHEMA_VERSION,
    create_all_indexes,
    create_all_tables,
)
from .sources import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_HOME_PIPE,
    discover_microcensus,
    discover_synthetic,
)
from .validation import CH_BBOX_LV95, validate_full, validate_schema

log = logging.getLogger(__name__)

_VALID_MODES = {"both", "synthetic", "microcensus"}


def configure(context):
    context.config("webmap_export", "both")
    context.stage("matsim.simulation.run")


def execute(context):
    mode = str(context.config("webmap_export")).strip().lower()
    if mode not in _VALID_MODES:
        log.warning(
            "webmap_export=%r is not one of %s — stage skipped",
            mode, sorted(_VALID_MODES),
        )
        return {"skipped": True, "webmap_export": mode}

    matsim_dir = Path(context.stage("matsim.simulation.run"))
    output_dir = matsim_dir / "simulation_output" / "webmap"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = DEFAULT_CACHE_DIR
    data_path = DEFAULT_DATA_PATH
    home_pipe = DEFAULT_HOME_PIPE

    syn_path = output_dir / "synthetic.duckdb"
    mc_path = output_dir / "microcensus.duckdb"

    build_synthetic = mode in ("both", "synthetic")
    build_microcensus = mode in ("both", "microcensus")

    if build_synthetic:
        _build_synthetic(
            output_db=syn_path,
            matsim_dir=matsim_dir,
            cache_dir=cache_dir,
            data_path=data_path,
            home_pipe=home_pipe,
        )

    if build_microcensus:
        _build_microcensus(
            output_db=mc_path,
            cache_dir=cache_dir,
            data_path=data_path,
        )

    (output_dir / "schema_version.txt").write_text(SCHEMA_VERSION + "\n")
    _write_metadata_json(output_dir, syn_path, mc_path,
                         wrote_synthetic=build_synthetic,
                         wrote_microcensus=build_microcensus)

    return {
        "synthetic": str(syn_path) if build_synthetic else None,
        "microcensus": str(mc_path) if build_microcensus else None,
        "schema_version": SCHEMA_VERSION,
        "webmap_export": mode,
    }


def _build_synthetic(
    *, output_db: Path, matsim_dir: Path, cache_dir: Path, data_path: Path,
    home_pipe: Path,
) -> None:
    import duckdb
    log.info("=== synthetic.duckdb build START")
    src = discover_synthetic(matsim_dir, cache_dir=cache_dir, data_path=data_path, home_pipe=home_pipe)
    _unlink_db(output_db)

    db = duckdb.connect(str(output_db))
    try:
        db.execute("INSTALL spatial; LOAD spatial;")
        create_all_tables(db, "synthetic")

        acts_df = None
        if src.output_activities_csv is not None and src.output_activities_csv.exists():
            acts_df = raw_entities.parse_activities_csv(src.output_activities_csv)
        n_acts = raw_entities.insert_activities_df(db, acts_df)
        n_persons = raw_entities.load_persons_synthetic(
            db, src.persons_parquet, src.statpop_persons_pickle,
            activities_for_home_pt=acts_df,
            activities_for_n_acts=acts_df,
        )
        n_hh = raw_entities.load_households_synthetic(
            db, src.enriched_pickle, src.households_pickle,
        )
        n_trips = raw_entities.load_trips_synthetic(db, src.output_trips_csv)
        db.execute("CHECKPOINT")

        for r in GRID_RESOLUTIONS_M:
            grids.build_demo_grid(db, r)
        if n_trips:
            grids.build_trip_grid_origin(db, 500)
            grids.build_flow_grid(db, 500)
        if n_acts:
            grids.build_out_of_home_grid(db, 500)
        db.execute("CHECKPOINT")

        loaded = hot_polygons.load_hot_polygons(
            db, src.swisstopo_canton_shp, src.swisstopo_bezirk_shp, src.swisstopo_gemeinde_shp,
        )
        if loaded:
            hot_polygons.build_hot_polygon_demo(db)
            if n_trips:
                hot_polygons.build_hot_polygon_trips(db)
                hot_polygons.build_hot_polygon_flows(db)
            if n_acts:
                hot_polygons.build_hot_polygon_out_of_home(db)
        db.execute("CHECKPOINT")

        if src.output_events_xml is not None and src.output_events_xml.exists():
            spider.build_spider(db, src.output_events_xml)
        else:
            log.info("spider: skipped (events_xml=%s)", src.output_events_xml)
        if src.output_network_xml is not None and src.output_network_xml.exists():
            network.build_network(db, src.output_network_xml)
            if src.link_speeds_parquet:
                network.load_link_speeds(db, src.link_speeds_parquet)
        else:
            log.info("network: skipped (network_xml=%s)", src.output_network_xml)
        db.execute("CHECKPOINT")

        create_all_indexes(db, "synthetic")

        _insert_metadata(
            db, source_type="synthetic", n_persons=n_persons, n_trips=n_trips, n_acts=n_acts,
            hot_polygon_types=loaded, has_pt_static=False,
        )
        try:
            validate_full(db, "synthetic")
        except AssertionError as e:
            log.error("Validation FAILED for synthetic.duckdb: %s", e)

        db.execute("CHECKPOINT")
    finally:
        db.close()
    log.info("=== synthetic.duckdb build DONE → %s (%.1f MB)",
             output_db, output_db.stat().st_size / 1e6 if output_db.exists() else 0)


def _build_microcensus(
    *, output_db: Path, cache_dir: Path, data_path: Path,
) -> None:
    import duckdb
    log.info("=== microcensus.duckdb build START")
    try:
        src = discover_microcensus(cache_dir=cache_dir, data_path=data_path)
    except FileNotFoundError as e:
        log.error("microcensus.duckdb: missing inputs — %s", e)
        return
    _unlink_db(output_db)

    db = duckdb.connect(str(output_db))
    try:
        db.execute("INSTALL spatial; LOAD spatial;")
        create_all_tables(db, "microcensus")

        n_persons = raw_entities.load_persons_microcensus(
            db, src.household_persons_pickle, src.households_pickle,
        )
        n_hh = raw_entities.load_households_microcensus(db, src.households_pickle)
        n_trips = raw_entities.load_trips_microcensus(db, src.trips_pickle)

        n_acts = raw_entities.derive_activities_microcensus(db, src.trips_pickle)
        db.execute("CHECKPOINT")

        for r in GRID_RESOLUTIONS_M:
            grids.build_demo_grid(db, r)
        if n_trips:
            grids.build_trip_grid_origin(db, 500)
        if n_acts:
            grids.build_out_of_home_grid(db, 500)
        db.execute("CHECKPOINT")

        loaded = hot_polygons.load_hot_polygons(
            db, src.swisstopo_canton_shp, src.swisstopo_bezirk_shp, src.swisstopo_gemeinde_shp,
        )
        if loaded:
            hot_polygons.build_hot_polygon_demo(db)
            if n_trips:
                hot_polygons.build_hot_polygon_trips(db)
            if n_acts:
                hot_polygons.build_hot_polygon_out_of_home(db)
        db.execute("CHECKPOINT")

        create_all_indexes(db, "microcensus")
        _insert_metadata(
            db, source_type="microcensus", n_persons=n_persons, n_trips=n_trips, n_acts=n_acts,
            hot_polygon_types=loaded, has_pt_static=False,
        )
        try:
            validate_full(db, "microcensus")
        except AssertionError as e:
            log.error("Validation FAILED for microcensus.duckdb: %s", e)
        db.execute("CHECKPOINT")
    finally:
        db.close()
    log.info("=== microcensus.duckdb build DONE → %s (%.1f MB)",
             output_db, output_db.stat().st_size / 1e6 if output_db.exists() else 0)


def _unlink_db(path: Path) -> None:
    if path.exists():
        path.unlink()
    wal = path.with_suffix(".duckdb.wal")
    if wal.exists():
        wal.unlink()


def _insert_metadata(
    db, *, source_type: str, n_persons: int, n_trips: int, n_acts: int,
    hot_polygon_types: list, has_pt_static: bool,
) -> None:
    db.execute(
        """
        INSERT INTO metadata (
            schema_version, build_date, source_type, matsim_run_id,
            eqasim_commit_hash, person_count, trip_count, activity_count,
            grid_resolutions_m, bbox_lv95, hot_polygon_types, has_pt_static
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            SCHEMA_VERSION,
            datetime.now(timezone.utc),
            source_type,
            None,
            None,
            int(n_persons), int(n_trips), int(n_acts),
            list(GRID_RESOLUTIONS_M),
            list(CH_BBOX_LV95),
            list(hot_polygon_types),
            bool(has_pt_static),
        ],
    )


def _write_metadata_json(
    output_dir: Path, syn_path: Path, mc_path: Path,
    *, wrote_synthetic: bool, wrote_microcensus: bool,
) -> None:
    outputs = {}
    if wrote_synthetic:
        outputs["synthetic"] = {
            "path": str(syn_path),
            "size_bytes": syn_path.stat().st_size if syn_path.exists() else 0,
        }
    if wrote_microcensus:
        outputs["microcensus"] = {
            "path": str(mc_path),
            "size_bytes": mc_path.stat().st_size if mc_path.exists() else 0,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "build_date": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    (output_dir / "metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
