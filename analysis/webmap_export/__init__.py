"""eqasim post-process stage: webmap DuckDB export.

Writes to <matsim_dir>/simulation_output/webmap/, selected via the
``webmap_export`` config option:
    "both"        -> synthetic.duckdb and microcensus.duckdb
    "synthetic"   -> only synthetic.duckdb
    "microcensus" -> only microcensus.duckdb
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import (
    canton, events_extras, grids, hot_polygons, network, pre_aggregates,
    raw_entities, spider, static_assets, transit,
)
from .schema import (
    H3_RESOLUTIONS,
    SCHEMA_VERSION,
    create_all_indexes,
    create_all_tables,
)
from .sources import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_HOME_PIPE,
    discover_microcensus,
    discover_sample_rate,
    discover_synthetic,
)
from .validation import CH_BBOX_LV95, validate_full, validate_schema

log = logging.getLogger(__name__)

_VALID_MODES = {"both", "synthetic", "microcensus"}


def configure(context):
    context.config("webmap_export", "both")
    context.config("scale_pt_to_full_population", True)
    context.stage("matsim.simulation.run")


def execute(context):
    mode = str(context.config("webmap_export")).strip().lower()
    if mode not in _VALID_MODES:
        log.warning(
            "webmap_export=%r is not one of %s - stage skipped",
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

    sample_rate = discover_sample_rate(matsim_dir, home_pipe=home_pipe)
    scale_pt = bool(context.config("scale_pt_to_full_population"))
    run_name = matsim_dir.name.replace(".cache", "")
    log.info("webmap_export: sample_rate=%s scale_pt=%s run_name=%s",
             sample_rate, scale_pt, run_name)

    if build_synthetic:
        _build_synthetic(
            output_db=syn_path,
            matsim_dir=matsim_dir,
            cache_dir=cache_dir,
            data_path=data_path,
            home_pipe=home_pipe,
            sample_rate=sample_rate,
            run_name=run_name,
            scale_pt=scale_pt,
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
    home_pipe: Path, sample_rate: float | None = None, run_name: str = "",
    scale_pt: bool = True,
) -> None:
    import duckdb
    log.info("=== synthetic.duckdb build START (sample_rate=%s, scale_pt=%s, run=%s)",
             sample_rate, scale_pt, run_name)
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

        grids.build_demo_hex_pyramid(db)
        if n_trips:
            grids.build_trip_hex_origin(db)
            grids.build_flow_hex(db)
        if n_acts:
            grids.build_oh_hex(db)
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

        spider_loaded = False
        network_loaded = False
        if src.output_events_xml is not None and src.output_events_xml.exists():
            spider.build_spider(db, src.output_events_xml)
            spider_loaded = True
        else:
            log.info("spider: skipped (events_xml=%s)", src.output_events_xml)
        if src.output_network_xml is not None and src.output_network_xml.exists():
            network.build_network(db, src.output_network_xml)
            network_loaded = True
        else:
            log.info("network: skipped (network_xml=%s)", src.output_network_xml)
        db.execute("CHECKPOINT")

        try:
            canton.assign_canton_ids(
                db, source_type="synthetic", overwrite_persons=True,
                has_trips=bool(n_trips), has_network=network_loaded,
                has_activities=bool(n_acts),
            )
            # canton->canton OD requires trips.origin/dest_canton_id to be filled
            if loaded and n_trips:
                hot_polygons.build_hot_polygon_flows_canton(db)
        except Exception as e:
            log.error("canton assignment failed (non-fatal): %s", e)
        db.execute("CHECKPOINT")

        if spider_loaded and network_loaded:
            pre_aggregates.build_all(db)
            db.execute("CHECKPOINT")
        else:
            log.info("pre-aggregates: skipped (need both spider + network)")

        board_acc, transfer_data, pt_vol_acc = {}, {}, {}
        if src.output_events_xml is not None and src.output_events_xml.exists():
            try:
                handler = events_extras.extract(db, src.output_events_xml)
                board_acc = handler.board_acc
                transfer_data = handler.transfer_data
                pt_vol_acc = handler.pt_vol_acc
            except Exception as e:  # link_speeds/boardings are optional
                log.error("events_extras failed (non-fatal): %s", e)
            db.execute("CHECKPOINT")

        try:
            static_assets.build_all(db)
            static_assets.build_metadata_asset(
                db, sample_rate=sample_rate, run_name=run_name,
                scaled_to_full_population=bool(sample_rate and scale_pt),
            )
        except Exception as e:  # never let optional assets abort a long build
            log.error("static_assets build failed (non-fatal): %s", e)

        if src.output_transit_schedule_xml is not None and src.output_transit_schedule_xml.exists():
            try:
                transit.build_all(db, src.output_transit_schedule_xml,
                                  board_acc, transfer_data, sample_rate, scale_pt,
                                  pt_vol_acc=pt_vol_acc)
            except Exception as e:
                log.error("transit assets failed (non-fatal): %s", e)
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
    log.info("=== synthetic.duckdb build DONE -> %s (%.1f MB)",
             output_db, output_db.stat().st_size / 1e6 if output_db.exists() else 0)


def _build_microcensus(
    *, output_db: Path, cache_dir: Path, data_path: Path,
) -> None:
    import duckdb
    log.info("=== microcensus.duckdb build START")
    try:
        src = discover_microcensus(cache_dir=cache_dir, data_path=data_path)
    except FileNotFoundError as e:
        log.error("microcensus.duckdb: missing inputs - %s", e)
        return
    log.info("microcensus sources: household_persons=%s households=%s trips=%s "
             "respondents=%s",
             src.household_persons_pickle, src.households_pickle,
             src.trips_pickle, src.respondents_pickle)
    _unlink_db(output_db)

    db = duckdb.connect(str(output_db))
    try:
        db.execute("INSTALL spatial; LOAD spatial;")
        create_all_tables(db, "microcensus")

        n_persons = raw_entities.load_persons_microcensus(
            db, src.household_persons_pickle, src.households_pickle,
            src.respondents_pickle,
        )
        n_hh = raw_entities.load_households_microcensus(db, src.households_pickle)
        n_trips = raw_entities.load_trips_microcensus(db, src.trips_pickle)

        n_acts = raw_entities.derive_activities_microcensus(db, src.trips_pickle)
        raw_entities.backfill_n_activities(db)
        if n_trips:
            raw_entities.backfill_preceding_purpose(db)
        db.execute("CHECKPOINT")

        grids.build_demo_hex_pyramid(db)
        if n_trips:
            grids.build_trip_hex_origin(db)
        if n_acts:
            grids.build_oh_hex(db)
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

        try:
            canton.assign_canton_ids(
                db, source_type="microcensus", overwrite_persons=False,
                has_trips=bool(n_trips), has_network=False,
                has_activities=bool(n_acts),
            )
        except Exception as e:
            log.error("canton assignment failed (non-fatal): %s", e)
        db.execute("CHECKPOINT")

        create_all_indexes(db, "microcensus")
        _insert_metadata(
            db, source_type="microcensus", n_persons=n_persons, n_trips=n_trips, n_acts=n_acts,
            hot_polygon_types=loaded, has_pt_static=False,
        )
        db.execute("CHECKPOINT")  # keep the finished db on disk even if validation fails
        try:
            validate_full(db, "microcensus")
        except AssertionError:
            # microcensus validation is a hard failure: an unenriched db silently
            # blanks the webmap's comparison panels, so surface a non-zero exit
            log.error("Validation FAILED for microcensus.duckdb - failing the build")
            raise
    finally:
        db.close()
    log.info("=== microcensus.duckdb build DONE -> %s (%.1f MB)",
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
            grid_resolutions_m, bbox_lv95, hot_polygon_types,
            h3_resolutions, has_pt_static
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            SCHEMA_VERSION,
            datetime.now(timezone.utc),
            source_type,
            None,
            None,
            int(n_persons), int(n_trips), int(n_acts),
            [],
            list(CH_BBOX_LV95),
            list(hot_polygon_types),
            list(H3_RESOLUTIONS),
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
