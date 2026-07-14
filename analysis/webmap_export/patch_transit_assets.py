"""Patch the PT/transit static_assets into an existing synthetic.duckdb in place (no full rebuild).
DuckDB takes a single write lock - if the backend holds the file open this fails cleanly; patch a copy then.

Usage: venv/bin/python3 -m analysis.webmap_export.patch_transit_assets [--matsim-dir <cache>] [--db <path>]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from . import events_extras, static_assets, transit
from .run_standalone import _newest_run_cache
from .sources import DEFAULT_CACHE_DIR, discover_sample_rate, discover_scale_pt

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("webmap_export.patch_transit_assets")


def _pick(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def patch(matsim_dir: Path, db_path: Path) -> int:
    import duckdb
    sim = matsim_dir / "simulation_output"
    events = _pick(sim / "output_events.xml.gz", sim / "output_events.xml")
    sched = _pick(sim / "output_transitSchedule.xml.gz", sim / "output_transitSchedule.xml")
    if events is None or sched is None:
        log.error("missing events (%s) or schedule (%s)", events, sched)
        return 2
    if not db_path.exists():
        log.error("duckdb not found: %s", db_path)
        return 2

    sample_rate = discover_sample_rate(matsim_dir)
    scale_pt = discover_scale_pt()
    run_name = matsim_dir.name.replace(".cache", "")
    log.info("patching %s", db_path)
    log.info("  events=%s  schedule=%s  sample_rate=%s  scale_pt=%s  run=%s",
             events, sched, sample_rate, scale_pt, run_name)

    db = duckdb.connect(str(db_path))  # read-write; errors cleanly if locked
    try:
        db.execute("INSTALL spatial; LOAD spatial;")
        log.info("parsing events (PT-only)…")
        handler = events_extras.parse_events(db, events, track_speeds=False)
        log.info("  boarding keys=%d, transfer stops=%d",
                 len(handler.board_acc), len(handler.transfer_data))
        transit.build_all(db, sched, handler.board_acc, handler.transfer_data,
                          sample_rate, scale_pt)
        static_assets.build_metadata_asset(
            db, sample_rate=sample_rate, run_name=run_name,
            scaled_to_full_population=bool(sample_rate and scale_pt),
        )
        db.execute("CHECKPOINT")
    finally:
        db.close()
    log.info("patch DONE")
    return 0


def main(argv: list[str]) -> int:
    matsim_dir: Path | None = None
    db_path: Path | None = None
    i = 0
    while i < len(argv):
        if argv[i] == "--matsim-dir":
            matsim_dir = Path(argv[i + 1]); i += 2; continue
        if argv[i] == "--db":
            db_path = Path(argv[i + 1]); i += 2; continue
        i += 1
    if matsim_dir is None:
        matsim_dir = _newest_run_cache(DEFAULT_CACHE_DIR)
        if matsim_dir is None:
            log.error("no completed matsim.simulation.run cache found")
            return 2
    if db_path is None:
        db_path = matsim_dir / "simulation_output" / "webmap" / "synthetic.duckdb"
    return patch(matsim_dir, db_path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
