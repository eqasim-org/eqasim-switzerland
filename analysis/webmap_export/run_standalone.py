"""Standalone webmap_export driver: rebuild synthetic.duckdb + microcensus.duckdb
without going through synpp (which would re-run MATSim and wipe simulation_output).

Usage: venv/bin/python3 -m analysis.webmap_export.run_standalone [both|synthetic|microcensus]
           [--matsim-dir <run-cache.cache>] [--cache-dir <synpp-cache>]
           [--data-path <pipeline-data>] [--home-pipe <pipe-root>]
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from . import _build_microcensus, _build_synthetic, _write_metadata_json
from .schema import SCHEMA_VERSION
from .sources import (
    DEFAULT_CACHE_DIR, DEFAULT_DATA_PATH, DEFAULT_HOME_PIPE,
    discover_sample_rate, discover_scale_pt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("webmap_export.run_standalone")


def _newest_run_cache(cache_dir: Path) -> Path | None:
    """Newest matsim.simulation.run__<hash>.cache whose .p completion marker exists."""
    candidates = []
    for marker in cache_dir.glob("matsim.simulation.run__*.p"):
        cache = marker.with_suffix(".cache")
        if (cache / "simulation_output").is_dir():
            candidates.append((marker.stat().st_mtime, cache))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main(argv: list[str]) -> int:
    mode = "both"
    matsim_dir: Path | None = None
    cache_dir = DEFAULT_CACHE_DIR
    data_path = DEFAULT_DATA_PATH
    home_pipe = DEFAULT_HOME_PIPE
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--matsim-dir":
            matsim_dir = Path(argv[i + 1]); i += 2; continue
        if a == "--cache-dir":
            cache_dir = Path(argv[i + 1]); i += 2; continue
        if a == "--data-path":
            data_path = Path(argv[i + 1]); i += 2; continue
        if a == "--home-pipe":
            home_pipe = Path(argv[i + 1]); i += 2; continue
        if a in ("both", "synthetic", "microcensus"):
            mode = a
        i += 1

    if matsim_dir is None:
        matsim_dir = _newest_run_cache(cache_dir)
        if matsim_dir is None:
            log.error("No completed matsim.simulation.run cache under %s", cache_dir)
            return 2
    log.info("Using matsim_dir = %s", matsim_dir)

    output_dir = matsim_dir / "simulation_output" / "webmap"
    output_dir.mkdir(parents=True, exist_ok=True)
    syn_path = output_dir / "synthetic.duckdb"
    mc_path = output_dir / "microcensus.duckdb"

    sample_rate = discover_sample_rate(matsim_dir, home_pipe=home_pipe)
    scale_pt = discover_scale_pt(home_pipe=home_pipe)
    run_name = matsim_dir.name.replace(".cache", "")
    log.info("sample_rate=%s  scale_pt=%s  run_name=%s", sample_rate, scale_pt, run_name)

    if mode in ("both", "synthetic"):
        _build_synthetic(
            output_db=syn_path, matsim_dir=matsim_dir,
            cache_dir=cache_dir, data_path=data_path,
            home_pipe=home_pipe, sample_rate=sample_rate, run_name=run_name,
            scale_pt=scale_pt,
        )
    if mode in ("both", "microcensus"):
        _build_microcensus(
            output_db=mc_path, cache_dir=cache_dir, data_path=data_path,
        )

    (output_dir / "schema_version.txt").write_text(SCHEMA_VERSION + "\n")
    _write_metadata_json(
        output_dir, syn_path, mc_path,
        wrote_synthetic=mode in ("both", "synthetic"),
        wrote_microcensus=mode in ("both", "microcensus"),
    )
    log.info("standalone webmap_export DONE (schema=%s) -> %s", SCHEMA_VERSION, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
