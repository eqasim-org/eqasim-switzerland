"""Bundle a MATSim run's raw outputs into a single zip for the webmap upload.

The zip only collects files; it converts nothing. The webmap's own ingest
(webmap repo, dataset-backend/ingest.py) reads MATSim's output_*.csv.gz and
the synthesis CSVs directly and does all the processing there. The bundle has
no dependency on /cluster paths or synpp cache hashes.

Contents:

  Required (the webmap build fails without these):
    matsim/output_trips.csv.gz
    matsim/output_activities.csv.gz
    matsim/output_persons.csv.gz
    matsim/output_network.xml.gz
    matsim/output_events.xml.gz
    matsim/output_transitSchedule.xml.gz

  Optional, tier "full" only (build succeeds, features degrade):
    matsim/output_plans.xml.gz           - without it: Spider, Node Flows and
                                           Zone Flows are empty
    synthesis/<prefix>_persons.csv       - person -> household link
    synthesis/<prefix>_households.csv    - without these two: income, cars,
                                           bikes, OV-Gueteklasse charts empty

  manifest.json carries the run name, sample rate and a sha256 per file.

Usage:
    python3 -m analysis.export_for_webmap [--matsim-dir <run-cache.cache>]
        [--out <dir-or-zip>] [--tier full|minimal] [--dry-run]
        [--cache-dir <synpp-cache>] [--data-path <pipeline-data>] [--home-pipe <root>]
        [--output-path <synthesis output_path>]

Also usable as a synpp stage (`analysis.export_for_webmap` in config `run:`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from analysis.webmap_export.sources import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_HOME_PIPE,
    discover_sample_rate,
    discover_scale_pt,
    discover_synthetic,
)

log = logging.getLogger(__name__)

TIERS = ("full", "minimal")

_PRECOMPRESSED = {".gz", ".zip", ".parquet", ".png", ".7z", ".bz2", ".xz"}
_CHUNK = 8 << 20


def _newest_run_cache(cache_dir: Path) -> Optional[Path]:
    candidates = []
    for marker in cache_dir.glob("matsim.simulation.run__*.p"):
        cache = marker.with_suffix(".cache")
        if (cache / "simulation_output").is_dir():
            candidates.append((marker.stat().st_mtime, cache))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _add_file(zf: zipfile.ZipFile, arcname: str, path: Path, level: int) -> dict:
    """Stream one file into the zip, hashing as it goes. Returns its manifest row."""
    stat = path.stat()
    info = zipfile.ZipInfo(arcname,
                           date_time=datetime.fromtimestamp(stat.st_mtime).timetuple()[:6])
    info.compress_type = (zipfile.ZIP_STORED if path.suffix.lower() in _PRECOMPRESSED
                          else zipfile.ZIP_DEFLATED)
    info.file_size = stat.st_size
    info.external_attr = 0o644 << 16

    digest = hashlib.sha256()
    with path.open("rb") as src, zf.open(info, "w",
                                          force_zip64=stat.st_size >= 2**31) as dst:
        while chunk := src.read(_CHUNK):
            digest.update(chunk)
            dst.write(chunk)

    method = "stored" if info.compress_type == zipfile.ZIP_STORED else f"deflate-{level}"
    log.info("  + %-46s %8.1f MB (%s)", arcname, stat.st_size / 1e6, method)
    return {
        "arcname": arcname,
        "bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def build_bundle(
    matsim_dir: Path,
    out: Path,
    *,
    tier: str = "full",
    level: int = 6,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    home_pipe: Path = DEFAULT_HOME_PIPE,
    output_path: Optional[Path] = None,
    dry_run: bool = False,
) -> Path:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    syn = discover_synthetic(matsim_dir, cache_dir=cache_dir, data_path=data_path,
                             home_pipe=home_pipe, output_path=output_path)

    # Archived under their own names: the webmap recognises the files by name
    # (and the trips/activities layout by header).
    def matsim(path: Optional[Path], default: str) -> tuple[str, Optional[Path]]:
        return f"matsim/{path.name if path else default}", path

    required = dict([
        matsim(syn.output_trips_csv, "output_trips.csv.gz"),
        matsim(syn.output_activities_csv, "output_activities.csv.gz"),
        matsim(syn.output_persons_csv, "output_persons.csv.gz"),
        matsim(syn.output_network_xml, "output_network.xml.gz"),
        matsim(syn.output_events_xml, "output_events.xml.gz"),
        matsim(syn.output_transit_schedule_xml, "output_transitSchedule.xml.gz"),
    ])

    households_csv = syn.persons_csv.with_name(
        syn.persons_csv.name.replace("_persons.csv", "_households.csv"))
    optional = dict([
        matsim(syn.output_plans_xml, "output_plans.xml.gz"),
        (f"synthesis/{syn.persons_csv.name}", syn.persons_csv),
        (f"synthesis/{households_csv.name}", households_csv),
    ])

    run_name = matsim_dir.name.replace(".cache", "")
    short = run_name.split("__")[-1][:8] or "run"
    if out.suffix.lower() != ".zip":
        out = out / f"webmap_inputs_{short}_{tier}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    missing_required = [k for k, v in required.items() if v is None or not v.exists()]
    if missing_required:
        for arc in missing_required:
            log.error("  ! REQUIRED MISSING: %s", arc)
        raise FileNotFoundError(
            f"Cannot build bundle: {len(missing_required)} required file(s) missing: "
            + ", ".join(missing_required))

    files = dict(required)
    if tier == "full":
        for arc, path in optional.items():
            if path is not None and path.exists():
                files[arc] = path
            else:
                log.warning("  ! MISSING optional: %s", arc)

    if dry_run:
        log.info("DRY RUN - tier=%s  matsim_dir=%s", tier, matsim_dir)
        for arc, path in files.items():
            log.info("  . %-46s %8.1f MB", arc, path.stat().st_size / 1e6)
        return out

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tier": tier,
        "run_name": run_name,
        "sample_rate": discover_sample_rate(matsim_dir, home_pipe=home_pipe),
        "scale_pt_to_full_population": discover_scale_pt(home_pipe=home_pipe),
        "files": [],
    }

    tmp = out.with_suffix(".zip.part")
    total_raw = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=level, allowZip64=True) as zf:
        for arc, path in files.items():
            row = _add_file(zf, arc, path, level)
            manifest["files"].append(row)
            total_raw += row["bytes"]
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    tmp.replace(out)

    size = out.stat().st_size
    log.info("bundle DONE -> %s (%.2f GB, %.0f%% of raw)",
             out, size / 1e9, 100 * size / total_raw if total_raw else 0)
    return out


def configure(context):
    context.stage("matsim.simulation.run")
    context.config("webmap_bundle_tier", "full")
    context.config("webmap_bundle_path", "")
    context.config("output_path")


def execute(context):
    matsim_dir = Path(context.stage("matsim.simulation.run"))
    tier = str(context.config("webmap_bundle_tier")).strip().lower()
    configured = str(context.config("webmap_bundle_path")).strip()
    out = Path(configured) if configured else matsim_dir / "simulation_output" / "webmap"
    zip_path = build_bundle(matsim_dir, out, tier=tier, cache_dir=matsim_dir.parent,
                            output_path=Path(context.config("output_path")))
    return {"bundle": str(zip_path), "bytes": zip_path.stat().st_size, "tier": tier}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    matsim_dir: Optional[Path] = None
    out: Optional[Path] = None
    tier, level, dry_run = "full", 6, False
    cache_dir, data_path, home_pipe = DEFAULT_CACHE_DIR, DEFAULT_DATA_PATH, DEFAULT_HOME_PIPE
    output_path: Optional[Path] = None

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--matsim-dir":
            matsim_dir = Path(argv[i + 1]); i += 2; continue
        if a == "--out":
            out = Path(argv[i + 1]); i += 2; continue
        if a == "--tier":
            tier = argv[i + 1]; i += 2; continue
        if a == "--level":
            level = int(argv[i + 1]); i += 2; continue
        if a == "--cache-dir":
            cache_dir = Path(argv[i + 1]); i += 2; continue
        if a == "--data-path":
            data_path = Path(argv[i + 1]); i += 2; continue
        if a == "--output-path":
            output_path = Path(argv[i + 1]); i += 2; continue
        if a == "--home-pipe":
            home_pipe = Path(argv[i + 1]); i += 2; continue
        if a in ("--dry-run", "-n"):
            dry_run = True; i += 1; continue
        if a in ("--help", "-h"):
            print(__doc__)
            return 0
        log.error("unknown argument %r", a)
        return 2

    if matsim_dir is None:
        matsim_dir = _newest_run_cache(cache_dir)
        if matsim_dir is None:
            log.error("No completed matsim.simulation.run cache under %s", cache_dir)
            return 2
    log.info("matsim_dir = %s", matsim_dir)

    if out is None:
        out = matsim_dir / "simulation_output" / "webmap"
    build_bundle(matsim_dir, out, tier=tier, level=level, cache_dir=cache_dir,
                 data_path=data_path, home_pipe=home_pipe, output_path=output_path,
                 dry_run=dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
