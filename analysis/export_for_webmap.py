"""Bundle webmap inputs from a MATSim run into a single zip for download.

Produces a self-contained zip with no dependency on /cluster paths or synpp
cache hashes. Raw pickles are not bundled; instead persons.parquet and
households.parquet are built at export time with home coords and household
attributes baked in.

Contents:

  Required (synthetic.duckdb build fails without these):
    matsim/eqasim_trips.csv
    matsim/eqasim_activities.csv
    matsim/output_network.xml.gz
    matsim/output_events.xml.gz
    matsim/output_transitSchedule.xml.gz
    persons.parquet

  Optional (build succeeds, features degrade):
    matsim/output_plans.xml.gz   — without it: ~15% of agents get NULL home
                                   coordinates in the webapp preprocessor
    households.parquet            — without it: income, cars, bikes, OV-
                                   Gueteklasse charts all empty

Usage:
    python3 -m analysis.export_for_webmap [--matsim-dir <run-cache.cache>]
        [--out <dir-or-zip>] [--tier full|minimal] [--dry-run]
        [--cache-dir <synpp-cache>] [--data-path <pipeline-data>] [--home-pipe <root>]

Also usable as a synpp stage (`analysis.export_for_webmap` in config `run:`).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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


def _build_persons_parquet(
    persons_parquet: Path,
    activities_csv: Optional[Path],
    statpop_pickle: Optional[Path],
    scratch: Path,
) -> Path:
    """Read the synthesis persons parquet, merge in home coords from activities
    (and optionally statpop fallback), write a self-contained parquet."""
    df = pd.read_parquet(persons_parquet)
    log.info("persons: %d rows from %s", len(df), persons_parquet)

    home_x = pd.Series(np.nan, index=df.index, dtype="float64")
    home_y = pd.Series(np.nan, index=df.index, dtype="float64")

    if activities_csv is not None and activities_csv.exists():
        acts = pd.read_csv(activities_csv, sep=";",
                           usecols=["person_id", "activity_index", "purpose", "x", "y"])
        acts["person_id"] = pd.to_numeric(acts["person_id"], errors="coerce")
        acts = acts.dropna(subset=["person_id"])
        acts["person_id"] = acts["person_id"].astype("int64")
        homes = (acts[acts["purpose"] == "home"]
                 .sort_values(["person_id", "activity_index"])
                 .drop_duplicates("person_id", keep="first")[["person_id", "x", "y"]])
        merged = df[["person_id"]].merge(homes, on="person_id", how="left")
        home_x = merged["x"].to_numpy()
        home_y = merged["y"].to_numpy()
        home_x = pd.Series(home_x, index=df.index, dtype="float64")
        home_y = pd.Series(home_y, index=df.index, dtype="float64")
        n_from_acts = int(home_x.notna().sum())
        log.info("persons: %d/%d home coords from activities", n_from_acts, len(df))

    needs_statpop = home_x.isna()
    if needs_statpop.any() and statpop_pickle is not None and statpop_pickle.exists():
        if "statpop_person_id" in df.columns:
            with open(statpop_pickle, "rb") as f:
                sp = pickle.load(f)
            sp = sp[["person_id", "home_x", "home_y"]].rename(
                columns={"person_id": "statpop_person_id"})
            merged = df[["statpop_person_id"]].merge(sp, on="statpop_person_id", how="left")
            home_x = home_x.where(~needs_statpop, merged["home_x"].to_numpy())
            home_y = home_y.where(~needs_statpop, merged["home_y"].to_numpy())
            n_filled = int(needs_statpop.sum() - home_x.isna().sum())
            log.info("persons: %d more home coords from statpop fallback", n_filled)
        else:
            log.warning("persons: parquet has no statpop_person_id column, cannot use statpop fallback")

    bad = (home_x < 2_000_000) | (home_y < 1_000_000)
    home_x = home_x.where(~bad, other=np.nan)
    home_y = home_y.where(~bad, other=np.nan)
    df["home_x"] = home_x
    df["home_y"] = home_y

    n_with_home = int(df["home_x"].notna().sum())
    n_without = len(df) - n_with_home
    log.info("persons: %d with home coords, %d without (%.1f%%)",
             n_with_home, n_without, 100 * n_without / len(df) if len(df) else 0)

    out = scratch / "persons.parquet"
    df.to_parquet(out, index=False)
    log.info("persons: wrote %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def _build_households_parquet(enriched_pickle: Path, scratch: Path) -> Path:
    """Extract household attributes from the enriched pickle into a clean parquet."""
    with open(enriched_pickle, "rb") as f:
        enriched = pickle.load(f)

    df = (enriched[["household_id", "income_class",
                    "number_of_cars_class", "number_of_bikes_class", "ovgk"]]
          .drop_duplicates("household_id")
          .copy())
    df = df.rename(columns={
        "number_of_cars_class": "n_cars_class",
        "number_of_bikes_class": "n_bikes_class",
    })
    df["household_id"] = df["household_id"].astype("int64")
    for col in ("income_class", "n_cars_class", "n_bikes_class", "ovgk"):
        df[col] = df[col].astype("string")

    out = scratch / "households.parquet"
    df.to_parquet(out, index=False)
    log.info("households: %d rows, wrote %s (%.1f MB)",
             len(df), out, out.stat().st_size / 1e6)
    return out


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
    dry_run: bool = False,
) -> Path:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")

    syn = discover_synthetic(matsim_dir, cache_dir=cache_dir, data_path=data_path,
                             home_pipe=home_pipe)

    required = {
        "matsim/eqasim_trips.csv": syn.output_trips_csv,
        "matsim/eqasim_activities.csv": syn.output_activities_csv,
        "matsim/output_network.xml.gz": syn.output_network_xml,
        "matsim/output_events.xml.gz": syn.output_events_xml,
        "matsim/output_transitSchedule.xml.gz": syn.output_transit_schedule_xml,
    }

    optional = {
        "matsim/output_plans.xml.gz": syn.output_plans_xml,
    }

    run_name = matsim_dir.name.replace(".cache", "")
    short = run_name.split("__")[-1][:8] or "run"
    if out.suffix.lower() != ".zip":
        out = out / f"webmap_inputs_{short}_{tier}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    missing_required = {k: v for k, v in required.items()
                        if v is None or not v.exists()}
    if missing_required:
        for arc, _ in missing_required.items():
            log.error("  ! REQUIRED MISSING: %s", arc)
        raise FileNotFoundError(
            f"Cannot build bundle: {len(missing_required)} required file(s) missing: "
            + ", ".join(missing_required))

    has_enriched = (syn.enriched_pickle is not None and syn.enriched_pickle.exists())

    if dry_run:
        log.info("DRY RUN — tier=%s  matsim_dir=%s", tier, matsim_dir)
        for arc, path in required.items():
            log.info("  . %-46s %8.1f MB  (required)", arc, path.stat().st_size / 1e6)
        log.info("  . %-46s     (built from %s + activities)", "persons.parquet",
                 syn.persons_parquet.name)
        if tier == "full":
            for arc, path in optional.items():
                if path and path.exists():
                    log.info("  . %-46s %8.1f MB  (optional)", arc, path.stat().st_size / 1e6)
                else:
                    log.warning("  ! %-46s MISSING (optional)", arc)
            if has_enriched:
                log.info("  . %-46s     (built from enriched pickle)", "households.parquet")
            else:
                log.warning("  ! %-46s enriched pickle not found", "households.parquet")
        return out

    sample_rate = discover_sample_rate(matsim_dir, home_pipe=home_pipe)
    scale_pt = discover_scale_pt(home_pipe=home_pipe)

    with tempfile.TemporaryDirectory(prefix="webmap_export_") as scratch_str:
        scratch = Path(scratch_str)

        persons_path = _build_persons_parquet(
            syn.persons_parquet, syn.output_activities_csv,
            syn.statpop_persons_pickle, scratch)

        households_path = None
        if tier == "full" and has_enriched:
            households_path = _build_households_parquet(syn.enriched_pickle, scratch)
        elif tier == "full":
            log.warning("households.parquet skipped: enriched pickle not found at %s",
                        syn.enriched_pickle)

        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier": tier,
            "run_name": run_name,
            "sample_rate": sample_rate,
            "scale_pt_to_full_population": scale_pt,
            "files": [],
        }

        tmp = out.with_suffix(".zip.part")
        total_raw = 0
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=level, allowZip64=True) as zf:
            for arc, path in required.items():
                row = _add_file(zf, arc, path, level)
                manifest["files"].append(row)
                total_raw += row["bytes"]

            row = _add_file(zf, "persons.parquet", persons_path, level)
            manifest["files"].append(row)
            total_raw += row["bytes"]

            if tier == "full":
                for arc, path in optional.items():
                    if path and path.exists():
                        row = _add_file(zf, arc, path, level)
                        manifest["files"].append(row)
                        total_raw += row["bytes"]
                    else:
                        log.warning("  ! MISSING optional: %s", arc)

                if households_path:
                    row = _add_file(zf, "households.parquet", households_path, level)
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
    context.stage("synthesis.population.enriched")
    context.stage("data.statpop.persons")
    context.config("webmap_bundle_tier", "full")
    context.config("webmap_bundle_path", "")


def execute(context):
    matsim_dir = Path(context.stage("matsim.simulation.run"))
    tier = str(context.config("webmap_bundle_tier")).strip().lower()
    configured = str(context.config("webmap_bundle_path")).strip()
    out = Path(configured) if configured else matsim_dir / "simulation_output" / "webmap"
    zip_path = build_bundle(matsim_dir, out, tier=tier)
    return {"bundle": str(zip_path), "bytes": zip_path.stat().st_size, "tier": tier}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    matsim_dir: Optional[Path] = None
    out: Optional[Path] = None
    tier, level, dry_run = "full", 6, False
    cache_dir, data_path, home_pipe = DEFAULT_CACHE_DIR, DEFAULT_DATA_PATH, DEFAULT_HOME_PIPE

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
                 data_path=data_path, home_pipe=home_pipe, dry_run=dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
