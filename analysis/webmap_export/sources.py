"""Resolve input-file paths for a given source ('synthetic' | 'microcensus').
Missing optional inputs become None - the caller decides how to react."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Columns the webmap's person panels need. A parquet missing any of these is a
# stale/partial leftover, not a synthesis.output product - see _pick_persons_parquet.
REQUIRED_PERSON_COLUMNS = frozenset({
    "person_id", "household_id", "age", "sex",
    "car_availability", "has_driving_license", "employed",
    "subscriptions_ga", "subscriptions_halbtax", "subscriptions_verbund",
    "subscriptions_strecke", "subscriptions_gleis7", "subscriptions_junior",
    "subscriptions_other",
})


@dataclass
class SyntheticSources:
    """Inputs needed for the synthetic.duckdb build."""
    persons_parquet: Path
    statpop_persons_pickle: Optional[Path]
    households_pickle: Optional[Path]
    enriched_pickle: Optional[Path]
    output_trips_csv: Optional[Path]
    output_activities_csv: Optional[Path]
    output_plans_xml: Optional[Path]
    output_events_xml: Optional[Path]
    output_network_xml: Optional[Path]
    output_transit_schedule_xml: Optional[Path]
    link_speeds_parquet: Optional[Path]
    swisstopo_canton_shp: Optional[Path]
    swisstopo_bezirk_shp: Optional[Path]
    swisstopo_gemeinde_shp: Optional[Path]
    json_preview_dir: Optional[Path]


@dataclass
class MicrocensusSources:
    """Inputs needed for the microcensus.duckdb build."""
    household_persons_pickle: Path
    households_pickle: Path
    trips_pickle: Path
    respondents_pickle: Optional[Path]
    swisstopo_canton_shp: Optional[Path]
    swisstopo_bezirk_shp: Optional[Path]
    swisstopo_gemeinde_shp: Optional[Path]


DEFAULT_CACHE_DIR = Path("/cluster/work/ivt_vpl/anding/cache")
DEFAULT_DATA_PATH = Path("/cluster/project/cmdp/ch_data/pipeline")
DEFAULT_HOME_PIPE = Path("/cluster/home/anding/ch")


def _newest_cache(name_prefix: str, cache_dir: Path) -> Optional[Path]:
    """Pick the newest matching synpp .p file (different hashes from re-runs)."""
    candidates = sorted(cache_dir.glob(f"{name_prefix}__*.p"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _config_output_paths(home_pipe: Path) -> list[Path]:
    """Every absolute `output_path:` declared in the repo's config*.yml files.

    synthesis/output.py writes its parquets to <output_path>/webmap_data/synthetic/,
    so this is where the genuine persons artifact lives.
    """
    out: list[Path] = []
    repo = home_pipe / "ch-zh-synpop"
    for yml in sorted(repo.glob("config*.yml")):
        try:
            for line in yml.read_text(errors="ignore").splitlines():
                s = line.strip()
                if s.startswith("output_path:"):
                    v = s.split(":", 1)[1].split("#")[0].strip().strip("\"'")
                    if v.startswith("/"):
                        out.append(Path(v))
        except OSError:
            continue
    return out


def _parquet_columns(path: Path) -> Optional[set[str]]:
    """Column names from the parquet footer only (no data read); None if unreadable."""
    try:
        import pyarrow.parquet as pq
        return set(pq.ParquetFile(path).schema_arrow.names)
    except Exception as exc:  # noqa: BLE001 - a bad candidate must not kill discovery
        log.warning("could not read parquet schema of %s: %s", path, exc)
        return None


def _pick_persons_parquet(candidates: list[Path], fallback: Path) -> Path:
    """Newest candidate that carries the full expected column set.

    Selecting on mtime alone once silently picked a 10-column leftover over the
    real 21-column artifact, so completeness is the primary key and mtime only
    breaks ties among complete files.
    """
    seen: set[Path] = set()
    existing: list[Path] = []
    for c in candidates:
        try:
            if not c.exists():
                continue
            key = c.resolve()
        except OSError:
            # config*.yml may name another user's scratch dir we cannot stat
            continue
        if key in seen:
            continue
        seen.add(key)
        existing.append(c)

    if not existing:
        log.error("no persons parquet found; falling back to %s (does not exist)", fallback)
        return fallback

    complete: list[Path] = []
    for c in existing:
        cols = _parquet_columns(c)
        if cols is None:
            continue
        missing = REQUIRED_PERSON_COLUMNS - cols
        if missing:
            log.warning(
                "skipping persons parquet %s: missing %d expected column(s): %s",
                c, len(missing), ", ".join(sorted(missing)),
            )
            continue
        complete.append(c)

    pool = complete or existing
    chosen = max(pool, key=lambda p: p.stat().st_mtime)
    if not complete:
        log.error(
            "NO persons parquet has the full expected column set - falling back to %s; "
            "subscriptions/car_availability panels in the webmap will be NULL", chosen,
        )
    else:
        log.info("persons parquet -> %s (%d on disk, %d complete)",
                 chosen, len(existing), len(complete))
    return chosen


def _swisstopo_paths(data_path: Path) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    spatial = data_path / "spatial"
    canton = next(iter((spatial / "canton").glob("swissBOUNDARIES3D_*_TLM_KANTONSGEBIET.shp")), None) if (spatial / "canton").exists() else None
    bezirk = next(iter((spatial / "districts").glob("swissBOUNDARIES3D_*_TLM_BEZIRKSGEBIET.shp")), None) if (spatial / "districts").exists() else None

    gemeinde = None
    muni_root = spatial / "municipality"
    if muni_root.exists():
        years = sorted([p for p in muni_root.iterdir() if p.is_dir() and p.name.isdigit()], reverse=True)
        for y in years:
            cand = next(iter(y.glob("swissBOUNDARIES3D_*_TLM_HOHEITSGEBIET.shp")), None)
            if cand:
                gemeinde = cand
                break
    return canton, bezirk, gemeinde


def discover_synthetic(
    matsim_dir: Path,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
    home_pipe: Path = DEFAULT_HOME_PIPE,
) -> SyntheticSources:
    """Best-effort discovery - missing inputs become None."""
    sim_out = matsim_dir / "simulation_output"

    # The genuine artifact first: synthesis/output.py writes to
    # <output_path>/webmap_data/synthetic/ and never into its own synpp cache dir
    # (those .cache dirs stay empty), so anything found under synthesis.output__*.cache
    # is a leftover. Candidates are ranked by column completeness, then mtime.
    _persons_candidates = [
        p / "webmap_data" / "synthetic" / "switzerland_persons.parquet"
        for p in _config_output_paths(home_pipe)
    ]
    _persons_candidates.append(
        cache_dir / "synthesis_output" / "webmap_data" / "synthetic" / "switzerland_persons.parquet"
    )
    _persons_candidates.append(home_pipe / "switzerland_persons.parquet")
    _persons_candidates.extend(
        sorted(cache_dir.glob("synthesis.output__*.cache/switzerland_persons.parquet"))
    )
    persons_parquet = _pick_persons_parquet(
        _persons_candidates, home_pipe / "switzerland_persons.parquet"
    )

    canton_shp, bezirk_shp, gemeinde_shp = _swisstopo_paths(data_path)

    return SyntheticSources(
        persons_parquet=persons_parquet,
        statpop_persons_pickle=_newest_cache("data.statpop.persons", cache_dir),
        households_pickle=_newest_cache("data.statpop.households", cache_dir),
        enriched_pickle=_newest_cache("synthesis.population.enriched", cache_dir),

        output_trips_csv=_pick_existing(
            sim_out / "eqasim_trips.csv",
            sim_out / "output_trips.csv.gz", sim_out / "output_trips.csv",
        ),
        output_activities_csv=_pick_existing(
            sim_out / "eqasim_activities.csv",
            sim_out / "output_activities.csv.gz", sim_out / "output_activities.csv",
        ),
        output_plans_xml=_pick_existing(sim_out / "output_plans.xml.gz", sim_out / "output_plans.xml"),
        output_events_xml=_pick_existing(sim_out / "output_events.xml.gz", sim_out / "output_events.xml"),
        output_network_xml=_pick_existing(sim_out / "output_network.xml.gz", sim_out / "output_network.xml"),
        output_transit_schedule_xml=_pick_existing(
            sim_out / "output_transitSchedule.xml.gz", sim_out / "output_transitSchedule.xml"),
        link_speeds_parquet=_pick_existing(sim_out / "link_speeds.parquet"),
        swisstopo_canton_shp=canton_shp,
        swisstopo_bezirk_shp=bezirk_shp,
        swisstopo_gemeinde_shp=gemeinde_shp,
        json_preview_dir=_pick_existing(sim_out / "webmap" / "public" / "data" / "matsim"),
    )


def discover_microcensus(
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    data_path: Path = DEFAULT_DATA_PATH,
) -> MicrocensusSources:
    canton_shp, bezirk_shp, gemeinde_shp = _swisstopo_paths(data_path)
    hp = _newest_cache("data.microcensus.household_persons", cache_dir)
    hh = _newest_cache("data.microcensus.households", cache_dir)
    tr = _newest_cache("data.microcensus.trips", cache_dir)
    re_ = _newest_cache("data.microcensus.persons", cache_dir)  # survey respondents
    if hp is None or hh is None or tr is None:
        raise FileNotFoundError(
            "Missing microcensus caches under "
            f"{cache_dir}: need data.microcensus.household_persons, "
            "data.microcensus.households and data.microcensus.trips"
        )
    return MicrocensusSources(
        household_persons_pickle=hp,
        households_pickle=hh,
        trips_pickle=tr,
        respondents_pickle=re_,
        swisstopo_canton_shp=canton_shp,
        swisstopo_bezirk_shp=bezirk_shp,
        swisstopo_gemeinde_shp=gemeinde_shp,
    )


def _pick_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def discover_sample_rate(
    matsim_dir: Path, *, home_pipe: Path = DEFAULT_HOME_PIPE,
) -> Optional[float]:
    """Per-run population sample rate (e.g. 0.05 = 5%): the run's own
    output_config.xml sampleSize first, then config.yml input_downsampling; None if not found."""
    import re

    cfg = matsim_dir / "simulation_output" / "output_config.xml"
    if cfg.exists():
        try:
            txt = cfg.read_text(errors="ignore")
            m = re.search(r'name="sampleSize"\s+value="([0-9.eE+-]+)"', txt)
            if m:
                v = float(m.group(1))
                if 0 < v <= 1:
                    return v
        except (OSError, ValueError):
            pass

    yml = home_pipe / "ch-zh-synpop" / "config.yml"
    if yml.exists():
        try:
            for line in yml.read_text(errors="ignore").splitlines():
                s = line.strip()
                if s.startswith("input_downsampling:"):
                    v = float(s.split(":", 1)[1].split("#")[0].strip())
                    if 0 < v <= 1:
                        return v
        except (OSError, ValueError):
            pass
    return None


def discover_scale_pt(
    *, home_pipe: Path = DEFAULT_HOME_PIPE, default: bool = False,
) -> bool:
    """Read the scale_pt_to_full_population flag from config.yml (default False)."""
    yml = home_pipe / "ch-zh-synpop" / "config.yml"
    if yml.exists():
        try:
            for line in yml.read_text(errors="ignore").splitlines():
                s = line.strip()
                if s.startswith("scale_pt_to_full_population:"):
                    val = s.split(":", 1)[1].split("#")[0].strip().lower()
                    return val in ("true", "1", "yes", "on")
        except OSError:
            pass
    return default
