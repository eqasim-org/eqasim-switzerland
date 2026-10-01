"""Resolve input-file paths for a given source ('synthetic' | 'microcensus').
Missing optional inputs become None - the caller decides how to react."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Columns synthesis.output writes to <output_path>/switzerland_persons.csv. A CSV
# missing any of these is from an older pipeline version - see _pick_persons_csv.
SYNTHESIS_PERSON_COLUMNS = frozenset({
    "person_id", "household_id", "age", "sex", "employed",
    "has_driving_license", "pt_subscription", "mz_person_id", "canton_id",
})


@dataclass
class SyntheticSources:
    """Inputs needed for the synthetic.duckdb build."""
    persons_csv: Path
    statpop_persons_pickle: Optional[Path]
    households_pickle: Optional[Path]
    enriched_pickle: Optional[Path]
    output_trips_csv: Optional[Path]
    output_activities_csv: Optional[Path]
    output_persons_csv: Optional[Path]
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


# synpp working_directory (config_andrew.yml); stages use the run cache's parent instead
DEFAULT_CACHE_DIR = Path("/cluster/work/ivt_vpl/anding/cache_uv")
DEFAULT_DATA_PATH = Path("/cluster/project/cmdp/ch_data/pipeline")
DEFAULT_HOME_PIPE = Path("/cluster/home/anding/ch")


def _newest_cache(name_prefix: str, cache_dir: Path) -> Optional[Path]:
    """Pick the newest matching synpp .p file (different hashes from re-runs)."""
    candidates = sorted(cache_dir.glob(f"{name_prefix}__*.p"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _config_output_paths(home_pipe: Path) -> list[Path]:
    """Every absolute `output_path:` declared in the repo's config*.yml files.

    synthesis/output.py writes switzerland_*.csv directly into <output_path>/.
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


def _csv_columns(path: Path) -> Optional[set[str]]:
    """Header of a ';'-separated CSV; None if unreadable."""
    try:
        with path.open(errors="ignore") as f:
            return set(f.readline().strip().split(";"))
    except OSError as exc:
        log.warning("could not read header of %s: %s", path, exc)
        return None


def _pick_persons_csv(candidates: list[Path]) -> Path:
    """Newest candidate carrying the current synthesis.output persons columns.

    The legacy webmap_data/synthetic/*.parquet files are no longer written by
    synthesis.output and belong to older populations, so they are not candidates.
    """
    seen: set[Path] = set()
    complete: list[Path] = []
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
        cols = _csv_columns(c)
        if cols is None:
            continue
        missing = SYNTHESIS_PERSON_COLUMNS - cols
        if missing:
            log.warning("skipping persons CSV %s: missing column(s): %s",
                        c, ", ".join(sorted(missing)))
            continue
        complete.append(c)

    if not complete:
        raise FileNotFoundError(
            "No synthesis.output persons CSV found (looked for switzerland_persons.csv in: "
            + ", ".join(str(c.parent) for c in candidates) + ")")
    chosen = max(complete, key=lambda p: p.stat().st_mtime)
    log.info("persons CSV -> %s (%d candidate(s))", chosen, len(complete))
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
    output_path: Optional[Path] = None,
) -> SyntheticSources:
    """Best-effort discovery - missing inputs become None.

    output_path is the synpp `output_path` config (where synthesis.output writes);
    without it every absolute output_path in the repo's config*.yml is tried.
    """
    sim_out = matsim_dir / "simulation_output"

    output_paths = [output_path] if output_path else _config_output_paths(home_pipe)
    persons_csv = _pick_persons_csv(
        [Path(p) / "switzerland_persons.csv" for p in output_paths])

    canton_shp, bezirk_shp, gemeinde_shp = _swisstopo_paths(data_path)

    return SyntheticSources(
        persons_csv=persons_csv,
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
        output_persons_csv=_pick_existing(
            sim_out / "output_persons.csv.gz", sim_out / "output_persons.csv"),
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
