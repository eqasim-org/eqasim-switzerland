"""Read optional hourly vehicle breakdowns for count-station details."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matsim.config.tools import resolve_output_file
from ..paths import configure_simulation_path, get_simulation_path


VEHICLE_COLUMNS = (
    "cars", "trucks", "bikes", "buses", "cars_french", "cars_crossborder",
    "cars_swiss", "cars_unknown",
)
SWISS_RATIO = "MATSim Swiss / (French + cross-border) (%)"
FOREIGN_SHARE = "MATSim foreign cars (%)"
FRENCH_CROSSBORDER = "MATSim French + cross-border (raw vehicle passages)"
DETAIL_COLOR_LABELS = {
    "MATSim cars french (raw vehicle passages)": "French cars (raw passages)",
    FRENCH_CROSSBORDER: "French + cross-border cars (raw passages)",
    "MATSim cars swiss (raw vehicle passages)": "Swiss cars (raw passages)",
    SWISS_RATIO: "Swiss / (French + cross-border) (%)",
    FOREIGN_SHARE: "Foreign cars (%)",
}


def configure(context):
    configure_simulation_path(context)


def execute(context):

    # Load optional vehicle statistics once for daily and peak comparisons.

    return read_detailed_statistics(get_simulation_path(context))


def validate(context):
    """Refresh the shared cache when the optional output appears or changes."""
    path = statistics_path(get_simulation_path(context))
    if not path.is_file():
        return None
    stat = path.stat()
    return [str(path), stat.st_mtime_ns, stat.st_size]


def statistics_path(simulation_path):
    return Path(resolve_output_file(
        Path(simulation_path) / "output_detailed_link_statistics.csv.gz", required=False
    ))


def read_detailed_statistics(simulation_path, link_ids=None):
    """Read a compact link/hour table; an optional subset is useful for diagnostics.

    Missing links remain unavailable, rather than being reported as zero.
    For represented links, absent hourly records are treated as zero passages.
    A single indexed table avoids creating a separate DataFrame for every link
    in the full network. The returned table is shared read-only by consumers.
    """
    columns = ["link_id", "hour", *VEHICLE_COLUMNS]
    empty = pd.DataFrame(columns=columns).set_index(["link_id", "hour"])
    path = statistics_path(simulation_path)
    if not path.is_file():
        logging.getLogger("synpp").info("Optional detailed statistics not found: %s; skipping vehicle breakdowns.", path)
        return empty
    selected = None if link_ids is None else set(map(str, link_ids))
    pieces = []
    for chunk in pd.read_csv(path, usecols=columns, dtype={"link_id": str}, chunksize=250_000):
        if selected is not None:
            chunk = chunk[chunk.link_id.isin(selected)].copy()
        if chunk.empty:
            continue
        values = chunk[["hour", *VEHICLE_COLUMNS]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Detailed link statistics must contain finite non-negative counts and hours.")
        if (chunk.hour % 1 != 0).any():
            raise ValueError("Detailed link statistics must use whole-hour bins.")
        # A daily profile is 00:00–24:00; do not wrap next-day traffic into it.
        chunk = chunk[chunk.hour < 24]
        pieces.append(chunk)
    if not pieces:
        return empty
    data = pd.concat(pieces, ignore_index=True)
    if data.duplicated(["link_id", "hour"]).any():
        raise ValueError("Detailed link statistics has duplicate link/hour rows.")
    data["hour"] = data["hour"].astype("uint8")
    data["link_id"] = data["link_id"].astype("category")
    return data.set_index(["link_id", "hour"])[list(VEHICLE_COLUMNS)].sort_index()


def add_point_details(points, statistics, sample_size):
    """Attach available vehicle totals for the shared click-details panel.

    Sum each matched current link once. Missing outputs leave the usual
    comparison metrics usable, without inventing zero vehicle totals.
    """
    result = points.copy()
    records = [point_details(ids, statistics, sample_size)[0]
               for ids in result.matched_link_ids]
    details = pd.DataFrame(records, index=result.index)
    # These maps have no observed profile to compare with the simulated curve.
    details = details.drop(columns=[field for field in details if field.startswith("MATSim profile")])
    for field in details:
        result[field] = details[field]
    return result, list(details.columns)


def point_details(link_ids, statistics, sample_size, peak_hours=None):
    """Summarize a matched link once, regardless of its detector count.

    Tooltip category totals are raw simulated vehicle passages. For the chart,
    expand sampled cars/trucks; scheduled buses are not population-sampled.
    Bikes are displayed separately, not added to motor-vehicle traffic.
    """
    if not np.isfinite(sample_size) or not 0 < sample_size <= 1:
        raise ValueError("input_downsampling must be in (0, 1].")
    ids = list(dict.fromkeys(map(str, link_ids)))
    if statistics is None or statistics.empty or not ids or any(link_id not in statistics.index for link_id in ids):
        return {}, None
    hourly = statistics.loc[ids].groupby(level="hour", observed=True).sum().reindex(range(24), fill_value=0)
    start, end = peak_hours or (0, 24)
    totals = hourly.loc[start:end - 1].sum()
    details = {f"MATSim {column.replace('_', ' ')} (raw vehicle passages)": round(float(totals[column]), 2)
               for column in VEHICLE_COLUMNS}
    denominator = float(totals.cars_french + totals.cars_crossborder)
    details[FRENCH_CROSSBORDER] = denominator
    details[SWISS_RATIO] = round(100 * float(totals.cars_swiss) / denominator, 2) if denominator > 0 else np.nan
    details[FOREIGN_SHARE] = round(100 * denominator / float(totals.cars), 2) if totals.cars > 0 else np.nan
    details["MATSim detailed statistics period"] = f"{start:02d}:00–{end:02d}:00"
    details["MATSim profile scaling"] = f"Cars/trucks ×{1 / sample_size:g}; buses unscaled; bikes excluded"
    details["MATSim categories"] = "Car categories are subsets of cars, not additional vehicles or unique people."
    details["MATSim profile source"] = "Detailed output file; may differ from the comparison's iteration-averaged flow."
    profile = {"hour": list(range(24)),
               "flow": ((hourly.cars + hourly.trucks) / sample_size + hourly.buses).tolist()}
    return details, profile
