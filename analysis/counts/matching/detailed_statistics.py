"""Read optional hourly vehicle breakdowns for Transcality hover information."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matsim.config.tools import resolve_output_file


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


def read_detailed_statistics(simulation_path, link_ids):
    """Stream only matched current link IDs; never mutate simulation outputs.

    Missing links remain unavailable, rather than being reported as zero.
    For represented links, absent hourly records are treated as zero passages.
    """
    path = Path(resolve_output_file(Path(simulation_path) / "output_detailed_link_statistics.csv.gz", required=False))
    if not path.is_file():
        logging.getLogger("synpp").warning("No %s; Transcality detailed hover statistics omitted.", path)
        return {}
    selected = set(map(str, link_ids))
    pieces = []
    columns = ["link_id", "hour", *VEHICLE_COLUMNS]
    for chunk in pd.read_csv(path, usecols=columns, dtype={"link_id": str}, chunksize=250_000):
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
        return {}
    data = pd.concat(pieces, ignore_index=True)
    if data.duplicated(["link_id", "hour"]).any():
        raise ValueError("Detailed link statistics has duplicate link/hour rows.")
    return {
        link_id: group.set_index("hour")[list(VEHICLE_COLUMNS)].reindex(range(24), fill_value=0)
        for link_id, group in data.groupby("link_id", sort=False)
    }


def point_details(link_ids, statistics, sample_size, peak_hours=None):
    """Summarize a matched link once, regardless of its detector count.

    Tooltip category totals are raw simulated vehicle passages. For the chart,
    expand sampled cars/trucks; scheduled buses are not population-sampled.
    Bikes are displayed separately, not added to motor-vehicle traffic.
    """
    if not np.isfinite(sample_size) or not 0 < sample_size <= 1:
        raise ValueError("input_downsampling must be in (0, 1].")
    ids = list(dict.fromkeys(map(str, link_ids)))
    if not ids or any(link_id not in statistics for link_id in ids):
        return {}, None
    hourly = sum((statistics[link_id] for link_id in ids[1:]), statistics[ids[0]].copy())
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
