"""
Rebuilds tpg_counts_agg_workdays.csv from TPG's raw daily passenger counts.

This reproduces a day-type classification, then a groupby/agg into
min/max/mean/p10/p20/p50/p80/p90, to match the schema consumed by
comparison_passenger_counts_geneva.py / stages._build_tpg_mofr (via
config.Config.tpg_processed_counts_path) and used for the TPG confidence
interval in comparison.py: std is included alongside the percentiles
(comparison.py's CI is mean +/- 1.96*std), and both count variants present
in the raw file are aggregated - "Brut" (raw, observed counts -> the
*_raw_* columns) and the plain one (TPG's own corrected/expanded estimate
-> the columns without _raw_).

Source data: one row per (date, line+direction, theoretical hour slot,
stop), covering a full year. This is on the order of tens of millions of
rows / ~1GB - expect this to take a few minutes and a few GB of memory. It
is NOT run as a pipeline stage; run it separately (e.g. once, to
regenerate/verify the aggregated file) via:

    python tpg_raw_stats.py --raw-counts-path ... --tpg-data-path ... --out PATH
"""

import argparse

import numpy as np
import pandas as pd

# Same calendar as TPGstops.ipynb's classify_day() (hardcoded for 2024).
_BANK_HOLIDAYS_2024 = pd.to_datetime([
    "01.01.2024",
    "29.03.2024",
    "01.04.2024",
    "01.05.2024",
    "09.05.2024",
    "20.05.2024",
    "01.08.2024",
    "05.09.2024",
    "25.12.2024",
    "31.12.2024",
], format = "%d.%m.%Y")

_SCHOOL_HOLIDAYS_2024 = [
    (pd.to_datetime(start, format = "%d.%m.%Y"), pd.to_datetime(end, format = "%d.%m.%Y"))
    for start, end in [
        ("01.01.2024", "07.01.2024"),  # Christmas 2023
        ("19.02.2024", "23.02.2024"),  # Winter break
        ("29.03.2024", "14.04.2024"),  # Easter break
        ("09.05.2024", "12.05.2024"),  # Ascension
        ("29.06.2024", "18.08.2024"),  # Summer break
        ("21.10.2024", "27.10.2024"),  # Autumn break
        ("23.12.2024", "31.12.2024"),  # Christmas 2024
    ]
]

_DIRECTION_LETTER = {"Aller": "H", "Retour": "R"}

# (raw_column, adjusted_column, output_prefix)
_COUNT_COLUMNS = [
    ("Nb Montées Brut", "Nb Montées", "boardings"),
    ("Nb Descentes Brut", "Nb Descentes", "alightings"),
]


def classify_day_type(dates, bank_holidays = _BANK_HOLIDAYS_2024, school_holidays = _SCHOOL_HOLIDAYS_2024):
    """
    Vectorized version of TPGstops.ipynb's classify_day(): for each date,
    priority (highest wins) is Sunday > Saturday > Bank holiday >
    School holiday > Weekday. `bank_holidays`/`school_holidays` default to
    the 2024 calendar above; pass a different year's calendar to reuse this
    for another year (see tpg_raw_stats_2025.py).
    """

    dates   = pd.DatetimeIndex(dates)
    weekday = dates.weekday

    day_type = pd.Series(np.where(weekday <= 4, "Weekday", None), index = range(len(dates)))

    is_school_holiday = np.zeros(len(dates), dtype = bool)
    for start, end in school_holidays:
        is_school_holiday |= (dates >= start) & (dates <= end)
    day_type[is_school_holiday] = "School holiday"

    day_type[dates.isin(bank_holidays)] = "Bank holiday"
    day_type[weekday == 5] = "Saturday"
    day_type[weekday == 6] = "Sunday"

    return day_type.values


def load_stop_crosswalk(tpg_data_path):
    """TPG stop_code ("Arrêt Code Long" in the raw file) -> GTFS stop id."""

    df = pd.read_csv(f"{tpg_data_path}/TPG_stops_info/tpg_Arrets.csv", encoding = "latin1", sep = ";")
    df.columns = ["stop_code", "lon", "lat", "country", "name", "municipality", "gtfs_code", "date1", "date2"]
    df = df[["stop_code", "gtfs_code"]]
    df = df[df["gtfs_code"].notna()]
    df["gtfs_code"] = df["gtfs_code"].astype(int).astype(str)

    return df


def parse_line_direction(ligne_sens):
    """"15 - Retour" -> "15_R", "15 - Aller" -> "15_H" (same H/R convention as tpg_data.match_line_directions)."""

    parts = ligne_sens.astype(str).str.partition(" - ")
    line, direction = parts[0], parts[2]
    return line.str.strip() + "_" + direction.str.strip().map(_DIRECTION_LETTER)


def aggregate_workday_stats(raw_counts_path, tpg_data_path):
    """
    Returns a DataFrame with the same schema as tpg_counts_agg_workdays.csv:
    one row per (gtfs_code, line_direction, hour) on "Weekday"-type days in
    2024, with min/max/mean/std/q10/q20/q50/q80/q90 of boardings and
    alightings, in both their raw and TPG-adjusted variants.
    """

    print(f"Reading raw TPG counts from {raw_counts_path} (this is large, ~1GB)...")
    raw = pd.read_csv(
        raw_counts_path, sep = ";",
        usecols = ["Date", "Ligne-Sens", "Horaire Tranche Stop Théo.", "Arrêt Code Long",
                  "Nb Montées", "Nb Montées Brut", "Nb Descentes", "Nb Descentes Brut"],
        dtype = {
            "Ligne-Sens": "category", "Arrêt Code Long": "category",
            "Horaire Tranche Stop Théo.": "float32",  # a few rows have a missing hour bin
            "Nb Montées": "float32", "Nb Montées Brut": "float32",
            "Nb Descentes": "float32", "Nb Descentes Brut": "float32",
        },
    )
    print(f"  -> {len(raw)} rows")

    raw = raw[raw["Horaire Tranche Stop Théo."].notna()]
    raw["Horaire Tranche Stop Théo."] = raw["Horaire Tranche Stop Théo."].astype("int16")
    print(f"  -> {len(raw)} rows with a valid hour bin")

    print("Classifying day type per date and restricting to Weekdays...")
    dates = pd.to_datetime(raw["Date"], format = "%d.%m.%Y")
    unique_dates = dates.drop_duplicates()
    day_type_by_date = pd.Series(classify_day_type(unique_dates), index = unique_dates)
    raw = raw[dates.map(day_type_by_date).values == "Weekday"]
    print(f"  -> {len(raw)} rows on Weekday-type days")

    print("Mapping stops to GTFS ids...")
    crosswalk = load_stop_crosswalk(tpg_data_path)
    raw = raw.merge(crosswalk, left_on = "Arrêt Code Long", right_on = "stop_code", how = "inner")
    print(f"  -> {len(raw)} rows with a matched GTFS stop")

    print("Parsing line/direction...")
    raw["line_direction"] = parse_line_direction(raw["Ligne-Sens"])
    raw = raw.rename(columns = {"Horaire Tranche Stop Théo.": "hour"})

    print("Aggregating (min/max/mean/std/q10/q20/q50/q80/q90) per stop/line/hour...")
    quantile_labels = ["q10", "q20", "q50", "q80", "q90"]

    def make_quantile_fn(q, name):
        fn = lambda s: s.quantile(q)
        fn.__name__ = name
        return fn

    agg_funcs = ["min", "max", "mean", "std"] + [
        make_quantile_fn(q, name) for q, name in zip([0.1, 0.2, 0.5, 0.8, 0.9], quantile_labels)
    ]

    count_columns = [c for pair in _COUNT_COLUMNS for c in pair[:2]]
    grouped = raw.groupby(["gtfs_code", "line_direction", "hour"], observed = True)[count_columns].agg(agg_funcs)
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    out = grouped[["gtfs_code", "line_direction", "hour"]].copy()
    out.insert(0, "day_type", "Weekday")

    stats = ["min", "max", "mean", "std"] + quantile_labels
    for raw_col, adjusted_col, prefix in _COUNT_COLUMNS:
        for stat in stats:
            out[f"{prefix}_raw_{stat}"] = grouped[f"{raw_col}_{stat}"]
            out[f"{prefix}_{stat}"]     = grouped[f"{adjusted_col}_{stat}"]

    return out


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-counts-path", required = True, help = "Path to TPG's raw 2024 daily passenger counts (data.txt)")
    parser.add_argument("--tpg-data-path", required = True, help = "Path to the TPG_passenger_counts folder (crosswalk files)")
    parser.add_argument("--out", required = True, help = "Where to write the rebuilt tpg2024_agg_workday_*.csv - drop it into config.Config.tpg_processed_counts_path")
    args = parser.parse_args()

    result = aggregate_workday_stats(args.raw_counts_path, args.tpg_data_path)
    print(result.shape)
    print(result.head())

    if args.out:
        result.to_csv(args.out, index = False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
