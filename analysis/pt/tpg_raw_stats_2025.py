"""
Same idea as tpg_raw_stats.py (min/max/mean/std/q10/q20/q50/q80/q90 of
boardings/alightings, per stop/line/hour, on "Weekday"-type days), but for
the 2025 raw dataset instead of the 2024 one. The two source files have a
genuinely different schema, so this is a separate script rather than a
parameter change to tpg_raw_stats.py - please read the differences below
before trusting the output.

Source columns used: IdCourse, DExploitCourse (date), Ligne, CodeLong (TPG
stop code), NbMontees, NbDescentes, FlagOk, DTEntreeFenetreArretReal (actual
arrival timestamp at that stop). On the order of tens of millions of rows,
several GB - expect several minutes and several GB of memory. Not run as
a pipeline stage; run separately via:

    python tpg_raw_stats_2025.py --raw-counts-path ... --tpg-data-path ... --out PATH

KNOWN DIFFERENCES FROM THE 2024 SCRIPT / OUTPUT, READ BEFORE USING:

1. Quality filter: this file has an explicit FlagOk column (Y/E/D observed
   in a sample). Only FlagOk == "Y" rows are kept; E/D counts are printed
   so you can judge whether that's dropping a meaningful share of the data.
   The 2024 source has no equivalent flag.

2. Hour: there is no per-stop theoretical-hour-bin column like 2024's
   "Horaire Tranche Stop Théo." - HDepartTheo is constant per IdCourse (the
   course's scheduled departure time from its origin, not this stop), so it
   doesn't identify which hour a given stop event happened in. Instead,
   `hour` here is taken from DTEntreeFenetreArretReal (the actual arrival
   timestamp at that specific stop). This is a real-time hour, not a
   theoretical/scheduled one - a methodologically different quantity from
   the 2024 file's, even though the output column is named the same.

3. No direction: the 2024 source has a "Ligne-Sens" field (line + Aller/
   Retour), letting tpg_raw_stats.py build "line_direction" values like
   "15_H"/"15_R". This 2025 source has no direction field at all - RangArretAsc
   (stop sequence within a course) could in principle be used to infer
   direction by clustering courses' first/last stops per line, but that is
   real extra work this script does NOT do. `line_direction` here is just
   the bare line number (e.g. "15"), with no H/R suffix. This means the
   output does NOT plug into tpg_data.match_line_directions /
   stages._build_tpg_mofr as-is - those need the H/R suffix to join against
   MATSim's line_main_direction. Treat this as the raw-to-stats aggregation
   step only, not a full drop-in replacement for the 2024 comparison
   pipeline, until direction is derived some other way.

4. Only one count variant: the 2024 file has both a raw/observed count
   ("... Brut") and TPG's own adjusted/expanded estimate: two different
   numbers. This 2025 file has just NbMontees/NbDescentes - one number. To
   keep the same 40-column output schema as the 2024 script's output (so
   this file can be dropped into TPG_processed_counts/ and picked up the
   same way via stages._resolve_tpg_stats_path), the "_raw_" and
   non-"_raw_" columns are filled with the SAME values here - they are not
   actually two independent measurements for 2025.

5b. Real-arrival hour means several distinct stop-visits (courses) can land
   in the same (stop, line, hour) bucket on a single day, unlike 2024's
   theoretical hour bins where that's rare. mean/std/quantiles are computed
   ACROSS DAYS on the day's summed total for that bucket (one number per
   day), not across individual stop-visit rows - averaging across raw rows
   directly would silently divide the true per-day total by however many
   passes happened to land in that hour, undercounting by ~4-8x in
   practice. If you are re-deriving this file or writing something similar,
   always sum to one row per (stop, line, hour, day) first.

6. Day-type calendar: classify_day_type's bank-holiday dates are
   recomputed for 2025 (Good Friday/Easter Monday/Ascension/Whit Monday
   shift every year; Jeûne genevois is the Thursday after the first Sunday
   of September). The SCHOOL HOLIDAY dates below are a best-effort estimate
   from the general Geneva DIP school calendar pattern and are NOT verified
   against an official source - if the exact "School holiday" vs "Weekday"
   split matters to you, please confirm/correct _SCHOOL_HOLIDAYS_2025 below
   (or pass your own via aggregate_workday_stats's school_holidays arg)
   before trusting results that hinge on it. Only "Weekday"-type rows are
   kept in the output either way, so this mostly affects which of two
   plausible sets of dates get excluded as "not a plain weekday".
"""

import argparse

import pandas as pd

import tpg_raw_stats

_BANK_HOLIDAYS_2025 = pd.to_datetime([
    "2025-01-01",  # New Year
    "2025-04-18",  # Good Friday
    "2025-04-21",  # Easter Monday
    "2025-05-01",  # Labour Day
    "2025-05-29",  # Ascension
    "2025-06-09",  # Whit Monday
    "2025-08-01",  # Swiss National Day
    "2025-09-11",  # Jeune genevois (Thursday after the first Sunday of September)
    "2025-12-25",  # Christmas
    "2025-12-31",  # Restauration de la Republique (Geneva)
])

# Best-effort / UNVERIFIED - see point 5 in the module docstring.
_SCHOOL_HOLIDAYS_2025 = [
    (pd.to_datetime(start), pd.to_datetime(end))
    for start, end in [
        ("2025-01-01", "2025-01-05"),  # Christmas 2024/25
        ("2025-02-24", "2025-03-02"),  # Winter break
        ("2025-04-18", "2025-05-04"),  # Easter / spring break
        ("2025-06-30", "2025-08-17"),  # Summer break
        ("2025-10-20", "2025-10-26"),  # Autumn break
        ("2025-12-22", "2025-12-31"),  # Christmas 2025
    ]
]

_COUNT_COLUMNS = [("NbMontees", "boardings"), ("NbDescentes", "alightings")]


def aggregate_workday_stats(raw_counts_path, tpg_data_path,
                             bank_holidays = _BANK_HOLIDAYS_2025, school_holidays = _SCHOOL_HOLIDAYS_2025):
    """
    Returns a DataFrame with the same 40-column schema as
    tpg_counts_agg_workdays.csv, computed from the 2025 raw counts - see
    the module docstring for how this differs from the 2024 version
    (no direction, single count variant duplicated into the raw/non-raw
    columns, hour from actual arrival time not a theoretical bin).
    """

    print(f"Reading raw TPG counts from {raw_counts_path} (this is large, ~4.85GB)...")
    raw = pd.read_csv(
        raw_counts_path,
        usecols = ["DExploitCourse", "Ligne", "CodeLong", "NbMontees", "NbDescentes",
                  "FlagOk", "DTEntreeFenetreArretReal"],
        dtype = {
            "Ligne": "category", "CodeLong": "category", "FlagOk": "category",
            "NbMontees": "float32", "NbDescentes": "float32",
        },
    )
    print(f"  -> {len(raw)} rows")

    print("Filtering to FlagOk == 'Y'...")
    flag_counts = raw["FlagOk"].value_counts()
    print(f"  FlagOk counts: {flag_counts.to_dict()}")
    raw = raw[raw["FlagOk"] == "Y"]
    print(f"  -> {len(raw)} rows with FlagOk == 'Y'")

    print("Deriving hour from the actual arrival timestamp...")
    arrival = pd.to_datetime(raw["DTEntreeFenetreArretReal"], errors = "coerce")
    raw = raw[arrival.notna()]
    raw["hour"] = arrival[arrival.notna()].dt.hour.astype("int16")
    print(f"  -> {len(raw)} rows with a valid arrival timestamp")

    print("Classifying day type per date and restricting to Weekdays...")
    dates = pd.to_datetime(raw["DExploitCourse"], format = "%Y-%m-%d")
    unique_dates = dates.drop_duplicates()
    day_type_by_date = pd.Series(
        tpg_raw_stats.classify_day_type(unique_dates, bank_holidays = bank_holidays, school_holidays = school_holidays),
        index = unique_dates,
    )
    raw = raw[dates.map(day_type_by_date).values == "Weekday"]
    print(f"  -> {len(raw)} rows on Weekday-type days")

    print("Mapping stops to GTFS ids...")
    crosswalk = tpg_raw_stats.load_stop_crosswalk(tpg_data_path)
    raw = raw.merge(crosswalk, left_on = "CodeLong", right_on = "stop_code", how = "inner")
    print(f"  -> {len(raw)} rows with a matched GTFS stop")

    raw = raw.rename(columns = {"Ligne": "line_direction"})  # bare line number, no direction - see docstring point 3
    raw["line_direction"] = raw["line_direction"].astype(str)

    print("Summing to one total per stop/line/hour/day (hour is real-arrival-time here, so a single "
          "day can have several distinct bus/tram passes landing in the same hour - see docstring point 5b)...")
    count_columns = [c for c, _ in _COUNT_COLUMNS]
    raw["_date"] = raw["DExploitCourse"]
    daily = raw.groupby(["gtfs_code", "line_direction", "hour", "_date"], observed = True)[count_columns].sum().reset_index()
    print(f"  -> {len(daily)} stop/line/hour/day rows (from {len(raw)} individual stop-visit rows)")

    print("Aggregating (min/max/mean/std/q10/q20/q50/q80/q90) across days, per stop/line/hour...")
    quantile_labels = ["q10", "q20", "q50", "q80", "q90"]

    def make_quantile_fn(q, name):
        fn = lambda s: s.quantile(q)
        fn.__name__ = name
        return fn

    agg_funcs = ["min", "max", "mean", "std"] + [
        make_quantile_fn(q, name) for q, name in zip([0.1, 0.2, 0.5, 0.8, 0.9], quantile_labels)
    ]

    grouped = daily.groupby(["gtfs_code", "line_direction", "hour"], observed = True)[count_columns].agg(agg_funcs)
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    out = grouped[["gtfs_code", "line_direction", "hour"]].copy()
    out.insert(0, "day_type", "Weekday")

    stats = ["min", "max", "mean", "std"] + quantile_labels
    for source_col, prefix in _COUNT_COLUMNS:
        for stat in stats:
            value = grouped[f"{source_col}_{stat}"]
            out[f"{prefix}_raw_{stat}"] = value  # only one count variant available - see docstring point 4
            out[f"{prefix}_{stat}"]     = value

    return out


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-counts-path", required = True, help = "Path to TPG's raw 2025 passenger counts CSV")
    parser.add_argument("--tpg-data-path", required = True, help = "Path to the TPG_passenger_counts folder (crosswalk files)")
    parser.add_argument(
        "--out", required = True,
        help = "Where to write the rebuilt tpg2025_agg_workday_*.csv - drop it into config.Config.tpg_processed_counts_path",
    )
    args = parser.parse_args()

    result = aggregate_workday_stats(args.raw_counts_path, args.tpg_data_path)
    print(result.shape)
    print(result.head())

    if args.out:
        result.to_csv(args.out, index = False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
