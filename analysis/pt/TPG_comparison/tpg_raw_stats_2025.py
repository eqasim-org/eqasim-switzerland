import argparse
import pandas as pd
import tpg_raw_stats_2024

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
        tpg_raw_stats_2024.classify_day_type(unique_dates, bank_holidays = bank_holidays, school_holidays = school_holidays),
        index = unique_dates,
    )
    raw = raw[dates.map(day_type_by_date).values == "Weekday"]
    print(f"  -> {len(raw)} rows on Weekday-type days")

    print("Mapping stops to GTFS ids...")
    crosswalk = tpg_raw_stats_2024.load_stop_crosswalk(tpg_data_path)
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
