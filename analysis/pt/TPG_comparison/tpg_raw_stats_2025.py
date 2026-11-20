import argparse
import datetime
import os
import sys

import pandas as pd
from pandas.api.types import union_categoricals

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def configure(context):
    context.config("data_path")

    context.config("analysis.pt.tpg_data",                  default = "Geneva_passenger_counts/TPG")
    context.config("analysis.pt.tpg_raw_counts_path",       default = "Geneva_passenger_counts/TPG/raw/2025/raw_transit_data_2025.csv")
    context.config("analysis.pt.tpg_processed_counts_path", default = "Geneva_passenger_counts/TPG/processed")

def execute(context):
    data_path = context.config("data_path")
    out_dir   = os.path.join(data_path, context.config("analysis.pt.tpg_processed_counts_path"))

    result, by_direction = aggregate_workday_stats(
        os.path.join(data_path, context.config("analysis.pt.tpg_raw_counts_path")),
        os.path.join(data_path, context.config("analysis.pt.tpg_data")),
    )

    # the comparison reads the latest tpg2025_agg_workday_<date>.csv of the folder, the line comparison the latest
    # tpg2025_agg_workday_<date>_by_direction.csv
    path = os.path.join(out_dir, f"tpg2025_agg_workday_{datetime.date.today():%Y%m%d}.csv")
    result.to_csv(path, index = False)
    by_direction.to_csv(path.replace(".csv", "_by_direction.csv"), index = False)
    print(f"Wrote {path} ({result.shape[0]} rows) and its _by_direction version ({by_direction.shape[0]} rows)")

    return dict(done = True, path = path)


def aggregate_workday_stats(raw_counts_path, tpg_data_path,
                             bank_holidays = _BANK_HOLIDAYS_2025, school_holidays = ()):
    """Returns (stats per stop, line and hour, the same per stop, line, direction and hour). The raw counts have no
    direction: the direction of a run is the station (first 4 letters of CodeLong) of its last stop, and line_direction
    is "<line>_<station>" in the second table (used by the line comparison, analysis.pt.TPG_comparison.line_comparison)."""

    print(f"Reading raw TPG counts from {raw_counts_path} (this is large, ~4.85GB)...")
    raw = _read_raw_counts(raw_counts_path)
    print(f"  -> {len(raw)} rows")

    print("Direction of each run: the station of its last stop (all the rows of the run, before the FlagOk filter)...")
    last = raw.loc[raw.groupby(["DExploitCourse", "IdCourse"], observed = True)["RangArretAsc"].idxmax(),
                   ["DExploitCourse", "IdCourse", "CodeLong"]]
    last["direction"] = last["CodeLong"].astype(str).str[:4].astype("category")

    print("Expansion of the counted runs to all the runs of their line, day and scheduled departure hour...")
    expansion = _run_expansion_factors(raw)
    runs = last[["DExploitCourse", "IdCourse", "direction"]].merge(expansion, on = ["DExploitCourse", "IdCourse"], how = "left")
    del last, expansion

    raw = raw.merge(runs, on = ["DExploitCourse", "IdCourse"], how = "left")
    raw = raw.drop(columns = ["IdCourse", "RangArretAsc", "HDepartTheo"])

    print("Filtering to FlagOk == 'Y'...")
    flag_counts = raw["FlagOk"].value_counts()
    print(f"  FlagOk counts: {flag_counts.to_dict()}")
    raw = raw[raw["FlagOk"] == "Y"]
    print(f"  -> {len(raw)} rows with FlagOk == 'Y'")

    # the runs that are not FlagOk carry counts flagged as unreliable: the valid runs stand in for them
    for column, _ in _COUNT_COLUMNS:
        raw[column] = raw[column] * raw["expansion"].astype("float32")
    raw = raw.drop(columns = ["expansion"])

    print("Keeping the rows with a valid actual arrival timestamp (hour derived while reading)...")
    raw = raw[raw["hour"].notna()]
    raw["hour"] = raw["hour"].astype("int16")
    print(f"  -> {len(raw)} rows with a valid arrival timestamp")

    print("Classifying day type per date and restricting to the working days (Monday to Friday)...")
    dates = pd.to_datetime(raw["DExploitCourse"], format = "%Y-%m-%d")
    unique_dates = dates.drop_duplicates()
    day_type_by_date = pd.Series(
        tpg_raw_stats_2024.classify_day_type(unique_dates, bank_holidays = bank_holidays, school_holidays = school_holidays),
        index = unique_dates,
    )
    print(f"  days per type: {day_type_by_date.value_counts().to_dict()}")
    raw = raw[pd.Series(dates.map(day_type_by_date).values).isin(tpg_raw_stats_2024.WORKDAY_TYPES).values]
    print(f"  -> {len(raw)} rows on working days")

    # Days each line was counted (any FlagOk row, before the stop crosswalk): a stop/line/hour without
    # passengers on one of these days counts as a zero in the daily statistics, see _aggregate
    line_days = raw[["Ligne", "DExploitCourse"]].drop_duplicates()
    line_days = line_days.rename(columns = {"Ligne": "line", "DExploitCourse": "_date"})
    line_days["line"] = line_days["line"].astype(str)
    print(f"  {len(line_days)} line/day pairs with counts, {line_days['line'].nunique()} lines")

    print("Mapping stops to GTFS ids...")
    crosswalk = tpg_raw_stats_2024.load_stop_crosswalk(tpg_data_path)
    raw = raw.merge(crosswalk, left_on = "CodeLong", right_on = "stop_code", how = "inner")
    print(f"  -> {len(raw)} rows with a matched GTFS stop")

    raw = raw.rename(columns = {"Ligne": "line_direction"})  # bare line number, no direction - see docstring point 3
    raw["line_direction"] = raw["line_direction"].astype(str)
    raw["line_with_direction"] = raw["line_direction"] + "_" + raw["direction"].astype("object").fillna("").astype(str)
    raw["line"] = raw["line_direction"]  # the bare line, to look up the days it was counted

    return _aggregate(raw, "line_direction", line_days), _aggregate(raw, "line_with_direction", line_days)


def _aggregate(raw, line_column, line_days):
    raw = raw.drop(columns = [c for c in ("line_direction", "line_with_direction") if c != line_column])
    raw = raw.rename(columns = {line_column: "line_direction"})

    print("Summing to one total per stop/line/hour/day (hour is real-arrival-time here, so a single "
          "day can have several distinct bus/tram passes landing in the same hour - see docstring point 5b)...")
    count_columns = [c for c, _ in _COUNT_COLUMNS]
    raw["_date"] = raw["DExploitCourse"]
    keys  = ["gtfs_code", "line_direction", "hour"]
    daily = raw.groupby(keys + ["line", "_date"], observed = True)[count_columns].sum().reset_index()
    print(f"  -> {len(daily)} stop/line/hour/day rows (from {len(raw)} individual stop-visit rows)")

    daily = _fill_missing_line_days(daily, keys, line_days, count_columns)

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


def _read_raw_counts(raw_counts_path, chunksize = 5_000_000):
    """Reads the raw counts in chunks to limit the memory: the arrival timestamp (one string per row) is reduced to
    its hour (NaN if invalid) and the text columns are stored as categories."""
    categorical = ["DExploitCourse", "Ligne", "CodeLong", "FlagOk", "HDepartTheo"]

    chunks = []
    reader = pd.read_csv(
        raw_counts_path, chunksize = chunksize,
        usecols = ["IdCourse", "RangArretAsc", "NbMontees", "NbDescentes", "DTEntreeFenetreArretReal"] + categorical,
        dtype = {
            **{column: "category" for column in categorical},
            "NbMontees": "float32", "NbDescentes": "float32", "RangArretAsc": "int16",
        },
    )
    for chunk in reader:
        arrival = pd.to_datetime(chunk["DTEntreeFenetreArretReal"], errors = "coerce")
        chunk["hour"] = arrival.dt.hour.astype("float32")
        chunks.append(chunk.drop(columns = ["DTEntreeFenetreArretReal"]))
        print(f"  {sum(len(c) for c in chunks)} rows read", flush = True)

    # the categories differ between chunks: union them, a plain concat would turn the columns into strings
    raw = pd.concat([c.drop(columns = categorical) for c in chunks], ignore_index = True)
    for column in categorical:
        raw[column] = union_categoricals([c[column] for c in chunks])
        for c in chunks:
            c.drop(columns = [column], inplace = True)

    return raw


def _run_expansion_factors(raw):
    """Expansion factor of each FlagOk run: number of runs / number of FlagOk runs of its line, operating day and
    scheduled departure hour (HDepartTheo, the departure of the run). FlagOk is set per run. The runs of a group
    without any FlagOk run cannot be represented: their share is printed."""
    runs = raw.drop_duplicates(["DExploitCourse", "IdCourse"])[["DExploitCourse", "IdCourse", "Ligne", "FlagOk", "HDepartTheo"]].copy()
    runs["ok"]           = runs["FlagOk"] == "Y"
    runs["depart_hour"]  = pd.to_numeric(runs["HDepartTheo"].astype(str).str.split(":").str[0], errors = "coerce").fillna(-1)

    group = ["Ligne", "DExploitCourse", "depart_hour"]
    per_group = runs.groupby(group, observed = True).agg(n_runs = ("ok", "size"), n_ok = ("ok", "sum")).reset_index()
    per_group["expansion"] = per_group["n_runs"] / per_group["n_ok"].where(per_group["n_ok"] > 0)

    runs = runs.merge(per_group[group + ["expansion"]], on = group, how = "left")

    n_ok         = runs["ok"].sum()
    uncovered    = per_group.loc[per_group["n_ok"] == 0, "n_runs"].sum()
    print(f"  {len(runs)} runs, {n_ok} FlagOk ({100 * n_ok / len(runs):.1f}%); mean factor of the FlagOk runs "
          f"{runs.loc[runs['ok'], 'expansion'].mean():.3f}")
    print(f"  {uncovered} runs ({100 * uncovered / len(runs):.1f}%) are in a line/day/hour without any FlagOk run "
          f"and are not represented")

    return runs.loc[runs["ok"], ["DExploitCourse", "IdCourse", "expansion"]]


def _fill_missing_line_days(daily, keys, line_days, count_columns):
    """Adds a zero row for every day on which the line was counted but this stop/line(/direction)/hour has no
    passengers, so that the statistics across days are per counted day of the line and not only per day on which
    the stop/line/hour appears. Without it, rare patterns (short runs, diversions, late runs landing in another
    hour) count as if they ran every day, which inflates the summed means."""
    mean_before = daily.groupby(keys, observed = True)[count_columns].mean().sum()

    combinations = daily[keys + ["line"]].drop_duplicates()
    full         = combinations.merge(line_days, on = "line")

    filled = full.merge(daily, on = keys + ["line", "_date"], how = "left")
    filled[count_columns] = filled[count_columns].fillna(0).astype(daily[count_columns].dtypes)

    mean_after = filled.groupby(keys, observed = True)[count_columns].mean().sum()

    print(f"  -> {len(filled)} stop/line/hour/day rows after adding the zeros of the days the line was counted "
          f"({len(filled) - len(daily)} added)")
    for column in count_columns:
        print(f"  {column}: sum of the daily means {mean_before[column]:,.0f} -> {mean_after[column]:,.0f}")

    return filled


def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-counts-path", required = True, help = "Path to TPG's raw 2025 passenger counts CSV")
    parser.add_argument("--tpg-data-path", required = True, help = "Path to the Geneva_passenger_counts/TPG folder (crosswalk files in TPG_stops_info)")
    parser.add_argument(
        "--out", required = True,
        help = "Where to write the rebuilt tpg2025_agg_workday_*.csv - drop it into config.Config.processed_counts_path",
    )
    args = parser.parse_args()

    result, by_direction = aggregate_workday_stats(args.raw_counts_path, args.tpg_data_path)
    print(result.shape)
    print(result.head())

    if args.out:
        result.to_csv(args.out, index = False)
        by_direction.to_csv(args.out.replace(".csv", "_by_direction.csv"), index = False)
        print(f"Wrote {args.out} and its _by_direction version")


if __name__ == "__main__":
    main()
