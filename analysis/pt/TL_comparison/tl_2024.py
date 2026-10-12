import datetime
import glob
import os
import re
import unicodedata
import zipfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

WEEKDAY_TYPE = "SU"

_USECOLS = ["date", "type_horaire", "ligne", "sens", "id_parcours_type", "code_arret", "libelle_arret",
            "heure_arrivee_arret", "monte", "descendu"]

_DIRECTION_LETTER = {"A": "H", "R": "R"}

_COUNT_COLUMNS = [("monte", "boardings"), ("descendu", "alightings")]
_STATS = ["min", "max", "mean", "std", "q10", "q20", "q50", "q80", "q90"]
_QUANTILES = {"q10": 0.1, "q20": 0.2, "q50": 0.5, "q80": 0.8, "q90": 0.9}


def _normalize_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def _read_file(path):
    df = pd.read_csv(
        path, sep = ";", encoding = "utf-8-sig", decimal = ",", usecols = _USECOLS,
        dtype = {"type_horaire": "category", "sens": "category", "code_arret": "category", "libelle_arret": "category"},
    )

    df = df[(df["type_horaire"] == WEEKDAY_TYPE) & df["sens"].notna() & df["monte"].notna() & df["descendu"].notna()]
    df = df.assign(
        hour = df["heure_arrivee_arret"].astype(str).str.slice(11, 13).astype(int) % 24,
        sens = df["sens"].astype(str), code_arret = df["code_arret"].astype(str), libelle_arret = df["libelle_arret"].astype(str),
    )

    daily = df.groupby(["date", "ligne", "sens", "code_arret", "libelle_arret", "hour"], as_index = False)[["monte", "descendu"]].sum()

    # the rows of a run are in the order of its stops
    runs = df.groupby(["ligne", "sens", "id_parcours_type"], observed = True).agg(
        first_stop = ("libelle_arret", "first"), last_stop = ("libelle_arret", "last"), visits = ("date", "size"),
    ).reset_index().drop(columns = "id_parcours_type")

    return os.path.basename(path), df["date"].unique().tolist(), daily, runs


def read_daily_counts(results_dir, workers = 8):
    paths = sorted(p for p in glob.glob(os.path.join(results_dir, "*_frequentation_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_frequentation_*.csv file in {results_dir}")

    print(f"Reading {len(paths)} TL frequentation files from {results_dir} ...", flush = True)

    dailies, runs, dates_by_file = [], [], {}

    with ProcessPoolExecutor(max_workers = workers) as executor:
        for i, (name, dates, daily, run) in enumerate(executor.map(_read_file, paths), start = 1):
            dailies.append(daily)
            runs.append(run)
            dates_by_file[name] = (daily["ligne"].unique().tolist(), dates)

            if i % 10 == 0 or i == len(paths):
                print(f"  -> {i}/{len(paths)} files read, {len(paths) - i} remaining", flush = True)

    daily = pd.concat(dailies, ignore_index = True)

    # a date must be in one file only per line, otherwise its counts would be summed twice
    files_per_date = pd.Series(
        [(line, date) for _, (lines, dates) in dates_by_file.items() for line in lines for date in dates]
    ).value_counts()
    if (files_per_date > 1).any():
        print(f"WARNING: {int((files_per_date > 1).sum())} (line, date) are in more than one file")

    print(f"  -> {len(daily)} date/line/direction/stop/hour rows, {daily['date'].nunique()} school-term weekdays, "
          f"{daily['ligne'].nunique()} lines")

    return daily, pd.concat(runs, ignore_index = True)


def build_directions(runs):
    runs = runs.groupby(["ligne", "sens", "first_stop", "last_stop"], as_index = False)["visits"].sum()
    runs = runs.sort_values("visits", ascending = False).drop_duplicates(["ligne", "sens"])

    out = runs.rename(columns = {"ligne": "line", "sens": "tl_direction"}).drop(columns = "visits")
    out["direction"]          = out["tl_direction"].map(_DIRECTION_LETTER)
    out["line_direction"]     = out["line"].astype(str) + "_" + out["direction"]
    out["direction_extended"] = out["first_stop"] + "->" + out["last_stop"]

    return out.sort_values("line_direction").reset_index(drop = True)[
        ["line_direction", "line", "direction", "tl_direction", "first_stop", "last_stop", "direction_extended"]
    ]


# The TL stops of the other communes of the network are named "Commune, Stop" in the GTFS; the area of the
# network (lat/lon) restricts the matching of the names without commune.
_NETWORK_AREA = {"lat": (46.45, 46.62), "lon": (6.50, 6.75)}


# A stop must be served on at least this share of the weekdays of its line direction. The stops of a temporary
# diversion are only counted on a few days: averaged over these days alone, they would look like regular stops.
MIN_DAY_SHARE = 0.5


def drop_occasional_stops(daily, min_day_share = MIN_DAY_SHARE):
    line_days = daily.groupby("line_direction")["date"].nunique()
    stop_days = daily.groupby(["line_direction", "gtfs_code"])["date"].nunique().rename("stop_days").reset_index()
    stop_days["share"] = stop_days["stop_days"] / stop_days["line_direction"].map(line_days)

    occasional = stop_days[stop_days["share"] < min_day_share]
    for row in occasional.sort_values("stop_days", ascending = False).head(20).itertuples():
        print(f"  occasional stop dropped: {row.line_direction} at {row.gtfs_code} ({row.stop_days}/{line_days[row.line_direction]} days)")
    print(f"Dropping {len(occasional)} (line direction, stop) served on less than {min_day_share:.0%} of the weekdays of the line")

    keep = stop_days.loc[stop_days["share"] >= min_day_share, ["line_direction", "gtfs_code"]]
    return daily.merge(keep, on = ["line_direction", "gtfs_code"], how = "inner")


def fill_missing_days(per_day, line_dates, count_columns):
    """A stop/hour without any visit on a day of its line direction counts as 0 on that day.

    Otherwise the statistics are computed over the days with visits only: an hour served on a single night would
    be given the counts of that night as its mean.
    """
    filled = []
    for line_direction, rows in per_day.groupby("line_direction", sort = False):
        keys  = rows[["gtfs_code", "hour"]].drop_duplicates()
        dates = pd.DataFrame({"date": line_dates[line_direction]})
        full  = keys.merge(dates, how = "cross").assign(line_direction = line_direction)
        filled.append(full.merge(rows, on = ["gtfs_code", "line_direction", "hour", "date"], how = "left"))

    filled = pd.concat(filled, ignore_index = True)
    filled[count_columns] = filled[count_columns].fillna(0.0)

    return filled


def build_stop_crosswalk(gtfs_zip, tl_stops):
    with zipfile.ZipFile(gtfs_zip) as zf:
        slot = next(n for n in zf.namelist() if n.endswith("stops.txt"))
        with zf.open(slot) as f:
            gtfs = pd.read_csv(f, dtype = {"stop_id": str, "stop_name": str}, usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"])

    gtfs["gtfs_code"] = gtfs["stop_id"].str.replace("^Parent", "", regex = True).str.split(":").str[0]
    gtfs["name"]      = gtfs["stop_name"].map(_normalize_name)
    gtfs["tail"]      = gtfs["stop_name"].str.partition(", ")[2].map(_normalize_name)
    gtfs["in_area"]   = gtfs["stop_lat"].between(*_NETWORK_AREA["lat"]) & gtfs["stop_lon"].between(*_NETWORK_AREA["lon"])
    gtfs["has_commune"] = gtfs["stop_name"].str.contains(", ", regex = False)

    def candidates_by(column, frame):
        stops = frame.groupby([column, "gtfs_code"]).agg(n_stops = ("stop_id", "size"), has_commune = ("has_commune", "max")).reset_index()
        stops = stops.sort_values(["has_commune", "n_stops"], ascending = False)
        return {key: group["gtfs_code"].tolist() for key, group in stops.groupby(column)}

    by_name = candidates_by("name", gtfs)
    by_tail = candidates_by("tail", gtfs[gtfs["in_area"] & (gtfs["tail"] != "")])

    stops = tl_stops.drop_duplicates("code_arret").copy()
    codes, status = [], []

    for libelle in stops["libelle_arret"]:
        found = []
        # the GTFS writes the stops of Renens "Renens VD, gare" where the TL has "Renens-Gare" or "Renens, Piscine"
        renens = re.sub(r"^Renens(-|, )", "Renens VD, ", libelle)

        for lookup, key in ((by_name, libelle), (by_name, f"Lausanne, {libelle}"), (by_name, renens), (by_tail, libelle)):
            found = lookup.get(_normalize_name(key), [])
            if found:
                break

        codes.append(found[0] if found else None)
        status.append("unmatched" if not found else ("ambiguous" if len(found) > 1 else "matched"))

    stops["gtfs_code"] = codes
    stops["match"]     = status

    return stops[["code_arret", "libelle_arret", "gtfs_code", "match"]]


def aggregate_workday_stats(results_dir, gtfs_zip, workers = 8):
    daily, runs = read_daily_counts(results_dir, workers)

    print("Mapping stops to GTFS ids...")
    stops = daily[["code_arret", "libelle_arret"]].drop_duplicates()
    crosswalk = build_stop_crosswalk(gtfs_zip, stops)

    movements = daily.groupby("code_arret")[["monte", "descendu"]].sum().sum(axis = 1)
    crosswalk["movements"] = crosswalk["code_arret"].map(movements)
    summary = crosswalk.groupby("match")["movements"].agg(["size", "sum"])
    summary["share"] = summary["sum"] / summary["sum"].sum()
    print(summary.to_string())

    daily = daily.merge(crosswalk[["code_arret", "gtfs_code"]].dropna(), on = "code_arret", how = "inner")
    print(f"  -> {len(daily)} rows with a matched GTFS stop")

    daily["line_direction"] = daily["ligne"].astype(str) + "_" + daily["sens"].map(_DIRECTION_LETTER)

    daily = drop_occasional_stops(daily)
    line_dates = daily.groupby("line_direction")["date"].unique()

    print("Summing the visits of each day per stop/line direction/hour...")
    count_columns = [c for c, _ in _COUNT_COLUMNS]
    per_day = daily.groupby(["gtfs_code", "line_direction", "hour", "date"], as_index = False)[count_columns].sum()
    print(f"  -> {len(per_day)} stop/line direction/hour/day rows")

    per_day = fill_missing_days(per_day, line_dates, count_columns)
    print(f"  -> {len(per_day)} rows once the days without any visit are counted as 0")

    print("Aggregating (min/max/mean/std/q10/q20/q50/q80/q90) across days, per stop/line direction/hour...")

    def quantile_function(q, name):
        fn = lambda s: s.quantile(q)
        fn.__name__ = name
        return fn

    agg_funcs = ["min", "max", "mean", "std"] + [quantile_function(q, name) for name, q in _QUANTILES.items()]

    grouped = per_day.groupby(["gtfs_code", "line_direction", "hour"])[count_columns].agg(agg_funcs)
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()

    out = grouped[["gtfs_code", "line_direction", "hour"]].copy()
    out.insert(0, "day_type", "Weekday")

    for source_col, prefix in _COUNT_COLUMNS:
        for stat in _STATS:
            value = grouped[f"{source_col}_{stat}"]
            out[f"{prefix}_raw_{stat}"] = value  # only one count variant available - see docstring point 5
            out[f"{prefix}_{stat}"]     = value

    return out, build_directions(runs), crosswalk


def write_outputs(result, directions, crosswalk, out_dir):
    os.makedirs(out_dir, exist_ok = True)
    stamp = datetime.date.today().strftime("%Y%m%d")
    base  = os.path.join(out_dir, f"tl2024_agg_workday_{stamp}")

    result.to_csv(f"{base}.csv", index = False)
    directions.to_csv(f"{base}_line_directions.csv", index = False)
    crosswalk.to_csv(f"{base}_stop_crosswalk.csv", index = False)
    crosswalk[crosswalk["match"] != "matched"].sort_values("movements", ascending = False).to_csv(
        f"{base}_unmatched_stops.csv", index = False
    )

    return base


def configure(context):
    context.config("data_path")

    context.config("analysis.pt.tl_path",   default = "Lausanne_passenger_counts/tl data 2024/raw")
    context.config("analysis.pt.tl_processed_path", default = "Lausanne_passenger_counts/tl data 2024/processed")
    context.config("analysis.pt.tl_gtfs_zip",       default = "gtfs/gtfs_fp2024_2024-11-11.zip")
    context.config("analysis.pt.tl_workers",        default = 8)


def execute(context):
    data_path = context.config("data_path")
    out_dir   = os.path.join(data_path, context.config("analysis.pt.tl_processed_path"))

    result, directions, crosswalk = aggregate_workday_stats(
        os.path.join(data_path, context.config("analysis.pt.tl_path")),
        os.path.join(data_path, context.config("analysis.pt.tl_gtfs_zip")),
        context.config("analysis.pt.tl_workers"),
    )

    base = write_outputs(result, directions, crosswalk, out_dir)
    print(f"Wrote {base}.csv and its companion files")

    return dict(done = True, path = out_dir)
