"""MBC (Morges-Bière-Cossonay) passenger counts of the 2024 timetable, in the format of the TL counts (tl_2024).

MBC sends one "diagramme de ligne" per line (H24_<line>.xlsx): one sheet per day type and direction, with, for each
scheduled trip, the boardings and alightings at each stop averaged over the measured runs of the year, its departure time
and the number of measured runs. There are no daily counts, hence no spread between days, only means:
- the working-day mean of a stop, line direction and hour is the mean of the day types weighted by their number of
  weekdays (school term, school holidays, and the public holidays on a weekday, which run the Sunday timetable), as the
  simulation is an average working day and the TL counts average all the dates from Monday to Friday;
- each trip is put in the hour of its passage at the stop: its departure plus the travel time to the stop in the GTFS;
- the std between days is estimated from the TL counts, where the variance grows as a power of the mean
  (var = a mean^b fitted on the TL stops/hours). The minimum, maximum and quantiles are left empty.
"""
import glob
import os
import re
import sys
import unicodedata

import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TPG_comparison"))

import gtfs_utils # pyright: ignore[reportMissingImports]

# Weekdays of the 2024 timetable (10.12.2023 - 14.12.2024, 53 full weeks) per MBC day type, from the footers of the
# diagrams: school term (PS, 190 days) and school holidays (VS, 64). The BAM (line 56) splits the school weekdays into
# Monday to Thursday and Friday. The public holidays on a weekday run the Sunday timetable (DI): its 63 days are the
# 53 Sundays and 10 such holidays (the Saturday timetable has the 53 Saturdays only). Weekends are left out.
WEEKDAY_DAYS = {"PS": 190, "VS": 64, "LU-JE": 152, "VE": 38, "DI": 10}
WEEKDAYS = WEEKDAY_DAYS["PS"] + WEEKDAY_DAYS["VS"] + WEEKDAY_DAYS["DI"]

# The MBC lines named differently in the GTFS (route_short_name): the BAM train is R56 (the TL have a bus 56)
MBC_TO_MATSIM_LINE = {"56": "R56"}

# The "Didok" column of the diagrams looks the names up in their sheet Feuil1, the station of the BAM as "Morges, gare".
# Two stations have another name in the current GTFS. The other stops that it does not have (Lonay, église, ...) are lost.
NAME_ALIASES = {"Morges": "Morges, gare", "Allaman, gare sud": "Allaman, gare", "Lonay, gare": "Lonay-Préverenges, gare nord"}

# GTFS stops matched by name, around the MBC network
NETWORK_AREA = {"lat": (46.40, 46.70), "lon": (6.20, 6.75)}

TRIP_HEADER = re.compile(r"^(Mon|Des)\.:.*?dép\.:\s*(\d+):(\d+)(?::(\d+))?.*?mes\.:\s*(\d+)", re.S)

_STATS = ["min", "max", "mean", "std", "q10", "q20", "q50", "q80", "q90"]


def configure(context):
    context.stage("data.gtfs.cleaned")
    context.config("data_path")

    context.config("analysis.pt.mbc_raw_counts_path",     default = "Lausanne_passenger_counts/MBC/H24")
    context.config("analysis.pt.tl_processed_counts_path", default = "Lausanne_passenger_counts/tl data 2024/processed")


def execute(context):

    # Read the trips of the weekday diagrams and the stop names of their Feuil1 lookup.

    data_path = context.config("data_path")
    trips, lookup = read_diagrams(os.path.join(data_path, context.config("analysis.pt.mbc_raw_counts_path")))
    gtfs_path = "%s/output" % context.path("data.gtfs.cleaned")

    # Match the stops to the GTFS stations (DiDok numbers).

    stops = read_gtfs_stops(gtfs_path)
    crosswalk = build_stop_crosswalk(trips["stop_name"].unique(), lookup, stops)
    trips = trips.merge(crosswalk[["stop_name", "gtfs_code"]], on = "stop_name", how = "left")

    # Put each trip in the hour of its passage at each stop.

    times = read_gtfs_line_times(gtfs_path, set(trips["line"]))
    trips = assign_hours(trips, times)

    # Average the day types over the weekdays, and estimate the std between days from the TL counts.

    result = weekday_means(trips)
    variance = fit_tl_variance(os.path.join(data_path, context.config("analysis.pt.tl_processed_counts_path")))
    result = add_statistics(result, variance)

    # Write the counts in the format of tl_2024 and their companion files.

    path = os.path.join(context.path(), "mbc2024_agg_workday.csv")
    result.to_csv(path, index = False)
    line_directions(trips).to_csv(os.path.join(context.path(), "mbc2024_line_directions.csv"), index = False)
    crosswalk.to_csv(os.path.join(context.path(), "mbc2024_stop_crosswalk.csv"), index = False)
    print_check(trips, result)

    return dict(done = True, path = path)


def normalize_name(name):
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def sheet_type(title):
    """(direction H/R, day type) of a sheet ("PS A", "VS Semaine R", "Aller_LU-JE", "Retour_VS", "Aller", "Dimanche A"),
    None for the Saturdays and the other sheets. The sheets without day type (line 706) run on school weekdays; the
    Sunday sheets (DI) stand for the public holidays on a weekday."""
    tokens = ["DI" if token == "Dimanche" else token for token in title.replace("_", " ").split()]
    if not tokens or any(token in ("SA", "Samedi", "Feuil1") for token in tokens):
        return None

    direction = "H" if "Aller" in tokens or tokens[-1] == "A" else "R" if "Retour" in tokens or tokens[-1] == "R" else None
    day = next((token for token in tokens if token in WEEKDAY_DAYS), "PS")

    return (direction, day) if direction else None


def _value(cell):
    return float(cell) if isinstance(cell, (int, float)) else np.nan


def read_sheet(rows):
    """Rows (trip column, departure in s, measured runs, position, stop name, boardings, alightings) of a diagram: one
    per trip and stop it serves ("-" in both columns: not served)."""
    header = next((i for i, row in enumerate(rows) if row and row[0] in ("Lieu", '=("Lieu")')), None)
    if header is None:
        return []

    trips = []
    for column, cell in enumerate(rows[header]):
        match = TRIP_HEADER.match(cell) if isinstance(cell, str) else None
        if match and match.group(1) == "Mon":
            hours, minutes, seconds, measured = match.group(2), match.group(3), match.group(4) or 0, match.group(5)
            trips.append((column, int(hours) * 3600 + int(minutes) * 60 + int(seconds), int(measured)))

    records, position = [], 0
    for row in rows[header + 1:]:
        name = row[0] if row else None
        if not isinstance(name, str) or not name.strip():
            continue
        name = re.sub(r'^=\("(.*)"\)$', r"\1", name.strip())
        if name == "TOTAL":
            break

        values = [(column, departure, measured, _value(row[column]), _value(row[column + 1])) for column, departure, measured in trips]
        if all(np.isnan(b) and np.isnan(a) for *_, b, a in values):
            continue  # a stop that no trip serves, e.g. the first stop repeated as the arrival of a loop

        records += [(column, departure, measured, position, name, b, a) for column, departure, measured, b, a in values
                    if not (np.isnan(b) and np.isnan(a))]
        position += 1

    return records


def read_diagrams(folder):
    """The weekday trips of all the diagrams (columns line, direction, day_type, sheet, trip, departure, measured,
    position, stop_name, boardings, alightings) and the {stop name: DiDok} of their Feuil1 sheets."""
    paths = sorted(glob.glob(os.path.join(folder, "H24_*.xlsx")))
    if not paths:
        raise FileNotFoundError(f"No H24_<line>.xlsx in {folder}")

    frames, lookup = [], {}
    for path in paths:
        line = re.match(r"H24_(\w+)\.xlsx$", os.path.basename(path)).group(1)
        line = MBC_TO_MATSIM_LINE.get(line, line)
        workbook = openpyxl.load_workbook(path, read_only = True)

        if "Feuil1" in workbook.sheetnames:
            for row in workbook["Feuil1"].iter_rows(values_only = True, max_col = 2):
                name, code = (tuple(row) + (None, None))[:2]
                if name and isinstance(code, (int, float)):
                    lookup.setdefault(normalize_name(name), str(int(code)))

        for sheet in workbook.worksheets:
            kind = sheet_type(sheet.title)
            records = read_sheet(list(sheet.iter_rows(values_only = True))) if kind else []
            if not records:
                continue

            frame = pd.DataFrame(records, columns = ["trip", "departure", "measured", "position", "stop_name", "boardings", "alightings"])
            frames.append(frame.assign(line = line, direction = kind[0], day_type = kind[1], sheet = sheet.title))

        workbook.close()

    trips = pd.concat(frames, ignore_index = True)
    trips[["boardings", "alightings"]] = trips[["boardings", "alightings"]].fillna(0.0)

    sheets = trips.groupby(["line", "direction", "day_type"])["trip"].nunique()
    print(f"MBC: {len(paths)} diagrams, {len(sheets)} weekday sheets, {int(sheets.sum())} trips")

    return trips, lookup


def read_gtfs_stops(gtfs_path):
    stops = pd.read_csv(os.path.join(gtfs_path, "stops.txt"), dtype = str, usecols = ["stop_id", "stop_name", "stop_lat", "stop_lon"])
    stops["code"] = stops["stop_id"].map(gtfs_utils.stop_code)
    stops[["stop_lat", "stop_lon"]] = stops[["stop_lat", "stop_lon"]].astype(float)
    return stops


def build_stop_crosswalk(names, lookup, stops):
    """{MBC stop name: GTFS station (DiDok)}: the DiDok of Feuil1 when the GTFS has it, else the GTFS stop of the same
    name around the MBC network (the names of the diagrams have lost their accents)."""
    codes = set(stops["code"])
    area = stops[stops["stop_lat"].between(*NETWORK_AREA["lat"]) & stops["stop_lon"].between(*NETWORK_AREA["lon"])]
    by_name = area.assign(key = area["stop_name"].map(normalize_name)).groupby("key")["code"].agg(lambda s: s.mode().iloc[0]).to_dict()

    rows = []
    for name in sorted(names):
        key = normalize_name(NAME_ALIASES.get(name, name))
        code = lookup.get(key)
        if code in codes:
            rows.append((name, code, "didok"))
        elif key in by_name:
            rows.append((name, by_name[key], "name"))
        else:
            rows.append((name, None, "unmatched"))

    crosswalk = pd.DataFrame(rows, columns = ["stop_name", "gtfs_code", "match"])
    print("MBC stops: " + ", ".join(f"{n} by {m}" for m, n in crosswalk["match"].value_counts().items()))
    for name in crosswalk.loc[crosswalk["match"] == "unmatched", "stop_name"]:
        print(f"  MBC stop without GTFS stop, left out: {name}")

    return crosswalk


def read_gtfs_line_times(gtfs_path, lines):
    """Passages (line, trip_id, code, seconds) of the GTFS trips of the given lines (route_short_name) of the MBC."""
    agency = pd.read_csv(os.path.join(gtfs_path, "agency.txt"), dtype = str, usecols = ["agency_id", "agency_name"])
    agencies = agency.loc[agency["agency_name"].str.contains("MBC|Morges-Bi", case = False, na = False), "agency_id"]

    routes = pd.read_csv(os.path.join(gtfs_path, "routes.txt"), dtype = str, usecols = ["route_id", "agency_id", "route_short_name"])
    routes = routes[routes["agency_id"].isin(agencies) & routes["route_short_name"].isin(lines)]
    trips = pd.read_csv(os.path.join(gtfs_path, "trips.txt"), dtype = str, usecols = ["route_id", "trip_id"])
    trips = trips.merge(routes[["route_id", "route_short_name"]], on = "route_id")
    line_of_trip = dict(zip(trips["trip_id"], trips["route_short_name"]))

    columns = ["trip_id", "departure_time", "stop_id", "stop_sequence"]
    options = pacsv.ConvertOptions(include_columns = columns, column_types = {c: pa.string() for c in columns})
    wanted = pa.array(sorted(line_of_trip))
    batches = []
    with pacsv.open_csv(os.path.join(gtfs_path, "stop_times.txt"), convert_options = options) as reader:
        for batch in reader:
            batch = batch.filter(pc.is_in(batch.column("trip_id"), value_set = wanted))
            if batch.num_rows:
                batches.append(batch)

    times = pa.Table.from_batches(batches).to_pandas() if batches else pd.DataFrame(columns = columns)
    parts = times["departure_time"].str.split(":", expand = True).astype(float)
    times["seconds"] = parts[0] * 3600 + parts[1] * 60 + parts[2]
    times["code"] = times["stop_id"].map(gtfs_utils.stop_code)
    times["line"] = times["trip_id"].map(line_of_trip)
    times["stop_sequence"] = times["stop_sequence"].astype(int)

    print(f"GTFS: {times['trip_id'].nunique()} trips of the MBC lines {sorted(set(times['line']))}")
    return times.sort_values(["trip_id", "stop_sequence"])[["line", "trip_id", "code", "seconds"]]


def stop_offsets(codes, passages):
    """Time in s from the first stop to each stop of the sequence of codes: the median GTFS time between consecutive
    stops (served in this order), interpolated along the sequence for the stops that the GTFS does not have."""
    known, previous = {}, None
    for k, code in enumerate(codes):
        if code is None or code not in passages:
            continue
        if previous is None:
            known[k], previous = 0.0, k
            continue

        times = [b - a for trip, a in passages[codes[previous]].items() if (b := passages[code].get(trip)) is not None and b > a]
        if times:
            known[k], previous = known[previous] + float(np.median(times)), k

    if not known:
        return np.zeros(len(codes))

    positions = sorted(known)
    return np.interp(np.arange(len(codes)), positions, [known[p] for p in positions])


def assign_hours(trips, times):
    """The hour of each trip at each stop: its departure (at its first stop in the diagram) plus the travel time from
    there, from the GTFS trips of the line."""
    missing = sorted(set(trips["line"]) - set(times["line"]))
    if missing:
        print(f"  lines not in the GTFS, their trips are counted in the hour of their departure: {missing}")

    hours = []
    for (line, sheet), group in trips.groupby(["line", "sheet"], sort = False):
        codes = group.drop_duplicates("position").sort_values("position")["gtfs_code"].tolist()
        line_times = times[times["line"] == line]
        passages = {code: dict(zip(rows["trip_id"], rows["seconds"])) for code, rows in line_times.drop_duplicates(["trip_id", "code"]).groupby("code")}

        offsets = stop_offsets([c if isinstance(c, str) else None for c in codes], passages)

        offset = pd.Series(offsets, index = range(len(codes)))
        start = group.groupby("trip")["position"].transform("min")
        seconds = group["departure"] + group["position"].map(offset).to_numpy() - start.map(offset).to_numpy()
        hours.append(pd.Series((seconds // 3600).astype(int) % 24, index = group.index))

    return trips.assign(hour = pd.concat(hours))


def weekday_means(trips):
    """Mean working-day boardings/alightings per GTFS station, line direction and hour: the sums of the trips of each day
    type weighted by its share of the weekdays (no trip on the day types without sheet, e.g. the school line 706)."""
    trips = trips.dropna(subset = ["gtfs_code"])
    weight = trips["day_type"].map(WEEKDAY_DAYS) / WEEKDAYS
    trips = trips.assign(
        line_direction = trips["line"] + "_" + trips["direction"],
        boardings = trips["boardings"] * weight, alightings = trips["alightings"] * weight,
    )

    return trips.groupby(["gtfs_code", "line_direction", "hour"], as_index = False)[["boardings", "alightings"]].sum()


def fit_tl_variance(processed_dir):
    """(a, b) of var = a mean^b between the days, fitted (log-log) on the stops/hours of the latest processed TL counts;
    (1, 1) (Poisson) without them."""
    paths = sorted(p for p in glob.glob(os.path.join(processed_dir, "tl2024_agg_workday_*.csv"))
                   if re.fullmatch(r"tl2024_agg_workday_\d{8}\.csv", os.path.basename(p)))
    if not paths:
        print(f"No processed TL counts in {processed_dir}: the std of the MBC counts is the one of a Poisson distribution")
        return 1.0, 1.0

    tl = pd.read_csv(paths[-1], usecols = ["boardings_raw_mean", "boardings_raw_std", "alightings_raw_mean", "alightings_raw_std"])
    mean = np.r_[tl["boardings_raw_mean"], tl["alightings_raw_mean"]]
    std = np.r_[tl["boardings_raw_std"], tl["alightings_raw_std"]]
    keep = (mean > 0.1) & (std > 0)
    b, log_a = np.polyfit(np.log(mean[keep]), 2 * np.log(std[keep]), 1)

    print(f"Std of the MBC counts between days from {os.path.basename(paths[-1])}: var = {np.exp(log_a):.2f} mean^{b:.2f} "
          f"({keep.sum()} TL stops/hours)")
    return float(np.exp(log_a)), float(b)


def add_statistics(means, variance):
    """The columns of tl_2024: the mean, the std from the variance law, the other statistics empty."""
    a, b = variance
    out = means[["gtfs_code", "line_direction", "hour"]].copy()
    out.insert(0, "day_type", "Weekday")

    for prefix in ("boardings", "alightings"):
        values = {"mean": means[prefix], "std": np.sqrt(a * means[prefix].clip(lower = 0) ** b)}
        for stat in _STATS:
            column = values.get(stat, np.nan)
            out[f"{prefix}_raw_{stat}"] = column
            out[f"{prefix}_{stat}"] = column

    return out


def line_directions(trips):
    first = trips.sort_values("position").groupby(["line", "direction"])
    out = pd.DataFrame({"first_stop": first["stop_name"].first(), "last_stop": first["stop_name"].last()}).reset_index()
    out["line_direction"] = out["line"] + "_" + out["direction"]
    out["direction_extended"] = out["first_stop"] + "->" + out["last_stop"]
    return out[["line_direction", "line", "direction", "first_stop", "last_stop", "direction_extended"]]


def print_check(trips, result):
    """Weekday boardings per line, and the share lost with the stops without GTFS station."""
    weight = trips["day_type"].map(WEEKDAY_DAYS) / WEEKDAYS
    total = (trips["boardings"] * weight).groupby(trips["line"]).sum()
    kept = result.assign(line = result["line_direction"].str.rsplit("_", n = 1).str[0]).groupby("line")["boardings_raw_mean"].sum()
    table = pd.DataFrame({"weekday_boardings": total, "matched_share": kept.reindex(total.index) / total})
    print(table.round(3).to_string())
