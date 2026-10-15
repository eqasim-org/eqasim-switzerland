import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

# The comparison machinery (GTFS and MATSim readers, tables, maps) is the one of the TPG comparison
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TPG_comparison"))

from config import Config  # pyright: ignore[reportMissingImports]
from pt_data import PtData # pyright: ignore[reportMissingImports]
import comparison          # pyright: ignore[reportMissingImports]
import line_matching       # pyright: ignore[reportMissingImports]
import interactive_map     # pyright: ignore[reportMissingImports]
import plotting            # pyright: ignore[reportMissingImports]
import stages              # pyright: ignore[reportMissingImports]
import tpg_data            # pyright: ignore[reportMissingImports]

REFERENCE_NAME = "TL"

_SUM_COLUMNS = ["boardings_raw_mean", "alightings_raw_mean", "boardings_raw_var", "alightings_raw_var"]

# Lines that are numbered differently in the TL counts and in the simulated schedule (GTFS route_short_name)
TL_TO_MATSIM_LINE = {"70": "m1", "72": "m2", "77": "R20", "91": "N1", "92": "N2", "93": "N3", "94": "N4", "95": "N5", "96": "N6"}

# Maximum distance to move a TL stop to the stop served by the same line in MATSim
MAX_STOP_DISTANCE = 250.0

# Stops closer than this are the same station (e.g. the Flon of the m1 and the one of the m2 have different GTFS ids)
STATION_DISTANCE = 150.0


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

    context.config("analysis.pt.tl_processed_path", default = "Lausanne_passenger_counts/tl data 2024/processed")
    context.config("analysis.pt.tl_gtfs_zip",       default = "gtfs/gtfs_fp2024_2024-11-11.zip")
    context.config("analysis.pt.tl_perimeter",      default = "spatial/LausanneExtent/LausanneExtent.shp")

    context.config("analysis.pt.min_stop_avg_events",   default = 10.0)
    context.config("analysis.pt.min_stop_active_hours", default = 6)
    context.config("analysis.pt.map_hour_start",        default = 6)
    context.config("analysis.pt.map_hour_end",          default = 22)
    context.config("analysis.pt.night_hours",           default = 5)
    context.config("analysis.pt.counts_source",         default = "auto")

    context.config("analysis.pt.include_sbb", default = True)
    context.config("analysis.pt.sbb_path",    default = "Geneva_passenger_counts/SBB/passagierfrequenz.csv")
    context.config("analysis.pt.sbb_year",    default = 2024)


def resolve_tl_stats_path(processed_dir):
    """Latest tl2024_agg_workday_<date>.csv (written by tl_2024.py), without its companion files."""
    candidates = sorted(
        path for path in glob.glob(os.path.join(processed_dir, "tl2024_agg_workday_*.csv"))
        if re.fullmatch(r"tl2024_agg_workday_\d{8}\.csv", os.path.basename(path))
    )

    if not candidates:
        raise FileNotFoundError(
            f"No processed TL stats found in {processed_dir}: run the stage analysis.pt.TL_comparison.tl_2024 first"
        )

    return candidates[-1]


# Names of the transport modes of the transit schedule on the map, and the lines that get a submode of their own
SUBMODE_NAMES     = {"subway": "metro", "rail": "train"}
SUBMODE_OF_LINE   = {"R20": "LEB"} # the only R20 inside the perimeter (the other one is Neuchâtel's)


def submode_of(line_name, line_modes):
    return SUBMODE_OF_LINE.get(line_name) or SUBMODE_NAMES.get(line_modes.get(line_name), line_modes.get(line_name, "bus"))


def _sum_stats(tl):
    return tl.groupby(["gtfs_code", "line_name", "hour"], as_index = False)[_SUM_COLUMNS].sum()


def load_tl_line_stats(stats_path):
    tl = pd.read_csv(stats_path)
    tl["gtfs_code"] = tl["gtfs_code"].astype(str)
    tl["line_name"] = tl["line_direction"].astype(str).str.replace(r"_[HR]$", "", regex = True).replace(TL_TO_MATSIM_LINE)

    for prefix in ("boardings", "alightings"):
        tl[f"{prefix}_raw_var"] = tl[f"{prefix}_raw_std"].fillna(0) ** 2

    return _sum_stats(tl)


def _distance_m(lat1, lon1, lat2, lon2):
    return 111_320 * np.hypot(lat1 - lat2, (lon1 - lon2) * np.cos(np.radians(lat1)))


def remap_to_matsim_stops(tl, counts, stop_coordinates, max_distance = MAX_STOP_DISTANCE):
    served = counts.groupby("line_name")["stop_id_gtfs_base"].agg(lambda s: sorted(set(s))).to_dict()

    mapping = {}
    for line, code in tl[["line_name", "gtfs_code"]].drop_duplicates().itertuples(index = False):
        candidates = served.get(line, [])

        if code in candidates or code not in stop_coordinates:
            continue

        lat, lon = stop_coordinates[code]
        distances = [(_distance_m(lat, lon, *stop_coordinates[c]), c) for c in candidates if c in stop_coordinates]

        if distances and min(distances)[0] <= max_distance:
            mapping[(line, code)] = min(distances)[1]

    print(f"{len(mapping)} TL (line, stop) moved to the closest stop served by the line in MATSim")

    tl = tl.copy()
    tl["gtfs_code"] = [mapping.get(key, key[1]) for key in zip(tl["line_name"], tl["gtfs_code"])]

    return _sum_stats(tl)


def station_codes(codes, stop_coordinates, weights, max_distance = STATION_DISTANCE):
    """Maps each GTFS stop code to the code of its station: the busiest one among the codes closer than max_distance."""
    codes = sorted(c for c in set(codes) if c in stop_coordinates)
    lat0  = np.mean([stop_coordinates[c][0] for c in codes])
    xy    = np.array([
        [111_320 * np.cos(np.radians(lat0)) * stop_coordinates[c][1], 111_320 * stop_coordinates[c][0]] for c in codes
    ])

    pairs = np.array(sorted(cKDTree(xy).query_pairs(max_distance)), dtype = int).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape = (len(codes), len(codes)))
    _, component = connected_components(graph, directed = False)

    busiest = {}
    for code, group in zip(codes, component):
        if group not in busiest or weights.get(code, 0) > weights.get(busiest[group], 0):
            busiest[group] = code

    return {code: busiest[group] for code, group in zip(codes, component)}


def line_modes_of(counts, mode_of_line_id):
    """{line name: transport mode of the transit schedule}, the one that carries most passengers among the lines of the name"""
    lines = counts.assign(passengers = counts["boardings"] + counts["alightings"])
    lines = lines.groupby(["line_name", "line_id"], as_index = False)["passengers"].sum()
    lines["mode"] = lines["line_id"].map(mode_of_line_id)

    modes = lines.dropna(subset = ["mode"]).groupby(["line_name", "mode"], as_index = False)["passengers"].sum()
    modes = modes.sort_values("passengers", ascending = False).drop_duplicates("line_name")

    return {str(name): mode for name, mode in zip(modes["line_name"], modes["mode"])}


def build_tl_mofr(data, tl_stats_path):
    cfg = data.cfg

    counts = data.counts_in(data.stops_in_perimeter, own_lines_only = True)
    counts["stop_id_gtfs_base"] = counts["stop_id_gtfs"].astype(str).str.split(":").str[0]
    counts["line_name"] = counts["line_name"].astype(str)

    print(f"Loading the TL stats from {tl_stats_path}")
    tl_mofr = load_tl_line_stats(tl_stats_path)

    # one transit line per line name, the one that the TL counts: the others are other networks or other modes
    counts = line_matching.select_line_ids(counts, tl_mofr, data.stop_coordinates, MAX_STOP_DISTANCE, label = "TL")
    line_modes = line_modes_of(counts, data.line_modes)

    counts = counts.groupby(["stop_id_gtfs_base", "line_name", "hour"], as_index = False)[["boardings", "alightings"]].sum()
    counts[["boardings", "alightings"]] /= cfg.input_downsampling

    # TL lines that the simulation does not have (e.g. the depot line 23) would only add MATSim zeros
    absent = sorted(set(tl_mofr["line_name"]) - set(counts["line_name"]))
    if absent:
        print(f"TL lines without MATSim counterpart, left out: {absent}")
        tl_mofr = tl_mofr[~tl_mofr["line_name"].isin(absent)]

    tl_mofr = remap_to_matsim_stops(tl_mofr, counts, data.stop_coordinates)

    weights  = counts.groupby("stop_id_gtfs_base")[["boardings", "alightings"]].sum().sum(axis = 1).to_dict()
    stations = station_codes(set(counts["stop_id_gtfs_base"]) | set(tl_mofr["gtfs_code"]), data.stop_coordinates, weights)
    merged   = {code: station for code, station in stations.items() if code != station}
    print(f"{len(merged)} GTFS stops merged into the station within {STATION_DISTANCE:.0f} m that has the most MATSim passengers")

    # the stops of a station are drawn as one per submode: each keeps the name of its own stop of that submode, e.g.
    # "Lausanne, gare" for the metro and not the name of the train station that the station code comes from
    origins = pd.concat([
        pd.DataFrame({"code": counts["stop_id_gtfs_base"], "line_name": counts["line_name"],
                      "weight": counts["boardings"] + counts["alightings"]}),
        pd.DataFrame({"code": tl_mofr["gtfs_code"], "line_name": tl_mofr["line_name"],
                      "weight": tl_mofr["boardings_raw_mean"] + tl_mofr["alightings_raw_mean"]}),
    ])
    origins["submode"] = [submode_of(name, line_modes) for name in origins["line_name"]]
    origins["station"] = origins["code"].map(lambda c: stations.get(c, c))
    stop_names = data.gtfs_stops.drop_duplicates("stop_id").set_index("stop_id")["stop_name"].to_dict()
    origins = origins.groupby(["station", "submode", "code"], as_index = False)["weight"].sum()
    origins = origins.sort_values("weight", ascending = False).drop_duplicates(["station", "submode"])
    submode_names = {(r.station, r.submode): stop_names[r.code] for r in origins.itertuples() if r.code in stop_names}

    counts["stop_id_gtfs_base"] = counts["stop_id_gtfs_base"].map(lambda c: stations.get(c, c))
    counts = counts.groupby(["stop_id_gtfs_base", "line_name", "hour"], as_index = False)[["boardings", "alightings"]].sum()
    tl_mofr["gtfs_code"] = tl_mofr["gtfs_code"].map(lambda c: stations.get(c, c))
    tl_mofr = _sum_stats(tl_mofr)

    for prefix in ("boardings", "alightings"):
        tl_mofr[f"{prefix}_raw_std"] = np.sqrt(tl_mofr.pop(f"{prefix}_raw_var"))

    tl_mofr = tl_mofr.merge(
        counts.rename(columns = {"boardings": "boardings_matsim", "alightings": "alightings_matsim"}),
        left_on = ["gtfs_code", "line_name", "hour"], right_on = ["stop_id_gtfs_base", "line_name", "hour"], how = "left",
    )
    tl_mofr[["boardings_matsim", "alightings_matsim"]] = tl_mofr[["boardings_matsim", "alightings_matsim"]].fillna(0)

    gtfs_stops = data.gtfs_stops
    tl_mofr = tl_mofr.merge(gtfs_stops[["stop_id", "stop_name"]].drop_duplicates("stop_id"), left_on = "gtfs_code", right_on = "stop_id", how = "left")

    unmatched = tl_mofr["stop_name"].isna().sum()
    if unmatched:
        print(f"Dropping {unmatched}/{len(tl_mofr)} TL rows whose gtfs_code isn't in the current GTFS feed")
        tl_mofr = tl_mofr[tl_mofr["stop_name"].notna()]

    n_stops_before = tl_mofr["gtfs_code"].nunique()
    tl_mofr = comparison.filter_active_stops(
        tl_mofr, min_avg_events = cfg.min_stop_avg_events, min_active_hours = cfg.min_stop_active_hours,
    )
    print(
        f"Restricting to active stops (>{cfg.min_stop_avg_events} avg passenger movements during "
        f">={cfg.min_stop_active_hours} hours/day): {tl_mofr['gtfs_code'].nunique()}/{n_stops_before} stops kept"
    )

    # the map shows one marker per stop and submode (metro, LEB, train, tram, bus ...)
    tl_mofr["submode"] = [submode_of(name, line_modes) for name in tl_mofr["line_name"]]
    tl_mofr["stop_name"] = [submode_names.get(key, name) for key, name in zip(zip(tl_mofr["gtfs_code"], tl_mofr["submode"]), tl_mofr["stop_name"])]
    print("Submodes of the lines: " + ", ".join(f"{sub} {n}" for sub, n in tl_mofr.groupby("submode")["line_name"].nunique().items()))

    return tl_mofr


def run(gtfs_zip, perimeter_shapefile, tl_processed_path, matsim_output_folder, input_downsampling,
        output_dir = None, min_stop_avg_events = 10.0, min_stop_active_hours = 6, map_hour_start = 6, map_hour_end = 22,
        night_hours = 5, sbb_path = None, sbb_year = 2024, counts_source = "auto"):

    tpg_data.counts_source_available(matsim_output_folder, counts_source)

    cfg = Config(
        gtfs_zip                  = gtfs_zip,
        perimeter_shapefile       = perimeter_shapefile,
        tpg_data_path             = "",
        matsim_output_folder      = matsim_output_folder,
        output_path               = os.path.dirname(os.path.abspath(matsim_output_folder)),
        tpg_processed_counts_path = tl_processed_path,
        input_downsampling        = input_downsampling,
        min_stop_avg_events       = min_stop_avg_events,
        min_stop_active_hours     = min_stop_active_hours,
        map_hour_start            = map_hour_start,
        map_hour_end              = map_hour_end,
        night_hours               = night_hours,
        sbb_path                  = sbb_path,
        counts_source             = counts_source,
    )

    data       = PtData(cfg)
    output_dir = output_dir or os.path.join(matsim_output_folder, "pt_comparison_tl_2024")
    os.makedirs(output_dir, exist_ok = True)

    tl_mofr = build_tl_mofr(data, resolve_tl_stats_path(tl_processed_path))

    stop_hour_df     = comparison.build_stop_hour_table(tl_mofr, data.gtfs_stops, data.stops_in_perimeter)
    global_hourly_df = comparison.build_global_hourly_table(stop_hour_df)

    hour_range            = (map_hour_start, map_hour_end)
    windowed_stop_hour_df = stop_hour_df[stop_hour_df["hour"].between(*hour_range)]
    windowed_full_day_df  = comparison.build_full_day_table(windowed_stop_hour_df)

    stop_hour_df.to_csv(os.path.join(output_dir, "stop_hour_comparison.csv"), index = False)
    windowed_full_day_df.to_csv(os.path.join(output_dir, "stop_full_day_comparison.csv"), index = False)

    plotting.plot_global_hourly_comparison(
        global_hourly_df, os.path.join(output_dir, "global_hourly_comparison.png"), reference_name = REFERENCE_NAME,
    )
    interactive_map.build_hourly_map(
        stop_hour_df, os.path.join(output_dir, "pt_stop_map_by_hour.html"),
        perimeter_shapefile = perimeter_shapefile, hour_range = hour_range, reference_name = REFERENCE_NAME,
    )

    sbb_station_df = None
    if sbb_path:
        print(f"Running the SBB station passenger counts comparison -> {output_dir}")
        sbb_station_df = stages.run_sbb_stations(data, output_dir, year = sbb_year)

    interactive_map.build_full_day_map(
        windowed_stop_hour_df, windowed_full_day_df, os.path.join(output_dir, f"pt_stop_map_{map_hour_start}_{map_hour_end}.html"),
        perimeter_shapefile = perimeter_shapefile, reference_name = REFERENCE_NAME, sbb_station_df = sbb_station_df,
    )

    return output_dir


def execute(context):
    data_path = context.config("data_path")

    matsim_output_folder = os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory")
    )

    output_dir = run(
        gtfs_zip                = os.path.join(data_path, context.config("analysis.pt.tl_gtfs_zip")),
        perimeter_shapefile     = os.path.join(data_path, context.config("analysis.pt.tl_perimeter")),
        tl_processed_path       = os.path.join(data_path, context.config("analysis.pt.tl_processed_path")),
        matsim_output_folder    = matsim_output_folder,
        input_downsampling      = context.config("input_downsampling"),
        min_stop_avg_events     = context.config("analysis.pt.min_stop_avg_events"),
        min_stop_active_hours   = context.config("analysis.pt.min_stop_active_hours"),
        map_hour_start          = context.config("analysis.pt.map_hour_start"),
        map_hour_end            = context.config("analysis.pt.map_hour_end"),
        night_hours             = context.config("analysis.pt.night_hours"),
        sbb_path                = (os.path.join(data_path, context.config("analysis.pt.sbb_path"))
                                   if context.config("analysis.pt.include_sbb") else None),
        sbb_year                = context.config("analysis.pt.sbb_year"),
        counts_source           = context.config("analysis.pt.counts_source"),
    )

    return dict(done = True, path = output_dir)


def main(argv = None):
    parser = argparse.ArgumentParser(description = "Interactive map comparing MATSim and TL passenger counts")
    parser.add_argument("--matsim-output", required = True, help = "MATSim output folder containing pt_passenger_counts.csv.gz or output_events.xml.gz")
    parser.add_argument("--counts-source", default = "auto", choices = tpg_data.COUNTS_SOURCES,
                        help = "file: pt_passenger_counts.csv.gz, events: count the passengers in output_events.xml.gz, "
                               "auto: the file if it exists")
    parser.add_argument("--gtfs-zip", required = True)
    parser.add_argument("--perimeter", required = True, help = "Shapefile of the Lausanne perimeter")
    parser.add_argument("--tl-processed", required = True, help = "Folder with tl2024_agg_workday_<date>.csv")
    parser.add_argument("--input-downsampling", type = float, required = True)
    parser.add_argument("--output-dir", default = None)
    parser.add_argument("--min-stop-avg-events", type = float, default = 10.0)
    parser.add_argument("--min-stop-active-hours", type = int, default = 6)
    parser.add_argument("--map-hour-start", type = int, default = 6)
    parser.add_argument("--map-hour-end", type = int, default = 22)
    parser.add_argument("--sbb-path", default = None, help = "SBB passagierfrequenz.csv: station counts are added to the map")
    parser.add_argument("--sbb-year", type = int, default = 2024)
    parser.add_argument("--night-hours", type = int, default = 5,
                        help = "The MATSim counts of the hours 0 to N-1 are replaced by the ones of the hours 24 to 24+N-1")
    args = parser.parse_args(argv)

    output_dir = run(
        gtfs_zip = args.gtfs_zip, perimeter_shapefile = args.perimeter, tl_processed_path = args.tl_processed,
        matsim_output_folder = args.matsim_output, input_downsampling = args.input_downsampling,
        output_dir = args.output_dir, min_stop_avg_events = args.min_stop_avg_events,
        min_stop_active_hours = args.min_stop_active_hours,
        map_hour_start = args.map_hour_start, map_hour_end = args.map_hour_end, night_hours = args.night_hours,
        sbb_path = args.sbb_path, sbb_year = args.sbb_year, counts_source = args.counts_source,
    )
    print("Results written to", output_dir)


if __name__ == "__main__":
    main()
