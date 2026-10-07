import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
import stages


def configure(context):
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("extent_prefix", default = "")
    context.config("simulation_directory", default = "simulation_output")

    context.config("analysis.pt.tpg_processed_counts_path", default = "TPG_passenger_counts")
    context.config("analysis.pt.perimeter", default = "spatial/MMT/CMDP_Limites_WG84.shp")
    context.config("analysis.pt.tpg_data",  default = "TPG_passenger_counts")
    context.config("analysis.pt.gtfs_zip",  default = "gtfs/gtfs_fp2024_2024-11-11.zip")

    context.config("analysis.pt.year", default = 2025)

    context.config("analysis.pt.min_stop_avg_events",   default = 10.0)
    context.config("analysis.pt.min_stop_active_hours", default = 6)
    context.config("analysis.pt.map_hour_start",        default = 6)
    context.config("analysis.pt.map_hour_end",          default = 22)

    context.config("analysis.pt.stop_line_comparison", default = False)
    context.config("analysis.pt.stop", default = "Genève, gare Cornavin")
    context.config("analysis.pt.line", default = None)
    context.config("analysis.pt.stop_all_lines", default = None)
    context.config("analysis.pt.skip_global_comparison", default = False)

    context.config("analysis.pt.include_lemanis",  default = False)
    context.config("analysis.pt.lemanis_csv_path", default = None)


def run(gtfs_zip, perimeter_shapefile, tpg_data_path, matsim_output_folder, input_downsampling,
        year = 2025, output_dir = None, tpg_processed_counts_path = None, output_path = None,
        min_stop_avg_events = 10.0, min_stop_active_hours = 6, map_hour_start = 6, map_hour_end = 22,
        include_lemanis = False, lemanis_csv_path = None,
        stop_line_comparison = False, stop = "Genève, gare Cornavin", line = None,
        use_schedule_stops = True, stop_all_lines = None, skip_global_comparison = False):
    
    if include_lemanis and not lemanis_csv_path:
        raise RuntimeError("include_lemanis is true but lemanis_csv_path is not set")

    if not os.path.isdir(matsim_output_folder):
        raise FileNotFoundError(
            f"MATSim output not found at {matsim_output_folder} - has the simulation been run "
            "and the PT passenger counts been written to it?"
        )

    counts_file = os.path.join(matsim_output_folder, "pt_passenger_counts.csv.gz")
    if not os.path.exists(counts_file):
        raise FileNotFoundError(f"{counts_file} not found in the MATSim output folder")

    cfg = Config(
        gtfs_zip                  = gtfs_zip,
        perimeter_shapefile       = perimeter_shapefile,
        tpg_data_path             = tpg_data_path,
        matsim_output_folder      = matsim_output_folder,
        output_path               = output_path or os.path.dirname(os.path.abspath(matsim_output_folder)),
        tpg_processed_counts_path = tpg_processed_counts_path or tpg_data_path,
        input_downsampling        = input_downsampling,
        min_stop_avg_events       = min_stop_avg_events,
        min_stop_active_hours     = min_stop_active_hours,
        map_hour_start            = map_hour_start,
        map_hour_end              = map_hour_end,
        include_lemanis           = include_lemanis,
        lemanis_csv_path          = lemanis_csv_path,
        use_schedule_stops        = use_schedule_stops,
    )

    output_dir = output_dir or os.path.join(matsim_output_folder, f"pt_comparison_tpg_{year}")

    if not skip_global_comparison:
        print(f"Running global (perimeter-wide) PT comparison for {year} -> {output_dir}")
        stages.run_global_comparison(cfg, output_dir, year)

    if stop_line_comparison:
        if line is None:
            line = "1_H" if year == 2024 else "1"

        print(f"Running {year} stop/line PT comparison for stop={stop!r} line={line!r} -> {output_dir}")
        stages.run_stop_line_comparison(cfg, output_dir, year, stop = stop, line = line)

    if stop_all_lines:
        print(f"Running all-lines passenger movements for stop {stop_all_lines!r} -> {output_dir}")
        stages.run_stop_all_lines(cfg, output_dir, stop = stop_all_lines)

    return output_dir


def execute(context):
    data_path = context.config("data_path")

    matsim_output_folder = os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory")
    )

    output_dir = run(
        gtfs_zip                  = os.path.join(data_path, context.config("analysis.pt.gtfs_zip")),
        perimeter_shapefile       = os.path.join(data_path, context.config("analysis.pt.perimeter")),
        tpg_data_path             = os.path.join(data_path, context.config("analysis.pt.tpg_data")),
        matsim_output_folder      = matsim_output_folder,
        input_downsampling        = context.config("input_downsampling"),
        year                      = context.config("analysis.pt.year"),
        tpg_processed_counts_path = os.path.join(data_path, context.config("analysis.pt.tpg_processed_counts_path")),
        output_path               = context.config("output_path"),
        min_stop_avg_events       = context.config("analysis.pt.min_stop_avg_events"),
        min_stop_active_hours     = context.config("analysis.pt.min_stop_active_hours"),
        map_hour_start            = context.config("analysis.pt.map_hour_start"),
        map_hour_end              = context.config("analysis.pt.map_hour_end"),
        include_lemanis           = context.config("analysis.pt.include_lemanis"),
        lemanis_csv_path          = context.config("analysis.pt.lemanis_csv_path"),
        stop_line_comparison      = context.config("analysis.pt.stop_line_comparison"),
        stop                      = context.config("analysis.pt.stop"),
        line                      = context.config("analysis.pt.line"),
        stop_all_lines            = context.config("analysis.pt.stop_all_lines"),
        skip_global_comparison    = context.config("analysis.pt.skip_global_comparison"),
    )

    return dict(done = True, path = output_dir)


def main(argv = None):
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matsim-output", required = True,
                        help = "MATSim output folder containing pt_passenger_counts.csv.gz")
    parser.add_argument("--gtfs-zip", required = True, help = "GTFS zip of the simulated schedule")
    parser.add_argument("--perimeter", required = True, help = "Shapefile of the Geneva perimeter")
    parser.add_argument("--tpg-data", required = True, help = "Folder with the TPG data (stops info, line routes ...)")
    parser.add_argument("--input-downsampling", type = float, required = True,
                        help = "Sample rate of the simulated population (counts are divided by it)")
    parser.add_argument("--tpg-processed-counts", default = None,
                        help = "Folder with tpg<year>_agg_workday_*.csv (default: --tpg-data)")
    parser.add_argument("--year", type = int, default = 2025)
    parser.add_argument("--output-dir", default = None,
                        help = "Where to write the results (default: <matsim-output>/pt_comparison_tpg_<year>)")
    parser.add_argument("--min-stop-avg-events", type = float, default = 10.0)
    parser.add_argument("--min-stop-active-hours", type = int, default = 6)
    parser.add_argument("--map-hour-start", type = int, default = 6)
    parser.add_argument("--map-hour-end", type = int, default = 22)
    parser.add_argument("--lemanis-csv", default = None, help = "If given, Léman Express is included")
    parser.add_argument("--no-schedule-stops", action = "store_true",
                        help = "Only use the stops of the GTFS. By default the stops of the simulated schedule "
                               "that the GTFS does not know (e.g. the French Leman Express stops) are added")
    parser.add_argument("--stop-line-comparison", action = "store_true")
    parser.add_argument("--stop-all-lines", default = None,
                        help = "Plot the MATSim boardings/alightings of all lines at this stop (e.g. Annemasse) over the day")
    parser.add_argument("--skip-global-comparison", action = "store_true",
                        help = "Skip the perimeter-wide TPG comparison (e.g. to only run --stop-all-lines)")
    parser.add_argument("--stop", default = "Genève, gare Cornavin")
    parser.add_argument("--line", default = None, help = "Default: 1_H for 2024, 1 otherwise")
    args = parser.parse_args(argv)

    output_dir = run(
        gtfs_zip = args.gtfs_zip, perimeter_shapefile = args.perimeter, tpg_data_path = args.tpg_data,
        matsim_output_folder = args.matsim_output, input_downsampling = args.input_downsampling,
        year = args.year, output_dir = args.output_dir, tpg_processed_counts_path = args.tpg_processed_counts,
        min_stop_avg_events = args.min_stop_avg_events, min_stop_active_hours = args.min_stop_active_hours,
        map_hour_start = args.map_hour_start, map_hour_end = args.map_hour_end,
        include_lemanis = args.lemanis_csv is not None, lemanis_csv_path = args.lemanis_csv,
        stop_line_comparison = args.stop_line_comparison, stop = args.stop, line = args.line,
        use_schedule_stops = not args.no_schedule_stops,
        stop_all_lines = args.stop_all_lines, skip_global_comparison = args.skip_global_comparison,
    )
    print("Results written to", output_dir)


if __name__ == "__main__":
    main()
