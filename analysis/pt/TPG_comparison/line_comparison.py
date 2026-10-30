"""Map comparing the passengers of the lines in MATSim and in the TPG counts: the same map as the TL line comparison
(analysis.pt.TL_comparison.line_comparison), written to the folder of the TPG comparison (pt_comparison_tpg_<year>) as
tpg_line_comparison_map.html, with tpg_line_stops_and_flows.csv and tpg_line_hourly_flows.csv, and the map of the
asymmetries of the lines (tpg_line_asymmetry_map.html, see analysis.pt.TL_comparison.line_asymmetry).

The 2025 TPG counts have no direction: the counts per direction (the station where the runs end) are the
tpg<year>_agg_workday_<date>_by_direction.csv written by analysis.pt.TPG_comparison.tpg_raw_stats_2025."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stages # pyright: ignore[reportMissingImports]
from analysis.pt.TL_comparison import line_comparison as shared


def configure(context):
    # TPG stops are placed at the stops of the GTFS of the simulation
    context.stage("data.gtfs.cleaned")
    context.config("data_path")
    context.config("input_downsampling")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")

    context.config("analysis.pt.tpg_processed_counts_path", default = "Geneva_passenger_counts/TPG/processed")
    context.config("analysis.pt.tpg_year",                  default = 2025)

    # Rebuild the processed TPG counts from the raw ones first (analysis.pt.TPG_comparison.tpg_raw_stats_2025, ~5 GB)
    context.config("analysis.pt.tpg_rebuild_counts", default = False)
    if context.config("analysis.pt.tpg_rebuild_counts"):
        context.stage("analysis.pt.TPG_comparison.tpg_raw_stats_%s" % context.config("analysis.pt.tpg_year"))
    context.config("analysis.pt.night_hours",               default = 5)

    context.config("analysis.pt.line_comparison_output_id", default = None)  # the run of output_id by default
    context.config("analysis.pt.tpg_line_comparison_lines", default = [])    # MATSim line names; empty: all the lines of the TPG data

def execute(context):
    data_path = context.config("data_path")
    year = context.config("analysis.pt.tpg_year")
    output_id = context.config("analysis.pt.line_comparison_output_id") or context.config("output_id")
    simulation_path = os.path.join(context.config("output_path"), output_id, context.config("simulation_directory"))

    lines = context.config("analysis.pt.tpg_line_comparison_lines")
    if isinstance(lines, str):
        lines = [line.strip() for line in lines.split(",") if line.strip()]

    output_dir = shared.run(
        simulation_path, read_tpg_table(os.path.join(data_path, context.config("analysis.pt.tpg_processed_counts_path")), year),
        "%s/output" % context.path("data.gtfs.cleaned"),
        input_downsampling = context.config("input_downsampling"), lines = lines,
        night_hours = context.config("analysis.pt.night_hours"),
        output_dir = os.path.join(simulation_path, f"pt_comparison_tpg_{year}"),
        reference_name = "TPG", prefix = "tpg", elevation_path = os.path.join(data_path, shared.ELEVATION_PATH),
    )

    return dict(done = True, path = output_dir)




def read_tpg_table(processed_dir, year):
    """TPG counts per direction, in the columns of line_comparison.read_tl_table: line, direction, code, hour and the means.
    The counts per direction (_by_direction.csv), or the main table if its line_direction has the direction (2024: _H/_R)."""
    try:
        path = stages._resolve_tpg_stats_path(processed_dir, year, "_by_direction")
    except FileNotFoundError:
        path = stages._resolve_tpg_stats_path(processed_dir, year)
        if not pd.read_csv(path, usecols = ["line_direction"])["line_direction"].astype(str).str.contains("_").all():
            raise FileNotFoundError(
                f"The TPG counts of {path} have no direction: run analysis.pt.TPG_comparison.tpg_raw_stats_2025 again, it "
                f"also writes the counts per direction (tpg{year}_agg_workday_<date>_by_direction.csv)"
            )

    print(f"TPG counts: {path}")
    tpg = pd.read_csv(path)
    split = tpg["line_direction"].astype(str).str.rsplit("_", n = 1)
    tpg["line"]      = split.str[0]
    tpg["direction"] = split.str[1]
    tpg["code"]      = tpg["gtfs_code"].astype(str).str.split(".").str[0]

    return tpg




def main(argv = None):
    parser = argparse.ArgumentParser(description = "Map comparing the MATSim and TPG passengers of the lines")
    parser.add_argument("--simulation-path", required = True, help = "MATSim output folder (pt_passenger_counts.csv.gz and transit schedule)")
    parser.add_argument("--tpg-processed", required = True, help = "Folder with tpg<year>_agg_workday_<date>[_by_direction].csv")
    parser.add_argument("--year", type = int, default = 2025)
    parser.add_argument("--gtfs", required = True, help = "GTFS of the simulated schedule: output folder of data.gtfs.cleaned (or a zip)")
    parser.add_argument("--input-downsampling", type = float, required = True)
    parser.add_argument("--lines", nargs = "*", default = [], help = "MATSim line names (default: all the lines of the TPG data)")
    parser.add_argument("--night-hours", type = int, default = 5)
    parser.add_argument("--output-dir", default = None)
    parser.add_argument("--elevations", default = None, help = "Elevation cache of the pipeline (<data_path>/osm/elevations.csv)")
    args = parser.parse_args(argv)

    shared.run(args.simulation_path, read_tpg_table(args.tpg_processed, args.year), args.gtfs, args.input_downsampling, args.lines,
               args.night_hours, args.output_dir, reference_name = "TPG", prefix = "tpg", elevation_path = args.elevations)


if __name__ == "__main__":
    main()
