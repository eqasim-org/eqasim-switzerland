import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

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

    context.config("analysis.pt.include_lemanis",  default = False)
    context.config("analysis.pt.lemanis_csv_path", default = None)


def execute(context):
    data_path = context.config("data_path")

    lemanis_csv_path = context.config("analysis.pt.lemanis_csv_path")
    include_lemanis  = context.config("analysis.pt.include_lemanis")

    if include_lemanis and not lemanis_csv_path:
        raise RuntimeError(
            "analysis.pt.include_lemanis is true but analysis.pt.lemanis_csv_path is not set"
        )

    matsim_output_folder = os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory")
    )
    
    if not os.path.isdir(matsim_output_folder):
        raise FileNotFoundError(
            f"runCutter MATSim output not found at {matsim_output_folder} - "
            "has the full pipeline (including matsim.output) been run for this output_id?"
        )

    cfg = Config(
        gtfs_zip                  = os.path.join(data_path, context.config("analysis.pt.gtfs_zip")),
        perimeter_shapefile       = os.path.join(data_path, context.config("analysis.pt.perimeter")),
        tpg_data_path             = os.path.join(data_path, context.config("analysis.pt.tpg_data")),
        matsim_output_folder      = matsim_output_folder,
        output_path               = context.config("output_path"),
        tpg_processed_counts_path = os.path.join(data_path, context.config("analysis.pt.tpg_processed_counts_path")),
        input_downsampling        = context.config("input_downsampling"),
        min_stop_avg_events       = context.config("analysis.pt.min_stop_avg_events"),
        min_stop_active_hours     = context.config("analysis.pt.min_stop_active_hours"),
        map_hour_start            = context.config("analysis.pt.map_hour_start"),
        map_hour_end              = context.config("analysis.pt.map_hour_end"),
        include_lemanis           = include_lemanis,
        lemanis_csv_path          = lemanis_csv_path,
    )

    year       = context.config("analysis.pt.year")
    output_dir = os.path.join(matsim_output_folder, f"pt_comparison_tpg_{year}")

    print(f"Running global (perimeter-wide) PT comparison for {year} -> {output_dir}")
    stages.run_global_comparison(cfg, output_dir, year)

    if context.config("analysis.pt.stop_line_comparison"):
        stop = context.config("analysis.pt.stop")
        line = context.config("analysis.pt.line")
        if line is None:
            line = "1_H" if year == 2024 else "1"

        print(f"Running {year} stop/line PT comparison for stop={stop!r} line={line!r} -> {output_dir}")
        stages.run_stop_line_comparison(cfg, output_dir, year, stop=stop, line=line)

    return dict(done = True, path = output_dir)
