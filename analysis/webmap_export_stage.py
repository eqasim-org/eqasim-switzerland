import os
import gzip
import shutil
import pandas as pd
import geopandas as gpd
import analysis.webmap_export as webmap_export
from analysis.network_reader import read_network
from analysis.matsim_pt_analysis import generate_source_destination_data
from analysis.matsim_pt_analysis import add_canton_to_pt_passenger, create_boarding_json
from analysis.synth_micro_comparison import *
from analysis.process_transfer_data import get_transfer_matrix_data

def configure(context):
    context.stage("matsim.simulation.run")  # get working directory
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons") # get canton boundaries
    context.config("working_directory")
    context.config("output_path")

# def fix_mojibake(s):
#     if s is None:
#         return s
#     if isinstance(s, bytes):
#         # If somehow bytes slipped through, decode properly first
#         try:
#             return s.decode("utf-8")
#         except UnicodeDecodeError:
#             return s.decode("latin1", errors="replace")

#     # Heuristic: try latin1→utf8 roundtrip if it looks like mojibake
#     try:
#         repaired = s.encode("latin1").decode("utf-8")
#         # Only accept if it actually improved the text
#         if "Ã" in s or "Â" in s or repaired != s:
#             return repaired
#     except UnicodeEncodeError:
#         pass
#     return s

def execute(context):
    matsim_dir = context.stage("matsim.simulation.run")
    simulation_output = os.path.join(matsim_dir, "simulation_output")

    # Create webmap output directory
    output_dir = os.path.join(matsim_dir, "simulation_output", "webmap")
    os.makedirs(output_dir, exist_ok=True)
    webmap_export.DEFAULT_WORKDIR = output_dir

    # Create all necessary subfolders
    os.makedirs(os.path.join(output_dir, "public", "data", "matsim", "transit"), exist_ok=True)

    # Input files
    synthetic_gz_path = os.path.join(simulation_output, "output_trips.csv.gz")

    ## temp set iterations to 1 for testing (final should be 50)
    linkstats_gz_path = os.path.join(simulation_output, "ITERS", "it.1", "1.linkstats.txt.gz")
    linkstats_path = os.path.join(output_dir, "1.linkstats.txt")
    linkstats_gz_path = os.path.join(simulation_output, "ITERS", "it.1", "1.linkstats.txt.gz")
    linkstats_path = os.path.join(output_dir, "50.linkstats.txt")
    matsim_network_path = os.path.join(simulation_output, "output_network.xml.gz")
    transit_schedule_path = os.path.join(simulation_output, "output_transitSchedule.xml.gz")
    volumes_path = os.path.join(simulation_output, "pt_passenger_counts.csv.gz")
    pt_legs_path = os.path.join(simulation_output, "output_legs.csv.gz")

    # Microcensus files (fixed path)
    microcensus_trips_df = context.stage("data.microcensus.trips")[0]  # first element of tuple
    microcensus_persons_df = context.stage("data.microcensus.persons")

    # Cantons
    canton_boundaries = context.stage("data.spatial.cantons")
    #canton_boundaries["canton_name"] = canton_boundaries["canton_name"].apply(fix_mojibake)
    # === Unzip if needed ===
    if not os.path.exists(linkstats_path):
        with gzip.open(linkstats_gz_path, 'rb') as f_in, open(linkstats_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # === Add canton names to synthetic data if missing ===
    synthetic_df = pd.read_csv(synthetic_gz_path, sep=';', compression='gzip', dtype={"person": str})

    # For synthetic
    if "canton_name" not in synthetic_df.columns:
        print("Assigning cantons to synthetic data...")
        synthetic_df = webmap_export.assign_cantons(synthetic_df, canton_boundaries)

    # For microcensus
    if "canton_name" not in microcensus_trips_df.columns:
        print("Assigning cantons to microcensus data...")
        microcensus_trips_df = webmap_export.assign_cantons(microcensus_trips_df, canton_boundaries, "origin_x", "origin_y")

    # === Load microcensus and synthetic data ===
    syn_df, mc_df = webmap_export.import_data(synthetic_df, microcensus_trips_df, microcensus_persons_df)

    # commented to make testing faster.
    for agg_col in ["mode", "purpose"]:
        webmap_export.export_by_aggregation(mc_df, syn_df, aggregation_col=agg_col)

    print("Reading network...")
    network = read_network(matsim_network_path)
    geo = network.as_geo(projection="EPSG:2056")

    print("Exporting merged network segments per canton...")
    webmap_export.export_merged_segments_by_canton(
        network_gdf=geo,
        cantons_gdf=canton_boundaries,
        linkstats_path=linkstats_path,
        output_dir=os.path.join(output_dir, "public", "data", "matsim"),
        skip_cantons=None,
        id_col="link_id",
        canton_name_col="canton_name",
        network_modes_col="modes",
        target_crs="EPSG:4326",
        write_link_hourly_json=True,
    )

    print("Parsing transit stops from schedule XML...")
    stops_gdf = webmap_export.parse_stops(transit_schedule_path)
    stops_gdf = stops_gdf.to_crs(epsg=2056)
    print(f"Parsed {len(stops_gdf)} stops")

    print("Assigning each stop to a canton...")
    joined = webmap_export.assign_cantons_stops(stops_gdf, canton_boundaries)

    print("Exporting per-canton stop GeoJSONs...")
    stops_dir = webmap_export.export_per_canton_stops(joined)

    print("Generating modes_by_canton.json...")
    webmap_export.generate_modes_by_canton(joined)

    print("Building route line geometries...")
    webmap_export.build_route_lines(transit_schedule_path, joined)

    print("Computing passenger counts per canton...")
    volumes_df = pd.read_csv(volumes_path, sep=',', compression='gzip')
    webmap_export.compute_passenger_counts(joined, volumes_df)

    print("Exporting inter-cantonal stops with volume...")
    webmap_export.export_inter_cantonal_stops(joined, volumes_df)


    # === Generate plots for comparing activities from microcensus and synthetic datasets
    microcensus_directory = context.config('working_directory')
    synthetic_directory = context.config("output_path")
    save_directory = os.path.join(output_dir, "public", "data")
    os.makedirs(save_directory, exist_ok=True)
    generate_microcensus_synthetic_comparison(microcensus_directory, synthetic_directory, canton_boundaries, save_directory)
   
    # === Additional functionality from matsim_destination_zones.py ===
    generate_source_destination_data(synthetic_gz_path, work_dir=output_dir, canton_boundaries=canton_boundaries)

    # === Additional functionality from canton_pt_lines.py ===
    stops_dir = os.path.join(output_dir, "public", "data", "matsim", "transit", "stops_by_canton")
    df_with_cantons = add_canton_to_pt_passenger(volumes_path, stops_dir)
    create_boarding_json(df_with_cantons, output_dir)

    # === Additional plots for looking at transfers between PT stops ===
    get_transfer_matrix_data(data_path=pt_legs_path, output_dir=output_dir, stops_dir=stops_dir)
    
    print("Webmap export complete. Output saved to:", output_dir)