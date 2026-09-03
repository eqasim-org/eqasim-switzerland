import logging
import os
import pandas as pd
import geopandas as gpd
import json
from .run_utils import (filter_data, compute_statistics, save_as_target,
                        create_comprehensive_plot, create_simple_scatter_plot, plot_by_road_cat,
                        print_detailed_statistics, get_average_flow_veh_h_by_category)
from .matching.plots import Plotter
from .paths import configure_simulation_path, get_analysis_output_path


logger = logging.getLogger("synpp")
runs = [i.split('.')[0] for i in os.listdir("analysis/counts/runs") if not (i.startswith("_") or i.startswith("."))]

def configure(context):    
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")
    configure_simulation_path(context)
    context.config("only_weekday", default=False)
    for run in runs:
        logger.info(f"Staging analysis.counts.runs.{run}")
        context.stage(f"analysis.counts.runs.{run}")

def execute(context):
    # Get the path to output
    path_to_output = get_analysis_output_path(context)
    
    os.makedirs(path_to_output, exist_ok=True)

    # Load all count files
    files = {}
    for run in runs:
        if 'annemasse' not in run.lower():
            files[run] = context.stage(f"analysis.counts.runs.{run}")

    # Combine all count files into a single file
    all_data = []
    for canton, file in files.items():
        if file is None:
            logger.warning(f"\t - {canton}: No count data found, skipping.")
            continue
        if not os.path.exists(file):
            logger.warning(f"\t - {canton}: Count data file {file} does not exist, skipping.")
            continue

        df = pd.read_pickle(file)        
        logger.info(f"\t - {canton}: {len(df)} records ({canton})")
        all_data.append(df)

    if not all_data:
        logger.warning("No count locations matched the simulation network; skipping combined analysis.")
        return dict(done=True, path=path_to_output)

    df = pd.concat(all_data, ignore_index=True)
    if df.empty:
        logger.warning("Matched count result files are empty; skipping combined analysis.")
        return dict(done=True, path=path_to_output)

    logger.info(f"\n\tCombined dataset: {len(df)} total records")
    logger.info(f"\tCities included: {', '.join(df['city'].unique())}")
    
    # filter outliers
    network = context.stage("analysis.counts.matching.network")
    df = filter_data(df, network)
    if df is None or df.empty:
        logger.warning("No count records remain after filtering; skipping combined analysis.")
        return dict(done=True, path=path_to_output)

    # Compute statistics
    stats = compute_statistics(df, output_path=path_to_output)

    # print and save detailed statistics
    print_detailed_statistics(stats)

    # plot comprehensive comparison
    create_comprehensive_plot(df, stats, output_path=path_to_output)

    # plot simple scatter plot
    create_simple_scatter_plot(df, stats, output_path=path_to_output)

    # Plot by road category
    plot_by_road_cat(df, output_path=path_to_output)

    # Average flow in vehicles per hour per lane by road category
    get_average_flow_veh_h_by_category(df, output_path=path_to_output)
    
    # save the data (this can be used as target for network calibration)
    save_as_target(network, df, path_to_output)

    # Plot the map with the remaining stations (after filters), showing absolute and relative differences
    roads_to_show = ['motorway', 'trunk', 'primary', 'motorway_link', 'trunk_link', 'primary_link']
    border = gpd.GeoDataFrame(context.stage("data.spatial.swiss_border").to_crs(epsg=4326))
    network_ways = network.get_ways(road_types = roads_to_show).to_crs(epsg=4326)
    points = Plotter.prepare_flow_map_points(
        gpd.GeoDataFrame(
            df[["id", "geometry"]], geometry="geometry", crs="EPSG:2056"
        ),
        df,
    ).to_crs(epsg=4326)

    Plotter.create_map([network_ways],
                        data_to_show=["link_id"],
                        point_gdf=[points],
                        point_data_to_show=Plotter.FLOW_MAP_TOOLTIP_FIELDS,
                        border=border,
                        path_to_save=os.path.join(path_to_output, "Switzerland_counts_comparaison.html"))


    logger.info("\n Analysis completed successfully!")
    return dict(done=True, path = path_to_output)