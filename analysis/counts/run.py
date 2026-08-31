import logging
import os
import pandas as pd
import geopandas as gpd
import json
from .run_utils import (filter_data, compute_statistics, save_as_target,
                        create_comprehensive_plot, create_simple_scatter_plot, plot_by_road_cat,
                        print_detailed_statistics, get_average_flow_veh_h_by_category)
from .matching.plots import Plotter


logger = logging.getLogger("synpp")
runs = [i.split('.')[0] for i in os.listdir("analysis/counts/runs") if not (i.startswith("_") or i.startswith("."))]

def configure(context):    
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("only_weekday", default=False)
    for run in runs:
        logger.info(f"Staging analysis.counts.runs.{run}")
        context.stage(f"analysis.counts.runs.{run}")

def execute(context):
    # Get the path to output
    path_to_output = os.path.join(context.config("output_path"), 
                            context.config("output_id"), 
                            context.config("simulation_directory"),
                            "compare_counts_weekdays" if context.config("only_weekday") else "compare_counts_all_days")
    
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

    df = pd.concat(all_data, ignore_index=True)
    logger.info(f"\n\tCombined dataset: {len(df)} total records")
    logger.info(f"\tCities included: {', '.join(df['city'].unique())}")
    
    # filter outliers
    network = context.stage("analysis.counts.matching.network")
    df = filter_data(df, network)

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
    points = gpd.GeoDataFrame(df[['id', 'geometry', 'pdiff', 'adiff']], geometry='geometry', crs='EPSG:2056').to_crs(epsg=4326)

    Plotter.create_map([network_ways],
                        data_to_show=["link_id"],
                        point_gdf=[points[['id', 'geometry', 'pdiff', 'adiff']]],
                        point_data_to_show=['id', 'pdiff','adiff'],
                        border=border,
                        path_to_save=os.path.join(path_to_output, "Switzerland_counts_comparaison.html"))


    logger.info("\n Analysis completed successfully!")
    return dict(done=True, path = path_to_output)