import logging
import os
import pandas as pd
import json
from .run_utils import (filter_data, compute_statistics,
                        create_comprehensive_plot, plot_by_road_cat,
                        print_detailed_statistics, get_average_flow_veh_h_by_category)


logger = logging.getLogger("synpp")
runs = [i.split('.')[0] for i in os.listdir("analysis/counts/runs") if not i.startswith("_")]

def configure(context):    
    context.stage("analysis.counts.matching.network")
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

    # Plot by road category
    plot_by_road_cat(df, output_path=path_to_output)

    # Average flow in vehicles per hour per lane by road category
    get_average_flow_veh_h_by_category(df, output_path=path_to_output)
    
    # save the data (this can be used as target for network calibration)
    df = df[['link_id','obs_vphpl']]
    df = df.explode("link_id")
    df = df.rename(columns={"link_id":"linkId",
                            "obs_vphpl":"count"})
    df.to_csv(os.path.join(path_to_output, "target_flow.csv"), index=False, sep=",")
    
    logger.info("\n Analysis completed successfully!")

    return dict(done=True, path = path_to_output)