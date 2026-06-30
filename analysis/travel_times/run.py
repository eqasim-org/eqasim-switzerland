import os
import logging

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from analysis.travel_times.plot_utils import (plot_scatter, plot_boxplot, plot_distribution,
                                              plot_average_by_distance_bin,
                                              plot_by, plot_boxplot_by,
                                              plot_link_error_on_network)

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.travel_times.APIs.target")
    context.stage("analysis.travel_times.APIs.get")
    context.stage("analysis.travel_times.matsim.get")
    context.stage("analysis.travel_times.matsim.route")
    context.stage("analysis.travel_times.advanced.process")
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("distance_bin_km", default=8.0)
    context.config("travel_times_from", default="tomtom")


def merge_and_filter_large_differences(df_api, df_matsim):
    # merge dataframes on identifiers
    df = pd.merge(df_api, df_matsim, on="identifier", suffixes=('_api', '_matsim'))
    assert (df["departure_time_matsim"]==df["departure_time_api"]).all(), "Departure times do not match between API and MATSim data"
    assert (np.abs(df["euclidean_distance_km_api"]-df["euclidean_distance_km_matsim"]) < 1e-3).all(), "Euclidean distances do not match between API and MATSim data"

    # Filter out rows with large distance difference (this might be due to network differences, maybe missing links)
    distance_diff = np.abs((df["distance_km_api"] - df["distance_km_matsim"])/df["distance_km_api"]) * 100.0
    distance_diff_threshold = 20.0  # percent
    n_before = len(df)
    df = df[distance_diff <= distance_diff_threshold].reset_index(drop=True).copy()
    n_after = len(df)
    logger.info(f"Filtered out {n_before - n_after} rows with distance difference > {distance_diff_threshold} %")

    # Filter out rows with high access or egress distances
    access_egress_threshold_km = 0.2  # km
    n_before = len(df)
    df = df[(df["access_distance_km"] <= access_egress_threshold_km) & 
            (df["egress_distance_km"] <= access_egress_threshold_km)].reset_index(drop=True).copy()
    n_after = len(df)
    logger.info(f"Filtered out {n_before - n_after} rows with access/egress distance > {access_egress_threshold_km} km")

    # filter out rows with very high travel time differences (this might happend because of one link overcongested in the simulation, which is an outlier)
    travel_time_threshold_min = 30.0 # this happen in very few points
    n_before = len(df)
    df = df[abs(df["travel_time_min_api"] - df["travel_time_min_matsim"]) <= travel_time_threshold_min].reset_index(drop=True).copy()
    n_after = len(df)
    logger.info(f"Filtered out {n_before - n_after} rows with high travel times discrepencies > {travel_time_threshold_min} min")


    return df

def execute(context):
    # Save travel times as csv in the output dir
    _ = context.stage("analysis.travel_times.APIs.target")
    
    # Load data from APIs and MATSim
    dfs_api = context.stage("analysis.travel_times.APIs.get")
    dfs_matsim = context.stage("analysis.travel_times.matsim.get")
    _, routed_paths = context.stage("analysis.travel_times.matsim.route")
    network = context.stage("analysis.counts.matching.network")
    swiss_border = context.stage("data.spatial.swiss_border")
    _ = context.stage("analysis.travel_times.advanced.process")
    
    # For each API, compare with MATSim data
    out_folders = []
    for api, df_api in dfs_api.items():
        # get corresponding MATSim dataframe
        df_matsim = dfs_matsim[api]
        
        # paths    
        out = os.path.join(context.config("output_path"), 
                                context.config("output_id"), 
                                context.config("simulation_directory"),
                                "travel_times_"+api)    
        os.makedirs(out, exist_ok=True)
        out_folders.append(out)

        # Merge and filter large differences
        df = merge_and_filter_large_differences(df_api, df_matsim)

        # Keep routed link paths for network error visualization
        routed_links = pd.read_csv(routed_paths[api], usecols=["identifier", "links"])
        df_links = df.merge(routed_links, on="identifier", how="left")

        # Plots
        bin_km = float(context.config("distance_bin_km"))

        # scatter plot
        stats_scatter = plot_scatter(
            x=df["travel_time_min_api"],
            y=df["travel_time_min_matsim"],
            title="Travel Time: MATSim vs. "+api.upper(),
            xlabel=api.upper()+" Travel Time (min)",
            ylabel="MATSim Travel Time (min)",
            out_path=os.path.join(out, "scatter_matsim_vs_"+api+".png")
        )
        logger.info(f"Scatter plot stats: {stats_scatter}")

        # boxplot of differences
        stats_boxplot = plot_boxplot(
            x=df["travel_time_min_api"],
            y=df["travel_time_min_matsim"],
            title="Travel Time Difference: MATSim minus "+api.upper(),
            xlabel="",
            out_path=os.path.join(out, "boxplot_diff_matsim_vs_"+api+".png")
        )
        logger.info(f"Boxplot difference stats: {stats_boxplot}")

        # binned average travel times
        plot_average_by_distance_bin(
            tt1=df["travel_time_min_api"],
            tt2=df["travel_time_min_matsim"],
            dist=df["euclidean_distance_km_matsim"],
            bin_km=bin_km,
            title="Average Travel Time by Euclidean Distance Bin: MATSim vs. "+api.upper(),
            source1=api.upper(),
            source2="MATSim",
            xlabel="Euclidean distance bin mid (km)",
            ylabel="Travel time (min)",
            out_path=os.path.join(out, "binned_travel_times_matsim_vs_"+api+".png")
        )

        # binned average distance
        plot_average_by_distance_bin(
            tt1=df["distance_km_api"],
            tt2=df["distance_km_matsim"],
            dist=df["euclidean_distance_km_matsim"],
            bin_km=bin_km,
            title="Average Distance by Euclidean Distance Bin: MATSim vs. "+api.upper(),
            source1=api.upper(),
            source2="MATSim",
            xlabel="Euclidean distance bin mid (km)",
            ylabel="Routed distance (km)",
            out_path=os.path.join(out, "binned_distances_matsim_vs_"+api+".png")
        )

        # plot average time by departure hour
        df["departure_hour"] = (df["departure_time_api"]%86400)//3600
        plot_by(
            df=df,
            by="departure_hour",
            value1="travel_time_min_api",
            value2="travel_time_min_matsim",
            title="Average Travel Time by Departure Hour: MATSim vs. "+api.upper(),
            source1=api.upper(),
            source2="MATSim",
            xlabel="Departure hour of day",
            ylabel="Average travel time (min)",
            between = [3,24],
            out_path=os.path.join(out, "binned_travel_times_by_departure_hour_matsim_vs_"+api+".png")
        )
        
        # plot average time by departure hour for long distances
        mask = df["euclidean_distance_km_matsim"] >= 20.0
        plot_by(
            df=df[mask].reset_index(drop=True),
            by="departure_hour",
            value1="travel_time_min_api",
            value2="travel_time_min_matsim",
            title="Average Travel Time by Departure Hour: MATSim vs. "+api.upper(),
            source1=api.upper(),
            source2="MATSim",
            xlabel="Departure hour of day",
            ylabel="Average travel time (min)",
            between = [3,24],
            out_path=os.path.join(out, "binned_travel_times_by_departure_hour_matsim_vs_"+api+"_long_distances.png")
        )

        # plot average speed by departrue hour
        df["average_speed_kmh_api"] = df["distance_km_api"] / (df["travel_time_min_api"] / 60.0)
        df["average_speed_kmh_matsim"] = df["distance_km_matsim"] / (df["travel_time_min_matsim"] / 60.0)
        plot_by(
            df=df,
            by="departure_hour",
            value1="average_speed_kmh_api",
            value2="average_speed_kmh_matsim",
            title="Average Speed by Departure Hour: MATSim vs. "+api.upper(),
            source1=api.upper(),
            source2="MATSim",
            xlabel="Departure hour of day",
            ylabel="Average speed (km/h)",
            between = [3,24],
            out_path=os.path.join(out, "binned_average_speed_by_departure_hour_matsim_vs_"+api+".png")
        )
        
        # boxplot by departure hour
        plot_boxplot_by(
            df=df,
            by="departure_hour",
            value1="travel_time_min_api",
            value2="travel_time_min_matsim",
            title="Travel Time Difference by Departure Hour: MATSim minus "+api.upper(),
            xlabel="Departure hour of day",
            out_path=os.path.join(out, "boxplot_diff_by_departure_hour_matsim_vs_"+api+".png"),
            between = [6,22]
        )

        # plot distribution of Euclidean distances
        df["departure_hour"] = df["departure_hour"].astype(int)
        plot_distribution(
            df=df,
            by="departure_hour",
            value="euclidean_distance_km_matsim",
            title="Distribution of Trips Euclidean Distances",
            xlabel="Euclidean distance (km)",
            out_path=os.path.join(out, "distribution_euclidean_distances_matsim_vs_"+api+".png")
        )

        # network map of over/underestimation, aggregated by traversed links
        # plot_link_error_on_network(
        #     network_gdf=network.net_geo,
        #     routed_df=df_links,
        #     swiss_border=swiss_border,
        #     title="Link-level travel-time error: MATSim vs " + api.upper(),
        #     out_path=os.path.join(out, "network_link_error_matsim_vs_" + api + ".png")
        # )

    return dict(done = True, path = out_folders)