import os
import logging

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from analysis.travel_times.plot_utils import (plot_scatter, plot_boxplot, 
                                              plot_average_by_distance_bin,
                                              plot_by, plot_boxplot_by)

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.travel_times.APIs.get")
    context.stage("analysis.travel_times.matsim.get")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default = "simulation_output")
    context.config("distance_bin_km", default=8.0)
    context.config("travel_times_from", default="tomtom")

def execute(context):
    # path
    api = context.config("travel_times_from")
    out = os.path.join(context.config("output_path"), 
                            context.config("output_id"), 
                            context.config("simulation_directory"),
                            "travel_times_"+api)    
    os.makedirs(out, exist_ok=True)

    # Load data from APIs and MATSim
    df_api = context.stage("analysis.travel_times.APIs.get")
    df_matsim = context.stage("analysis.travel_times.matsim.get")

    # merge dataframes on identifiers
    df = pd.merge(df_api, df_matsim, on="identifier", suffixes=('_api', '_matsim'))
    assert (df["departure_time_matsim"]==df["departure_time_api"]).all(), "Departure times do not match between API and MATSim data"
    assert (np.abs(df["euclidean_distance_km_api"]-df["euclidean_distance_km_matsim"]) < 1e-3).all(), "Euclidean distances do not match between API and MATSim data"
    
    # Filter out rows with large distance difference (this might be due to network differences, maybe missing links)
    distance_diff = np.abs((df["distance_km_api"] - df["distance_km_matsim"])/df["distance_km_api"]) * 100.0
    distance_diff_threshold = 25.0  # percent
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
    logger.info(f"Filtered out {n_before - n_after} rows with access/egress distance > {access_egress_threshold_km} m")

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
        between = [6,22],
        out_path=os.path.join(out, "binned_travel_times_by_departure_hour_matsim_vs_"+api+".png")
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

    return dict(done = True, path = out)