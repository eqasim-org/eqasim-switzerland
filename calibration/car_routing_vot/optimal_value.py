import pandas as pd
import os
import logging
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from .server import start_server, route_trips, set_server_config, stop_server
logger = logging.getLogger("synpp")

def configure(context):
    context.config("calibrate_routing_vot", default=False)
    context.config("events_file_for_routing_vot_calibration", default=None)
    context.config("network_file_for_routing_vot_calibration", default=None)

    if context.config("calibrate_routing_vot"):
        context.stage("calibration.car_routing_vot.dataset")
        context.stage("matsim.runtime.eqasim")
        context.stage("matsim.runtime.java")
        context.config("threads")

def execute(context):
    if not context.config("calibrate_routing_vot"):
        return 0.42

    if context.config("events_file_for_routing_vot_calibration") is None:
        raise ValueError("events_file_for_routing_vot_calibration must be provided for routing VOT calibration")

    if context.config("network_file_for_routing_vot_calibration") is None:
        raise ValueError("network_file_for_routing_vot_calibration must be provided for routing VOT calibration")

    df = context.stage("calibration.car_routing_vot.dataset")
    
    #compute distance bins
    distance_bins = [0, 2000, 3500, 5000, 7000, 9000, 12000, 16000, 20000, 30000, 50000, float("inf")]
    distance_bin_labels = ["0-2km", "2-3.5km", "3.5-5km","5-7km", "7-9km", "9-12km", "12-16km","16-20km", "20-30km", "30-50km", "50km+"]
    df["distance_bin"] = pd.cut(df["euclidean_distance"], bins=distance_bins, labels=distance_bin_labels)

    # rename columns for matsim routing
    df = transform_dataframe_for_routing(df)

    try:
        # starting the serve for routing
        server_process = start_server(
            context,
            network_path=context.config("network_file_for_routing_vot_calibration"),
            events_path=context.config("events_file_for_routing_vot_calibration"),
            port=8080,
            interval=900,
            routing_distance_utility=0.0
        )
        # compute optimal value through optimization  
        optimal_value = find_optimal_value(context, df)
    
    finally:
        stop_server(server_process)

    return optimal_value









################################ Calibration functions ################################
def find_optimal_value(context, df):
    logger.info("\t Starting optimization of routing_distance_utility...")
    # Prepare trip records for the server
    matsim_cols = ['identifier', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']
    trips_list = df[matsim_cols].to_dict(orient='records')
    _objective = lambda value: objective(context, df, trips_list, value)
    result = minimize_scalar(_objective, bounds=(0.0, 2.0), method='bounded', tol = 1e-3,
                             options={'disp': True, 'maxiter': 50})
    optimal = round(result.x, 3)
    logger.info(f"\t - Optimal routing_distance_utility: {optimal:.6f} (loss={result.fun:.6f})")
    return optimal

def transform_dataframe_for_routing(df):
    df = df.copy()
    df["identifier"] = (df["participant_id"].astype(str) + "_" + df["leg_id"].astype(str)).astype(str)
    df["departure_time"] = (df["started_at"].dt.hour * 3600 + df["started_at"].dt.minute * 60 + df["started_at"].dt.second).astype(int)    
    df = df.rename(columns={        
        "start_x": "origin_x",
        "start_y": "origin_y",
        "end_x": "destination_x",
        "end_y": "destination_y",        
    })    
    return df

def remove_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def compute_loss(context, df, value):
    df = df.copy()
    # compute detour factors
    df["detour_factor"] = df["length"] / df["euclidean_distance"]
    df["matsim_detour_factor"] = df["travel_distance"] / df["euclidean_distance"]
    #group by distance bin and compute average detour factor
    detour_factors = df.groupby("distance_bin")[["detour_factor", "matsim_detour_factor"]].median()
    # compute loss as mean absolute error between detour factors
    detour_factors["loss"] = (detour_factors["detour_factor"] - detour_factors["matsim_detour_factor"]).abs()
    loss = detour_factors["loss"].mean()
    logger.info(f"\t\t-[DETOUR FACTOR OPTIMIZATION] : routing_distance_utility={value:.3f}:{loss:.3f}")
    plot_figure(context, detour_factors, value)
    return loss

def objective(context, df, trips_list, value):
    set_server_config(routing_distance_utility=value)
    routed = route_trips(trips_list)
    # handle server response: list or {"trips": [...]}
    if isinstance(routed, dict):
        routed = routed.get("trips", list(routed.values())[0])
    df_routed = pd.DataFrame(routed)
    # we only keep trips with short access and egress distances to avoid noise in the calibration
    df_routed = df_routed[(df_routed["access_distance"]<100)&(df_routed["egress_distance"]<100)]
    df_merged = df.merge(df_routed[["identifier", "travel_distance"]], on="identifier", how="inner")
    # compute the loss for this value
    loss = compute_loss(context, df_merged, value)
    logger.info(f"\t - routing_distance_utility={value:.6f}, loss={loss:.6f}")
    return loss


##### plotting the figure
def plot_figure(context, detour_factors, value):    
    plt.figure(figsize=(10,6))
    plt.plot(detour_factors.index.astype(str), detour_factors["detour_factor"], label="Observed Detour Factor", marker='o')
    plt.plot(detour_factors.index.astype(str), detour_factors["matsim_detour_factor"], label="MATSIM Detour Factor", marker='o')
    plt.xlabel("Distance Bin")
    plt.ylabel("Median Detour Factor")
    plt.title("Detour Factor by Distance Bin")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    figure_path = os.path.join(context.path(), f"detour_factor_calibration_{value:.3f}.png")
    plt.savefig(figure_path)    