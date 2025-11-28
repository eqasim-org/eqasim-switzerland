import pandas as pd
import os
import numpy as np
import matsim.runtime.eqasim as eqasim
import geopandas as gpd

def configure(context):
    context.stage("pt_preparation.pt_pricing.run_price_estimation_from_uspat")


def execute(context):
    requests, results = context.stage("pt_preparation.pt_pricing.run_price_estimation_from_uspat")

    zone_info = requests[["ID", "origin_zone", "destination_zone"]].rename(columns = {"ID":"id"})

    results = results.merge(zone_info)

    agg_results_total_time    = results.groupby(["origin_zone", "destination_zone"])["total_travel_time_min"].mean().round(2).reset_index()
    agg_results_vehicle_time  = results.groupby(["origin_zone", "destination_zone"])["in_vehicle_time_min"].mean().round(2).reset_index()
    agg_results_access_time   = results.groupby(["origin_zone", "destination_zone"])["access_egress_time_min"].mean().round(2).reset_index()
    agg_results_waiting_time  = results.groupby(["origin_zone", "destination_zone"])["waiting_time_min"].mean().round(2).reset_index()
    agg_results_transfers     = results.groupby(["origin_zone", "destination_zone"])["number_of_line_switches"].mean().round(0).reset_index()
    agg_results_price         = results.groupby(["origin_zone", "destination_zone", "hasHalbtaxSubscription"])["price"].mean().round(2).reset_index()
    agg_results_dist          = results.groupby(["origin_zone", "destination_zone"])["networkDistance"].mean().round(2).reset_index()

    agg_results_price_pivot = (
        agg_results_price
            .pivot(
                index=["origin_zone", "destination_zone"],
                columns="hasHalbtaxSubscription",
                values="price"
            )
            .reset_index()
            .rename(columns={
                False: "price_no_halbtax",
                True: "price_halbtax"
            })
    )

    agg_results = agg_results_total_time.copy()
    for df in [agg_results_vehicle_time, agg_results_access_time, agg_results_waiting_time, agg_results_transfers, agg_results_price_pivot, agg_results_dist]:
        agg_results = agg_results.merge(df, on = ["origin_zone", "destination_zone"], how = "outer")

    #agg_results[~(agg_results["total_travel_time_min"].isna())].to_csv("/cluster/project/cmdp/asallard/eqasim-VD/Data/skim_matrices_20251127/vdgefrbenevs.csv", index=False)

    return agg_results[~(agg_results["total_travel_time_min"].isna())]


