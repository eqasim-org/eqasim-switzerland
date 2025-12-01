import numpy as np
import pandas as pd
import os

"""
This stage reads trip data from microcensus and computes modal shares.
It estimates global shares first, then shares by canton.
"""

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons")

def execute(context):
    trips_data, filtered_out_person_ids = context.stage("data.microcensus.trips")
    persons_data = context.stage("data.microcensus.persons")

    # Filter out excluded persons from trips
    valid_trips_mask = ~trips_data["person_id"].isin(filtered_out_person_ids)
    filtered_trips = trips_data.loc[valid_trips_mask, [
        'person_id', 'trip_id', 'mode', 'crowfly_distance', 'network_distance'
    ]]

    # Filter out excluded persons and weekend trips from persons data
    valid_persons_mask = (
        (~persons_data["person_id"].isin(filtered_out_person_ids)) & 
        (persons_data["weekend"] == False)
    )
    filtered_persons = persons_data.loc[valid_persons_mask, [
        'person_id', 'person_weight', 'age', 'age_class', 'sp_region',
        'sex', 'income_class', 'canton_id', 'household_weight'
    ]]

    # Merge trips with person attributes and weights
    trips_with_weights = filtered_trips.merge(filtered_persons, how="left", on="person_id")
    
    # Define valid transportation modes
    valid_modes = ["car", "pt", "bike", "walk", "car_passenger"]
    
    # Apply final filtering criteria
    final_filter_mask = (
        (trips_with_weights.household_weight.notna()) & 
        (trips_with_weights.person_weight.notna()) &
        (trips_with_weights.crowfly_distance > 1) &
        (trips_with_weights["mode"].isin(valid_modes))
    )
    
    final_trips = trips_with_weights[final_filter_mask].reset_index(drop=True)

    # Attach canton name
    df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name"]].copy()
    
    df_cantons["canton_id"] = df_cantons["canton_id"].astype(int)
    final_trips["canton_id"] = final_trips["canton_id"].astype(int)

    final_trips = pd.merge(final_trips, df_cantons, on="canton_id",  how="left")
    assert final_trips.canton_name.notnull().all(), "Not all persons have a canton name assigned. Check the canton data."

    # Compute global modal split shares
    total_weighted_trips = final_trips['person_weight'].sum()
    global_mode_weights = (
        final_trips.groupby('mode')['person_weight']
        .sum()
        .reset_index(name='total_weight')
    )
    global_mode_weights['mode_share'] = (global_mode_weights['total_weight'] / total_weighted_trips).round(4)
    global_modal_shares = global_mode_weights[["mode", "mode_share"]]

    # Compute cantonal modal split shares
    cantonal_modal_shares = (
        final_trips
        .groupby(["canton_name", "mode"], observed=False)["person_weight"]
        .sum()
        .groupby(level="canton_name", observed=False)
        .transform(lambda x: x / x.sum())
        .rename("mode_share")
        .reset_index()
        .pivot(index='canton_name', columns="mode", values="mode_share")
        .fillna(0)
        .round(3)
        .sort_values(by='canton_name')
    )

    # Define output file paths
    global_shares_output_path = os.path.join(context.path(), "globalModeShares.csv")
    cantonal_shares_output_path = os.path.join(context.path(), "cantonalModeShares.csv")

    # Save global modal shares
    global_modal_shares.to_csv(global_shares_output_path, index=False)

    # Save cantonal modal shares
    cantonal_modal_shares.to_csv(cantonal_shares_output_path, index=True)

    return global_shares_output_path, cantonal_shares_output_path