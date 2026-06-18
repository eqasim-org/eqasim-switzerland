import numpy as np
import pandas as pd
import os
from shapely import vectorized
import logging 

logger = logging.getLogger("synpp")
"""
This stage reads trip data from microcensus and computes modal shares.
It estimates global shares first, then shares by canton.
"""

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.swiss_border")

    context.config("include_external_population", default=False)
    if context.config("include_external_population"):
        context.stage("data.external_population.hts_trips.trips")
        context.stage("data.external_population.constants")


def merge_same_trips(context, df):
    """Merge consecutive legs that represent the same trip for a person.

    Two legs are merged when they belong to the same person, use the same mode
    and the first leg's ``arrival_time`` equals the next leg's ``departure_time``.
    Chains of such legs are collapsed together.
    """

    if df.empty:
        return df

    # Columns we aggregate differently when merging a chain of legs
    columns_to_keep_last = ['destination_x', 'destination_y', 'arrival_time', 'purpose', 'activity_duration', 'parking_cost']
    columns_to_sum = ['crowfly_distance', 'network_distance']

    columns_to_keep_last = [c for c in columns_to_keep_last if c in df.columns]
    columns_to_sum = [c for c in columns_to_sum if c in df.columns]

    # Ensure deterministic ordering (trip_id is chronological within the person/day)
    df = df.sort_values(["person_id", "trip_id"]).reset_index(drop=True)

    # Identify whether the current leg should be merged with the previous one.
    # Using shift(1) correctly handles chains A->B->C: both B and C stay in the same group
    # because each continues the previous leg.
    merge_with_prev = (
        (df["person_id"] == df["person_id"].shift(1)) &
        (df["mode"] == df["mode"].shift(1)) &
        (df["mode"] == "car") & 
        (df["departure_time"] == df["arrival_time"].shift(1))
    )

    # Start a new group when the current row does NOT merge with the previous one
    group_id = (~merge_with_prev).cumsum()
    
    # Build aggregation map: default to first row, override sums/lasts
    agg_map = {c: 'first' for c in df.columns}
    for c in columns_to_sum:
        agg_map[c] = 'sum'
    for c in columns_to_keep_last:
        agg_map[c] = 'last'

    merged = df.groupby(group_id, sort=False, as_index=False).agg(agg_map)
    logger.info(f"Merged trips: {len(df) - len(merged)}")
    return merged.reset_index(drop=True)


def load_clean_trips(context):
    """
    Load and filter Microcensus trips, enriching them with person attributes and applying
    quality and geographic constraints.
    This function:
    1) Loads persons and trips from the pipeline.
    2) Merges selected person attributes into the trips.
    3) Excludes trips whose ``person_id`` is in the provided ``filterout_ids`` as well as
        persons flagged as weekend (``persons['weekend'] == True``).
    4) Applies additional trip-level filters:
        - non-missing ``household_weight`` and ``person_weight``
        - ``crowfly_distance`` > 1
        - mode in ``{"car", "pt", "bike", "walk", "car_passenger"}``
    5) Keeps only trips whose origin and destination coordinates lie within the Swiss border
        polygon (loaded from ``data.spatial.swiss_border``).
    Parameters
    ----------
    context : object
    Returns
    -------
    pandas.DataFrame
         Filtered trips DataFrame including merged person attributes.
    """
    persons = context.stage("data.microcensus.persons")
    trips, filterout_ids, _, _ = context.stage("data.microcensus.trips")

    # Merge with persons
    persons = persons[['person_id', 'person_weight', 'age', 'sex', 
                        'income_class', 'canton_id', 'household_weight', 'weekend']]
    
    trips = trips.merge(persons, how="left", on="person_id")

    # Identify weekend persons to filter out
    week_end_persons = persons[persons['weekend']]["person_id"].unique()
    filterout_ids = filterout_ids.union(set(week_end_persons))

    # filter trips
    trips = trips[~trips['person_id'].isin(filterout_ids)]

    # further filter trips
    modes = ["car","pt","bike","walk","car_passenger"]        
    sel = ((trips.household_weight.notna()) & 
            (trips.person_weight.notna()) &
            (trips.crowfly_distance>1) &
            (trips["mode"].isin(modes)) )
    trips = trips[sel].reset_index(drop=True)       

    # keep only trips within switzerland
    df_switzerland = context.stage("data.spatial.swiss_border")
    ch_polygon = df_switzerland.buffer(0).iloc[0] 
    inside_origin = vectorized.contains(ch_polygon, trips["origin_x"].values, trips["origin_y"].values)
    inside_destination = vectorized.contains(ch_polygon, trips["destination_x"].values, trips["destination_y"].values)
    within_ch = inside_origin&inside_destination 
    trips = trips[within_ch].reset_index(drop=True)

    # merge same trips
    # trips = merge_same_trips(context, trips)
    return trips


def execute(context):
    final_trips = load_clean_trips(context)

    # Attach canton name
    df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name_en"]].copy()
    df_cantons = df_cantons.rename(columns={"canton_name_en": "canton_name"})
    
    df_cantons["canton_id"]  = df_cantons["canton_id"].astype(int)
    final_trips["canton_id"] = final_trips["canton_id"].astype(int)

    final_trips = pd.merge(final_trips, df_cantons, on = "canton_id",  how = "left")
    assert final_trips.canton_name.notnull().all(), "Not all persons have a canton name assigned. Check the canton data."

    # Compute global modal split shares
    total_weighted_trips = final_trips["person_weight"].sum()
    global_mode_weights  = (
        final_trips.groupby("mode")["person_weight"]
        .sum()
        .reset_index(name = "total_weight")
    )
    global_mode_weights["mode_share"] = (global_mode_weights["total_weight"] / total_weighted_trips).round(4)
    global_modal_shares = global_mode_weights[["mode", "mode_share"]]

    # Compute cantonal modal split shares
    cantonal_modal_shares = (
        final_trips
        .groupby(["canton_name", "mode"], observed = False)["person_weight"]
        .sum()
        .groupby(level = "canton_name", observed=False)
        .transform(lambda x: x / x.sum())
        .rename("mode_share")
        .reset_index()
        .pivot(index = "canton_name", columns = "mode", values = "mode_share")
        .fillna(0)
        .round(3)
        .sort_values(by = "canton_name")
    )

    if context.config("include_external_population"):
        fr_trips       = context.stage("data.external_population.hts_trips.trips")
        ex_constants   = context.stage("data.external_population.constants")

        mode_shares_fr = fr_trips.groupby("mode", as_index = False)["trip_weight"].sum() 

        mode_shares_fr["trip_weight"] = (mode_shares_fr["trip_weight"] / fr_trips["trip_weight"].sum()).round(3)
        mode_shares_fr = mode_shares_fr.rename(columns = {"trip_weight": ex_constants.canton_name})
        mode_shares_fr = mode_shares_fr.T

        mode_shares_fr.columns = mode_shares_fr.iloc[0]
        mode_shares_fr = mode_shares_fr.drop("mode")

        cantonal_modal_shares = pd.concat([cantonal_modal_shares, mode_shares_fr])

    # Define output file paths
    global_shares_output_path   = os.path.join(context.path(), "globalModeShares.csv")
    cantonal_shares_output_path = os.path.join(context.path(), "cantonalModeShares.csv")

    # Save global modal shares
    global_modal_shares.to_csv(global_shares_output_path, index=False)

    # Save cantonal modal shares
    cantonal_modal_shares.to_csv(cantonal_shares_output_path, index=True)

    return global_shares_output_path, cantonal_shares_output_path