import numpy as np
import pandas as pd
import os
from shapely import vectorized

"""
This stage reads trip data from microcensus and computes modal shares.
It estimates global shares first, then shares by canton.
"""

def configure(context):
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.swiss_border")

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
    trips, filterout_ids = context.stage("data.microcensus.trips")

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

    return trips


def execute(context):
    final_trips = load_clean_trips(context)

    # Attach canton name
    df_cantons = context.stage("data.spatial.cantons")[["canton_id","canton_name_en"]].copy()
    df_cantons = df_cantons.rename(columns={"canton_name_en": "canton_name"})
    
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