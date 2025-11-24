
import numpy as np
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmc.data.training_data")


def merge_same_trips(context, df):
    df_trips = context.stage("data.microcensus.trips")[0][["person_id","trip_id","departure_time","arrival_time","mode"]]
    df_trips = df_trips.sort_values(by=["person_id","trip_id"]).reset_index(drop=True)

    columns_to_keep_last = ['purpose', 'destination_x', 'destination_y', 'destination_home', 'destination_work','is_last',
                            'parking_duration_wo_travelTime_min','destination_municipality']

    columns_to_sum = ['euclidean_distance_km']
    # Identify trips to merge
    merge_mask = (
        (df_trips["arrival_time"] == df_trips["departure_time"].shift(-1)) &
        (df_trips["mode"] == df_trips["mode"].shift(-1)) &
        (df_trips["person_id"] == df_trips["person_id"].shift(-1))
    )
    merge_indices = np.where(merge_mask)[0]    
    df_index_map = df.set_index(["person_id", "trip_id"]).index

    indices_to_drop = []
    initial_size = len(df)
    for i in merge_indices:
        person_id = df_trips.loc[i, "person_id"]        
        trip_id = df_trips.loc[i, "trip_id"]
        next_trip_id = df_trips.loc[i+1, "trip_id"]
        # check if they exists in df
        if (person_id, trip_id) not in df_index_map or (person_id, next_trip_id) not in df_index_map:
            continue

        idx_i = df_index_map.get_loc((person_id, trip_id))
        idx_ip1 = df_index_map.get_loc((person_id, next_trip_id))

        # Keep first columns as is, update last columns, sum columns
        for c in columns_to_keep_last:
            df.iloc[idx_i, df.columns.get_loc(c)] = df.iloc[idx_ip1, df.columns.get_loc(c)]
        for c in columns_to_sum:
            df.iloc[idx_i, df.columns.get_loc(c)] += df.iloc[idx_ip1, df.columns.get_loc(c)]
        
        indices_to_drop.append(idx_ip1)
        
    df = df.drop(index=indices_to_drop)

    df = df.reset_index(drop=True)
    logger.info(f"Merged trips: {initial_size - len(df)}")
    return df

