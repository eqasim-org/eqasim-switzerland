
import numpy as np
import pandas as pd
import logging
from analysis.mode_shares.utils import ModeShareAnalyzer
from dmc.constants import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmc.data.training_data")

MS_REGIONS = constants.MS_REGIONS

def merge_same_trips(context, df):
    df_trips = context.stage("data.microcensus.trips")[0][["person_id","trip_id","departure_time","arrival_time","mode"]]
    df_trips = df_trips.sort_values(by=["person_id","trip_id"]).reset_index(drop=True)

    columns_to_keep_last = ['destination_x', 'destination_y', 'destination_home', 'destination_work', 'destination_municipality', 'is_last',
        'parking_duration_wo_travelTime_min']

    columns_to_sum = [ 'car_travel_time_min', 'car_distance_km', 'car_passenger_travel_time_min', 'car_passenger_distance_km', 'pt_travel_time_min',
                       'pt_in_vehicle_time_min', 'pt_egress_time_min', 'pt_access_time_min', 'pt_transfer_time_min', 'pt_access_egress_time_min', 'pt_transfers', 
                       'pt_in_vehicle_distance_km', 'walk_travel_time_min', 'walk_distance_km','bike_travel_time_min', 'bike_distance_km', 'euclidean_distance_km']

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





def adjust_weights(context, df):
    """
    Adjust person weights so that:
      1) Mode shares within distance bins are EXACT (post-stratification via IPF interaction)
      2) Sex, income, and purpose marginals are respected as softly as possible

    Returns
    -------
    pandas.Series
        Adjusted person_weight aligned with df index
    """
    from ipfn.ipfn import ipfn
    # --------------------------------------------------
    # 1. Build TARGET data (reference distribution)
    # --------------------------------------------------
    ModeShareAnalyzer.set_distance_bins(
        [0, 700, 1500, 3000, 4500, 7000, 10500, 15000, 20000, 1_000_000]
    )
    ModeShareAnalyzer.set_age_bins([0, 18, 30, 45, 65, 100])
    analyzer = ModeShareAnalyzer(context)
    df_target = analyzer.trips.copy()
    df_target["ms_region"] = df_target["canton_id"].map(lambda x: MS_REGIONS.loc[x,"cluster"])

    # --- apply same behavioral filters as training data
    remove = ((df_target.euclidean_distance_km < 0.01) | (df_target.euclidean_distance_km > 100))
    remove |= ((df_target["mode"] == "walk") & (df_target["euclidean_distance_km"] >= 6))
    remove |= ((df_target["mode"] == "bike") & (df_target["euclidean_distance_km"] >= 12))
    remove |= ((df_target["mode"] == "car") & (df_target["age"] < 18))
    df_target = df_target.loc[~remove].reset_index(drop=True)

    # normalize purpose
    df_target["purpose"] = df_target["purpose"].str.replace("_secondary", "", regex=False)

    # --------------------------------------------------
    # 2. Prepare ACTUAL data
    # --------------------------------------------------
    df_actual = df[
        ["mode", "euclidean_distance_km", "person_weight","age",
         "purpose", "sex", "income_class", "ms_region"]
    ].copy()

    df_actual["purpose"] = df_actual["purpose"].str.replace("_secondary", "", regex=False)

    # assign distance bins
    df_actual["distance_bin"] = analyzer.get_distance_bins(df_actual)
    df_target["distance_bin"] = analyzer.get_distance_bins(df_target)

    # assign age bins
    df_actual["age_class"] = analyzer.get_age_bins(df_actual)
    df_target["age_class"] = analyzer.get_age_bins(df_target)

    # double the weights of long distances (make the model more sensitive to long distances):
    # df_actual.loc[df_actual.euclidean_distance_km > 4.5, "person_weight"] *= 1.3
    # df_target.loc[df_target.euclidean_distance_km > 4.5, "person_weight"] *= 1.3
    # df_actual.loc[df_actual.euclidean_distance_km > 20, "person_weight"] *= 1.5
    # df_target.loc[df_target.euclidean_distance_km > 20, "person_weight"] *= 1.5

    # --------------------------------------------------
    # 3. CONSTRAINT
    # --------------------------------------------------
    df_target["distance_bin_mode"] = df_target["distance_bin"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["distance_bin_mode"] = df_actual["distance_bin"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_distance_bin_mode = (
        df_target
        .groupby("distance_bin_mode")["person_weight"]
        .sum()
    )
    
    df_target["bin_sex"] = df_target["sex"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_sex"] = df_actual["sex"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_bin_sex = (
        df_target
        .groupby("bin_sex")["person_weight"]
        .sum()
    )

    df_target["bin_age"] = df_target["age_class"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_age"] = df_actual["age_class"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_bin_age = (
        df_target
        .groupby("bin_age")["person_weight"]
        .sum()
    )

    df_target["bin_income"] = df_target["income_class"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_income"] = df_actual["income_class"].astype(str) + "|" + df_actual["mode"].astype(str)
    target_bin_income = (
        df_target
        .groupby("bin_income")["person_weight"]
        .sum()
    )

    df_target["bin_purpose"] = df_target["purpose"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_purpose"] = df_actual["purpose"].astype(str) + "|" + df_actual["mode"].astype(str)
    target_purpose = (
        df_target
        .groupby("bin_purpose")["person_weight"]
        .sum()
    )

    target_mode = (
        df_target
        .groupby("mode")["person_weight"]
        .sum()
    )

    df_target["bin_ms_region"] = df_target["ms_region"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_ms_region"] = df_actual["ms_region"].astype(str) + "|" + df_actual["mode"].astype(str)
    target_bin_ms_region = (
        df_target
        .groupby("bin_ms_region")["person_weight"]
        .sum()
    )

    targets = [
        target_distance_bin_mode,   # HARD
        target_bin_sex,         # SOFT
        target_bin_age,         # SOFT
        target_bin_income,      # SOFT        
        target_purpose,    # SOFT
        target_bin_ms_region, # SOFT   
        target_mode,       # SOFT  
    ]

    dimensions = [
        ["distance_bin_mode"],
        ["bin_sex"],
        ["bin_age"],
        ["bin_income"],          
        ["bin_purpose"]        ,
        ["bin_ms_region"],
        ["mode"],
    ]

    # --------------------------------------------------
    # 5. Feasibility checks (CRITICAL)
    # --------------------------------------------------
    def check_missing(col):
        missing = set(df_target[col]) - set(df_actual[col])
        if missing:
            raise ValueError(f"Infeasible raking: missing categories in {col}: {missing}")

    check_missing("distance_bin_mode")
    check_missing("bin_sex")
    check_missing("bin_income")
    check_missing("bin_purpose")

    # --------------------------------------------------
    # 6. RAKING with trim + rerake
    # --------------------------------------------------
    df_actual = df_actual.copy()
    df_actual["person_weight_orig"] = df_actual["person_weight"]    
    for j in range(3):
        logger.info(f"\t - Raking iteration {j+1}/3")
        ipf = ipfn(
            df_actual,
            aggregates=targets,
            dimensions=dimensions,
            weight_col="person_weight",
            convergence_rate=1e-6,
            max_iteration=500
        )
        df_actual = ipf.iteration()

        # trim extreme weights
        median_w = df_actual["person_weight"].median()
        df_actual["person_weight"] = df_actual["person_weight"].clip(
            lower= (0.2 if j < 2 else 0.1) * median_w,
            upper= (7.0 if j < 2 else 10.0) * median_w
        )
        if j == 0:
            logger.info(f"\t\t - Original weights stats: min={df_actual['person_weight_orig'].min():.2f}, median={df_actual['person_weight_orig'].median():.2f}, max={df_actual['person_weight_orig'].max():.2f}")
        logger.info(f"\t\t - weight stats after trimming: min={df_actual['person_weight'].min():.2f}, median={df_actual['person_weight'].median():.2f}, max={df_actual['person_weight'].max():.2f}")

    return df_actual["person_weight"]
