
import numpy as np
import pandas as pd
from analysis.mode_shares.utils import ModeShareAnalyzer
import logging
from mode_choice.dmc_defaults import Defaults
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dmc.data.training_data")


def merge_same_trips(context, df):
    df_trips = context.stage("data.microcensus.trips")[0][["person_id","trip_id","departure_time","arrival_time","mode"]]
    df_trips = df_trips.sort_values(by=["person_id","trip_id"]).reset_index(drop=True)

    columns_to_keep_last = ['purpose', 'destination_x', 'destination_y', 'destination_home', 'destination_work',
                            'destination_other', 'destination_leisure', 'is_last',
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
        [0, 700, 2000, 3500, 5000, 8000, 11000, 18000, 28000, 1_000_000]
    )
    analyzer = ModeShareAnalyzer(context)
    df_target = analyzer.trips.copy()

    # --- apply same behavioral filters as training data
    remove = ((df_target.euclidean_distance_km < 0.01) | (df_target.euclidean_distance_km > 100))
    remove |= ((df_target["mode"] == "walk") & (df_target["euclidean_distance_km"] >= Defaults.DEFAULT_WALK_THRESHOLD_KM))
    remove |= ((df_target["mode"] == "bike") & (df_target["euclidean_distance_km"] >= Defaults.DEFAULT_BIKE_THRESHOLD_KM))
    remove |= ((df_target["mode"] == "car") & (df_target["euclidean_distance_km"] < Defaults.DEFAULT_CAR_MIN_THRESHOLD_KM))
    remove |= ((df_target["mode"] == "pt") & (df_target["euclidean_distance_km"] < Defaults.DEFAULT_PT_MIN_THRESHOLD_KM))
    remove |= ((df_target["mode"] == "car_passenger") & (df_target["euclidean_distance_km"] < Defaults.DEFAULT_CAR_MIN_THRESHOLD_KM))
    df_target = df_target.loc[~remove].reset_index(drop=True)

    # normalize purpose
    df_target["purpose"] = df_target["purpose"].str.replace("_secondary", "", regex=False)

    # --------------------------------------------------
    # 2. Prepare ACTUAL data
    # --------------------------------------------------
    df_actual = df[
        ["mode", "euclidean_distance_km", "person_weight",
         "purpose", "sex", "income_class", "destination_municipality"]
    ].copy()
    
    # normalize purpose
    df_actual["purpose"] = df_actual["purpose"].str.replace("_secondary", "", regex=False)

    # assign distance bins
    distance_bins = np.array(analyzer.distance_bins) / 1000
    distance_labels = analyzer.get_distance_labels()

    df_actual["distance_bin"] = pd.cut(
        df_actual["euclidean_distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
        ordered=True
    )

    df_target["distance_bin"] = pd.cut(
        df_target["euclidean_distance_km"],
        bins=distance_bins,
        labels=distance_labels,
        include_lowest=True,
        ordered=True
    )

    # --------------------------------------------------
    # 3. HARD CONSTRAINT: distance × mode and sex x mode
    # --------------------------------------------------
    # double the weights of long distances:
    df_actual.loc[df_actual.euclidean_distance_km > 4, "person_weight"] *= 2
    df_target.loc[df_target.euclidean_distance_km > 4, "person_weight"] *= 2
    df_actual.loc[df_actual.euclidean_distance_km > 20, "person_weight"] *= 1.5
    df_target.loc[df_target.euclidean_distance_km > 20, "person_weight"] *= 1.5

    df_target["bin_mode"] = df_target["distance_bin"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_mode"] = df_actual["distance_bin"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_bin_mode = (
        df_target
        .groupby("bin_mode")["person_weight"]
        .sum()
    )
    
    # # double the weights of long distance trips
    # long_distance_labels = distance_labels[-3:]  # last 3 bins
    # for label in long_distance_labels:
    #     logger.info(f"\t - Doubling target weights for distance bin: {label}")
    #     mask = target_bin_mode.index.str.startswith(label + '|')
    #     target_bin_mode.loc[mask] *= 2

    # --------------------------------------------------
    df_target["bin_sex"] = df_target["sex"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["bin_sex"] = df_actual["sex"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_bin_sex = (
        df_target
        .groupby("bin_sex")["person_weight"]
        .sum()
    )

    df_target["municipality_type_mode"] = df_target["destination_municipality"].astype(str) + "|" + df_target["mode"].astype(str)
    df_actual["municipality_type_mode"] = df_actual["destination_municipality"].astype(str) + "|" + df_actual["mode"].astype(str)    
    target_municipality_type_mode = (
        df_target
        .groupby("municipality_type_mode")["person_weight"]
        .sum()
    )
    # --------------------------------------------------
    # 4. SOFT MARGINAL TARGETS
    # --------------------------------------------------
    target_income = (
        df_target
        .groupby("income_class")["person_weight"]
        .sum()
    )

    target_purpose = (
        df_target
        .groupby("purpose")["person_weight"]
        .sum()
    )

    target_mode = (
        df_target
        .groupby("mode")["person_weight"]
        .sum()
    )

    targets = [
        target_bin_mode,   # HARD
        target_bin_sex,    # SOFT
        target_municipality_type_mode,  # SOFT
        target_mode,       # SOFT
        target_purpose,    # SOFT 
        target_income      # SOFT (weakest)
    ]

    dimensions = [
        ["bin_mode"],
        ["bin_sex"],
        ["municipality_type_mode"],
        ["mode"],        
        ["purpose"],   
        ["income_class"],
    ]

    # --------------------------------------------------
    # 5. Feasibility checks (CRITICAL)
    # --------------------------------------------------
    def check_missing(col):
        missing = set(df_target[col]) - set(df_actual[col])
        if missing:
            raise ValueError(f"Infeasible raking: missing categories in {col}: {missing}")

    check_missing("bin_mode")
    check_missing("sex")
    check_missing("income_class")
    check_missing("purpose")

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

    return df_actual["person_weight"]
