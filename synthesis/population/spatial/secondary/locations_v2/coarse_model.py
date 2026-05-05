import numpy as np
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from .NNModel import MNLModel, train_model, MNLWrapper
from .h3 import within_ch
from .hierarchical_model_utils import (
    SECONDARY_ACTIVITIES,
    build_coarse_numerical_batch_numba,
    build_coarse_numerical_batch,
    build_coarse_scaled_feature_batch,
)
import os
import logging
import torch
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

logger = logging.getLogger("synpp: coarse_model")

MODEL_NAME = "coarse_model.pt"
    
def configure(context):
    context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
    context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    context.stage("synthesis.population.destinations")

    context.config("overwrite_coarse_model_if_exists", True)
    context.config("coarse_model_epochs", 15)
    context.config("coarse_model_batch_size", 512)
    context.config("coarse_model_learning_rate", 2e-2)
    context.config("coarse_model_torch_num_threads", 16)
    context.config("random_seed")


def execute(context):
    logger.info("Training coarse model for secondary location choice...")
    overwrite_model = context.config("overwrite_coarse_model_if_exists")
    model_path = os.path.join(context.working_directory, MODEL_NAME)
    if os.path.exists(model_path):
        if not overwrite_model:
            logger.info(f"Model {MODEL_NAME} already exists.")
            return model_path
        else:
            logger.info(f"Model {MODEL_NAME} already exists but will be overwritten as per configuration.")
    
    ### Load data
    logger.info("\t Loading data...")
    # Load Microcensus data
    mz_persons = context.stage("data.microcensus.persons")[["person_id","age","sex","car_availability","bike_availability","employed", "sp_region",
                                                            "income_class", "N_children_under_12", "home_x","home_y","work_x","work_y","weekend", "person_weight"]]
    c = context.stage("data.constants")
    mz_persons["car_availability"] = (mz_persons["car_availability"]!=c.CAR_AVAILABILITY_NEVER)

    mz_trips,_ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id","trip_id","origin_x","origin_y","destination_x", "destination_y", "purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")[[
        "person_id",
        "trip_id",
        "daily_longest_distance_from_home",
        "daily_crowfly_total",
        "crowfly_consumed_before_trip",
        "trip_position_class",
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    # I think it is important to filter out weekend trips, since we can go far in the weekend for shoping, or for leisure activities
    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    # we only keep trips within switzerland
    inside_ch = within_ch(context, mz_trips, cols1=["origin_x", "origin_y"], cols2=["destination_x", "destination_y"])
    mz_trips = mz_trips[inside_ch].reset_index(drop=True)

    # Load H3 levels
    h3, h3_geo, _ = context.stage("synthesis.population.spatial.secondary.locations_v2.h3")

    persons_h3 = h3["microcensus_persons"][["person_id", "home_level_0","work_level_0"]]
    trips_h3   = h3["microcensus_trips"][["person_id","trip_id", "origin_level_0", "destination_level_0"]]
    h3_geo     = h3_geo["level_0"]

    # determines what percentage of the polygone is outside the border
    if "outside_fraction" not in h3_geo.columns:
        raise RuntimeError("Missing outside_fraction in H3 level_0 geometry. Run h3 stage with outside_fraction enabled.")

    ### Processing data
    logger.info("\t Processing data...")
    # Merge person/trip H3 labels only (geometry merges are not needed for coarse model features).
    mz_persons = mz_persons.merge(persons_h3, on="person_id", how="left")
    mz_trips = mz_trips.merge(trips_h3, on=["person_id","trip_id"], how="left")

    # Filter secondary trips
    secondary_trips = mz_trips[mz_trips["purpose"].isin(SECONDARY_ACTIVITIES)].dropna(subset=["destination_level_0"])

    # Get all H3
    all_h3 = h3_geo["h3_index"].tolist()
    num_h3 = len(all_h3)
    h3_to_index = {h3_idx: i for i, h3_idx in enumerate(all_h3)}

    required_h3_cols = [
        "centroid",
        "outside_fraction",
        "num_statent",
        "employees",
        "urban_core",
        "urban",
        "education",
        "shop",
        "leisure",
        "ovgk_share_a",
        "ovgk_share_b",
        "ovgk_share_c",
        "ovgk_share_d",
        "ovgk_share_none",
    ]
    missing_h3_cols = [c for c in required_h3_cols if c not in h3_geo.columns]
    if missing_h3_cols:
        raise RuntimeError("Missing required destination-derived H3 columns in h3 stage: " + ", ".join(missing_h3_cols))

    # Load all H3 attributes precomputed by h3.py
    centroids = h3_geo.set_index("h3_index")[required_h3_cols].reindex(all_h3)
    centroid_x = centroids["centroid"].x.to_numpy(dtype=np.float64)
    centroid_y = centroids["centroid"].y.to_numpy(dtype=np.float64)
    outside_fraction = centroids["outside_fraction"].to_numpy(dtype=np.float64)
    statent_per_h3 = centroids["num_statent"].to_numpy(dtype=np.float64)
    employees_per_h3 = centroids["employees"].to_numpy(dtype=np.float64)
    urban_core_per_h3 = centroids["urban_core"].to_numpy(dtype=np.float64)
    urban_per_h3 = centroids["urban"].to_numpy(dtype=np.float64)
    education_per_h3 = centroids["education"].to_numpy(dtype=np.float64)
    shop_per_h3 = centroids["shop"].to_numpy(dtype=np.float64)
    leisure_per_h3 = centroids["leisure"].to_numpy(dtype=np.float64)
    ovgk_share_a_per_h3 = centroids["ovgk_share_a"].to_numpy(dtype=np.float64)
    ovgk_share_b_per_h3 = centroids["ovgk_share_b"].to_numpy(dtype=np.float64)
    ovgk_share_c_per_h3 = centroids["ovgk_share_c"].to_numpy(dtype=np.float64)
    ovgk_share_d_per_h3 = centroids["ovgk_share_d"].to_numpy(dtype=np.float64)
    ovgk_share_none_per_h3 = centroids["ovgk_share_none"].to_numpy(dtype=np.float64)

    # Trip table with all person attributes needed for feature construction
    person_cols = ["person_id", "age", "sex", "employed", "car_availability", "income_class", "home_x", "home_y", "work_x", "work_y"]
    trip_cols = [
           "person_id", "trip_id", "origin_x", "origin_y", "origin_level_0", "destination_level_0", "purpose",
        "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip",
        "trip_position_class",
    ]
    df_trips = secondary_trips[trip_cols].merge(mz_persons[person_cols + ["person_weight"]], on="person_id", how="left")

    y_series = df_trips["destination_level_0"].map(h3_to_index)
    valid = y_series.notna()
    if (~valid).any():
        logger.warning("\t Dropping %s trips with destination H3 not found in candidate set.", int((~valid).sum()))
    df_trips = df_trips[valid].reset_index(drop=True)
    y = y_series[valid].astype(np.int64).to_numpy()
    weights = df_trips["person_weight"].to_numpy(dtype=np.float32)

    n_trips = len(df_trips)
    logger.info(f"\t {n_trips} secondary trips with valid H3 destinations will be used for training.")
    if n_trips == 0:
        raise RuntimeError("No valid secondary trips available for coarse model training.")

    # USE NUMPY FOR COMPUTIONAL EFFICIENCY IN FEATURE CONSTRUCTION, THEN MOVE TO TORCH FOR MODEL TRAINING.
    home_x = df_trips["home_x"].to_numpy(dtype=np.float64)
    home_y = df_trips["home_y"].to_numpy(dtype=np.float64)
    work_x = df_trips["work_x"].to_numpy(dtype=np.float64)
    work_y = df_trips["work_y"].to_numpy(dtype=np.float64)
    has_work = np.isfinite(work_x) & np.isfinite(work_y)
    # Ensure invalid work coordinates never leak into distance computations.
    work_x = np.where(has_work, work_x, 0.0)
    work_y = np.where(has_work, work_y, 0.0)
    origin_x = df_trips["origin_x"].to_numpy(dtype=np.float64)
    origin_y = df_trips["origin_y"].to_numpy(dtype=np.float64)
    age = df_trips["age"].to_numpy(dtype=np.float64)
    daily_longest_distance_from_home = df_trips["daily_longest_distance_from_home"].to_numpy(dtype=np.float64)
    daily_longest_distance_from_home = np.where(np.isfinite(daily_longest_distance_from_home) & (daily_longest_distance_from_home >= 0.0), daily_longest_distance_from_home, 0.0)
    daily_crowfly_total = df_trips["daily_crowfly_total"].to_numpy(dtype=np.float64)
    daily_crowfly_total = np.where(np.isfinite(daily_crowfly_total) & (daily_crowfly_total >= 0.0), daily_crowfly_total, 0.0)
    crowfly_consumed_before_trip = df_trips["crowfly_consumed_before_trip"].to_numpy(dtype=np.float64)
    crowfly_consumed_before_trip = np.where(np.isfinite(crowfly_consumed_before_trip) & (crowfly_consumed_before_trip >= 0.0), crowfly_consumed_before_trip, 0.0)
    trip_position_class = df_trips["trip_position_class"].to_numpy(dtype=np.float64)
    trip_position_class = np.where(np.isfinite(trip_position_class), trip_position_class, 2.0)
    sex = df_trips["sex"].to_numpy(dtype=np.float32)
    employed = df_trips["employed"].to_numpy(dtype=np.float32)
    car_availability = df_trips["car_availability"].to_numpy(dtype=np.float32)
    income_class = df_trips["income_class"].to_numpy(dtype=np.float32)
    # origin_level_0 = df_trips["origin_level_0"].map(h3_to_index).to_numpy(dtype=np.int64)

    purpose_categories = [str(p) for p in SECONDARY_ACTIVITIES]
    purpose_to_idx = {p: i for i, p in enumerate(purpose_categories)}
    purpose_idx = df_trips["purpose"].astype(str).map(purpose_to_idx).fillna(0).astype(np.int64).to_numpy()
    purpose_one_hot = np.eye(len(purpose_categories), dtype=np.float32)[purpose_idx]

    batch_size = 1024 * 8
    batch_ranges = [(start, min(start + batch_size, n_trips)) for start in range(0, n_trips, batch_size)]

    logger.info("\t Using Numba kernel for numerical feature batches.")
    
    # First call pays JIT compile cost; warm up once on a tiny slice.
    warm_end = min(2, n_trips)
    build_coarse_numerical_batch_numba(home_x[:warm_end], home_y[:warm_end], work_x[:warm_end], work_y[:warm_end],has_work[:warm_end],
                                           origin_x[:warm_end], origin_y[:warm_end], age[:warm_end],
                                       daily_longest_distance_from_home[:warm_end], daily_crowfly_total[:warm_end],
                                       crowfly_consumed_before_trip[:warm_end], trip_position_class[:warm_end], income_class[:warm_end], centroid_x, centroid_y,
                                       statent_per_h3, employees_per_h3, urban_core_per_h3, urban_per_h3, education_per_h3, shop_per_h3, leisure_per_h3,
                                       ovgk_share_a_per_h3, ovgk_share_b_per_h3, ovgk_share_c_per_h3, ovgk_share_d_per_h3, ovgk_share_none_per_h3,
                                       outside_fraction)
    
    logger.info(f"\t Building feature tensor in batches: {n_trips} trips, {num_h3} H3 alternatives, batch_size={batch_size}")

    # Fit scaling in one shot.
    numerical = [
           "dist_home", "dist_work", "dist_last", "age",
        "num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure",
        "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none",
        "outside_fraction",
        "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip",
        "trip_position_class", "income_class",
    ]
    distance_cols = [numerical.index(col) for col in [
           "dist_home", "dist_work", "dist_last",
        "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip",
    ]]
    count_cols = [numerical.index(col) for col in ["num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure"]]
    positive_cols = [numerical.index(col) for col in ["age", "income_class"]]
    passthrough_cols = [numerical.index(col) for col in [
        "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none", "outside_fraction", "trip_position_class"
    ]]
    
    scaler = ColumnTransformer([
        ('dist', Pipeline([
            ('log1p', FunctionTransformer(np.log1p, validate=False)),
            ('scale', QuantileTransformer(output_distribution='uniform')),
        ]), distance_cols),
        ('count', QuantileTransformer(output_distribution='normal'), count_cols),
        ('positive', QuantileTransformer(output_distribution='uniform'), positive_cols),
        ('passthrough', 'passthrough', passthrough_cols),
    ])
    logger.info("\t Fitting scaler on full numerical matrix...")
    full_numerical = build_coarse_numerical_batch(0, n_trips, home_x, home_y, work_x, work_y, has_work, origin_x, origin_y,
                                                      age, daily_longest_distance_from_home, daily_crowfly_total,
                                                  crowfly_consumed_before_trip, trip_position_class, income_class, centroid_x, centroid_y,
                                                  statent_per_h3, employees_per_h3, urban_core_per_h3, urban_per_h3, education_per_h3, shop_per_h3, leisure_per_h3,
                                                  ovgk_share_a_per_h3, ovgk_share_b_per_h3, ovgk_share_c_per_h3, ovgk_share_d_per_h3, ovgk_share_none_per_h3,
                                                  outside_fraction)
    
    scaler.fit(full_numerical.reshape(-1, len(numerical)))
    del full_numerical

    # Build final X tensor (n_trips, num_h3, n_features) in batches.
    purpose_features = [f"purpose_{p}" for p in purpose_categories]
    features = numerical + ["sex", "employed", "car_availability"] + purpose_features
    X = np.empty((n_trips, num_h3, len(features)), dtype=np.float32)

    with context.progress(total=len(batch_ranges), label="Coarse model: building feature tensor") as progress:
        for start, end in batch_ranges:
            _, _, out = build_coarse_scaled_feature_batch(start, end, home_x, home_y, work_x, work_y, has_work, origin_x, origin_y, age,
                                                          daily_longest_distance_from_home, daily_crowfly_total, crowfly_consumed_before_trip,
                                                          trip_position_class, income_class, centroid_x, centroid_y,
                                                          statent_per_h3, employees_per_h3, urban_core_per_h3, urban_per_h3, education_per_h3, shop_per_h3, leisure_per_h3,
                                                          ovgk_share_a_per_h3, ovgk_share_b_per_h3, ovgk_share_c_per_h3, ovgk_share_d_per_h3, ovgk_share_none_per_h3,
                                                          outside_fraction,
                                                          scaler, len(numerical), num_h3, len(features), sex, employed, car_availability, purpose_one_hot)
            X[start:end] = out
            progress.update()

    # Train model
    logger.info("\t Training model...")
    seed = context.config("random_seed")
    np.random.seed(seed)    
    torch.manual_seed(seed)
    model = MNLModel(input_dim=len(features), num_h3=num_h3, hidden_dims=(128, 32), dropout_rate=0.1)
    train_model(model, X, y, weights=weights,
                epochs=context.config("coarse_model_epochs"),
                num_threads=int(context.config("coarse_model_torch_num_threads")),
                batch_size=context.config("coarse_model_batch_size"), 
                lr=context.config("coarse_model_learning_rate")
    )

    # Wrap and save
    wrapper = MNLWrapper(model, scaler, numerical, features, all_h3)
    wrapper.save(model_path)


    h3_geo_counts = plot_analysis(context, wrapper, X, y, all_h3, h3_geo, centroid_x, centroid_y, home_x, home_y, origin_x, origin_y, h3_to_index, n_trips)
    return wrapper, X, features, model_path, h3_geo_counts

    

def plot_analysis(context, wrapper, X, y, all_h3, h3_geo, centroid_x, centroid_y, home_x, home_y, origin_x, origin_y, h3_to_index, n_trips):
    # Predict on training data and plot
    logger.info("Predicting on training data and plotting...")
    predicted_h3 = wrapper.predict_from_X(X)
    real_h3_list = [all_h3[i] for i in y]
    real_counts = pd.Series(real_h3_list).value_counts().rename('real_count')
    pred_counts = pd.Series(predicted_h3).value_counts().rename('pred_count')
    counts_df = pd.DataFrame({'real_count': real_counts, 'pred_count': pred_counts}).fillna(0)
    h3_geo_counts = h3_geo.set_index('h3_index').join(counts_df, how='left').fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h3_geo_counts.plot(column='real_count', ax=axes[0], legend=True, cmap='viridis', legend_kwds={"shrink": 0.5})
    axes[0].set_title('Real H3 Counts')
    h3_geo_counts.plot(column='pred_count', ax=axes[1], legend=True, cmap='viridis', legend_kwds={"shrink": 0.5})
    axes[1].set_title('Predicted H3 Counts')
    plt.tight_layout()
    plot_path = os.path.join(context.path(), 'h3_counts_comparison.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_path}")

    # Plot distance distributions: predictions vs actual
    logger.info("Plotting distance distributions...")
    pred_indices = [h3_to_index[h3] for h3 in predicted_h3]
    
    real_dist_home = []
    real_dist_last = []
    pred_dist_home = []
    pred_dist_last = []
    
    for i in range(n_trips):
        real_idx = y[i]
        real_cx = centroid_x[real_idx]
        real_cy = centroid_y[real_idx]
        real_dist_home.append(np.sqrt((real_cx - home_x[i])**2 + (real_cy - home_y[i])**2))
        real_dist_last.append(np.sqrt((real_cx - origin_x[i])**2 + (real_cy - origin_y[i])**2))
        
        pred_idx = pred_indices[i]
        pred_cx = centroid_x[pred_idx]
        pred_cy = centroid_y[pred_idx]
        pred_dist_home.append(np.sqrt((pred_cx - home_x[i])**2 + (pred_cy - home_y[i])**2))
        pred_dist_last.append(np.sqrt((pred_cx - origin_x[i])**2 + (pred_cy - origin_y[i])**2))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    THRESHOLD = 80000  # 80 km
    real_dist_home = np.array(real_dist_home)
    real_dist_last = np.array(real_dist_last)
    pred_dist_home = np.array(pred_dist_home)
    pred_dist_last = np.array(pred_dist_last)
    real_dist_home = real_dist_home[real_dist_home <= THRESHOLD]
    real_dist_last = real_dist_last[real_dist_last <= THRESHOLD]
    pred_dist_home = pred_dist_home[pred_dist_home <= THRESHOLD]
    pred_dist_last = pred_dist_last[pred_dist_last <= THRESHOLD]
    # Distance from home
    axes[0].hist(real_dist_home, bins=50, histtype='step', color='black', linewidth=2, label='Real', density=True)
    axes[0].hist(pred_dist_home, bins=50, histtype='step', color='red', linewidth=1, label='Predicted', density=True, linestyle='dashed')
    axes[0].set_title('Distance from Home')
    axes[0].set_xlabel('Distance (m)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Distance from last
    axes[1].hist(real_dist_last, bins=50, histtype='step', color='black', linewidth=2, label='Real', density=True)
    axes[1].hist(pred_dist_last, bins=50, histtype='step', color='red', linewidth=1, label='Predicted', density=True, linestyle='dashed')
    axes[1].set_title('Distance from Last')
    axes[1].set_xlabel('Distance (m)')
    axes[1].set_ylabel('Density')
    axes[1].legend()

    plt.tight_layout()
    dist_plot_path = os.path.join(context.path(), 'distance_distributions.png')
    plt.savefig(dist_plot_path, dpi=200, bbox_inches='tight')
    logger.info(f"Distance distribution plot saved to {dist_plot_path}")

    return h3_geo_counts
