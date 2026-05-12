import os
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

from .h3 import within_ch
from .hierarchical_utils import SECONDARY_ACTIVITIES, build_coarse_candidate_batch_numba
from .feature_encoding import CANDIDATE_FEATURES, STATIC_CANDIDATE_FEATURES, N_CANDIDATE_DYNAMIC, ACTIVITY_CHAIN_N, fit_candidate_tensor, fit_person_trip_matrix
from .choice_model import NeuralChoiceModel, train_choice_model
from .model_wrappers import RegionalChoiceWrapper

logger = logging.getLogger("synpp: regional_model")

MODEL_NAME = "regional_model.pt"


def configure(context):
    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    
    context.config("threads")
    context.config("random_seed")

    # training params
    context.config("overwrite_regional_model_if_exists", True)
    context.config("regional_model_epochs", 50)
    context.config("regional_model_batch_size", 512)
    context.config("regional_model_learning_rate", 4e-3) 


def execute(context):
    logger.info("Training regional model (neural choice) for secondary location choice...")
    overwrite_model = context.config("overwrite_regional_model_if_exists")
    model_path = os.path.join(context.working_directory, MODEL_NAME)
    person_static_scaler_path  = os.path.join(context.working_directory, "person_static_scaler.sklearn.pkl")
    person_dynamic_scaler_path = os.path.join(context.working_directory, "person_dynamic_scaler.sklearn.pkl")

    model_and_scalers_exist = os.path.exists(model_path) and os.path.exists(person_static_scaler_path) and os.path.exists(person_dynamic_scaler_path)
    if model_and_scalers_exist and not overwrite_model:
        logger.info("Model %s already exists.", MODEL_NAME)
        return model_path, person_static_scaler_path, person_dynamic_scaler_path

    logger.info("\t Loading data...")
    mz_persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "car_availability", "employed", "income_class", "home_x", "home_y", "work_x", "work_y", "weekend", "person_weight"
    ]]
    constants = context.stage("data.constants")
    mz_persons["car_availability"] = (mz_persons["car_availability"] != constants.CAR_AVAILABILITY_NEVER)

    mz_trips, _ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "destination_x", "destination_y", "origin_purpose", "purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")[[
        "person_id", "trip_id", "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip", 
        "trip_position_class", "departure_time_normalized", "daily_longest_distance_from_work",
        "activity_duration_h", "activity_chain"
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    inside_ch = within_ch(context, mz_trips, cols1=["origin_x", "origin_y"], cols2=["destination_x", "destination_y"])
    mz_trips = mz_trips[inside_ch].reset_index(drop=True)

    h3_data, h3_geo, _ = context.stage("synthesis.population.spatial.secondary_nn.h3")
    trips_h3 = h3_data["microcensus_trips"][["person_id", "trip_id", "destination_level_0"]]
    h3_geo_level0 = h3_geo["level_0"]

    required_h3_cols = [
        "centroid", "outside_fraction", "num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure", "sport", 
        "gastronomy", "accommodation", "cultural", "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none",
    ]
    missing_h3_cols = [col for col in required_h3_cols if col not in h3_geo_level0.columns]
    if missing_h3_cols:
        raise RuntimeError("Missing required destination-derived H3 columns in h3 stage: " + ", ".join(missing_h3_cols))

    all_h3 = h3_geo_level0["h3_index"].tolist()
    num_h3 = len(all_h3)
    h3_to_index = {h3_idx: idx for idx, h3_idx in enumerate(all_h3)}

    h3_indexed = h3_geo_level0.set_index("h3_index").reindex(all_h3)
    centroids = h3_indexed["centroid"]
    centroid_x = centroids.x.to_numpy(dtype=np.float64)
    centroid_y = centroids.y.to_numpy(dtype=np.float64)
    outside_fraction = h3_indexed["outside_fraction"].to_numpy(dtype=np.float64)
    statent_per_h3 = h3_indexed["num_statent"].to_numpy(dtype=np.float64)
    employees_per_h3 = h3_indexed["employees"].to_numpy(dtype=np.float64)
    urban_core_per_h3 = h3_indexed["urban_core"].to_numpy(dtype=np.float64)
    urban_per_h3 = h3_indexed["urban"].to_numpy(dtype=np.float64)
    education_per_h3 = h3_indexed["education"].to_numpy(dtype=np.float64)
    shop_per_h3 = h3_indexed["shop"].to_numpy(dtype=np.float64)
    leisure_per_h3 = h3_indexed["leisure"].to_numpy(dtype=np.float64)
    sport_per_h3 = h3_indexed["sport"].to_numpy(dtype=np.float64)
    gastronomy_per_h3 = h3_indexed["gastronomy"].to_numpy(dtype=np.float64)
    accommodation_per_h3 = h3_indexed["accommodation"].to_numpy(dtype=np.float64)
    cultural_per_h3 = h3_indexed["cultural"].to_numpy(dtype=np.float64)
    ovgk_share_a_per_h3 = h3_indexed["ovgk_share_a"].to_numpy(dtype=np.float64)
    ovgk_share_b_per_h3 = h3_indexed["ovgk_share_b"].to_numpy(dtype=np.float64)
    ovgk_share_c_per_h3 = h3_indexed["ovgk_share_c"].to_numpy(dtype=np.float64)
    ovgk_share_d_per_h3 = h3_indexed["ovgk_share_d"].to_numpy(dtype=np.float64)
    ovgk_share_none_per_h3 = h3_indexed["ovgk_share_none"].to_numpy(dtype=np.float64)

    person_cols = ["person_id", "age", "sex", "employed", "car_availability", "income_class", "home_x", "home_y", "work_x", "work_y"]
    trip_cols = [
        "person_id", "trip_id", "origin_x", "origin_y", "destination_level_0", "purpose","origin_purpose",
        "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip", "trip_position_class",
        "departure_time_normalized", "daily_longest_distance_from_work",
        "activity_duration_h", "activity_chain"
    ]
    df_trips = mz_trips.merge(trips_h3, on=["person_id", "trip_id"], how="left")
    df_trips = df_trips[df_trips["purpose"].isin(SECONDARY_ACTIVITIES)].dropna(subset=["destination_level_0"])
    df_trips = df_trips[trip_cols].merge(mz_persons[person_cols + ["person_weight"]], on="person_id", how="left")

    y_series = df_trips["destination_level_0"].map(h3_to_index)
    valid = y_series.notna()
    if (~valid).any():
        logger.warning("\t Dropping %s trips with destination H3 not found in candidate set.", int((~valid).sum()))
    df_trips = df_trips[valid].reset_index(drop=True)
    y = y_series[valid].astype(np.int64).to_numpy()
    weights = df_trips["person_weight"].to_numpy(dtype=np.float32)

    n_trips = len(df_trips)
    if n_trips == 0:
        raise RuntimeError("No valid secondary trips available for coarse model training.")
    logger.info("\t %s secondary trips with valid H3 destinations.", n_trips)

    home_x = df_trips["home_x"].to_numpy(dtype=np.float64)
    home_y = df_trips["home_y"].to_numpy(dtype=np.float64)
    work_x = df_trips["work_x"].to_numpy(dtype=np.float64)
    work_y = df_trips["work_y"].to_numpy(dtype=np.float64)
    has_work = np.isfinite(work_x) & np.isfinite(work_y)
    work_x = np.where(has_work, work_x, 0.0)
    work_y = np.where(has_work, work_y, 0.0)
    origin_x = df_trips["origin_x"].to_numpy(dtype=np.float64)
    origin_y = df_trips["origin_y"].to_numpy(dtype=np.float64)

    age = df_trips["age"].to_numpy(dtype=np.float64)
    income_class = df_trips["income_class"].to_numpy(dtype=np.float64)
    daily_longest = df_trips["daily_longest_distance_from_home"].to_numpy(dtype=np.float64)
    daily_longest = np.where(np.isfinite(daily_longest) & (daily_longest >= 0.0), daily_longest, 0.0)
    daily_total = df_trips["daily_crowfly_total"].to_numpy(dtype=np.float64)
    daily_total = np.where(np.isfinite(daily_total) & (daily_total >= 0.0), daily_total, 0.0)
    consumed_before = df_trips["crowfly_consumed_before_trip"].to_numpy(dtype=np.float64)
    consumed_before = np.where(np.isfinite(consumed_before) & (consumed_before >= 0.0), consumed_before, 0.0)
    trip_position = df_trips["trip_position_class"].to_numpy(dtype=np.float64)
    trip_position = np.where(np.isfinite(trip_position), trip_position, 2.0)
    sex = df_trips["sex"].to_numpy(dtype=np.float32)
    employed = df_trips["employed"].to_numpy(dtype=np.float32)
    car_availability = df_trips["car_availability"].to_numpy(dtype=np.float32)
    daily_longest_work = df_trips["daily_longest_distance_from_work"].to_numpy(dtype=np.float64)
    daily_longest_work = np.where(np.isfinite(daily_longest_work) & (daily_longest_work >= 0.0), daily_longest_work, 0.0)
    departure_time = df_trips["departure_time_normalized"].to_numpy(dtype=np.float64)
    departure_time = np.where(np.isfinite(departure_time), departure_time, 0.5)
    activity_duration_h = df_trips["activity_duration_h"].to_numpy(dtype=np.float64)
    activity_duration_h = np.where(np.isfinite(activity_duration_h) & (activity_duration_h >= 0.0), activity_duration_h, 0.0)
    activity_chain_matrix = np.stack([np.asarray(v, dtype=np.float64)[:ACTIVITY_CHAIN_N] if isinstance(v, np.ndarray) else np.zeros(ACTIVITY_CHAIN_N, dtype=np.float64) for v in df_trips["activity_chain"].to_numpy()])
    activity_chain_matrix = np.where(np.isfinite(activity_chain_matrix) & (activity_chain_matrix >= 0.0), activity_chain_matrix, 0.0)

    ################ Building tensors and fitting scalers ################
    logger.info("\t Building person-trip matrix...")
    purpose_categories = [str(p) for p in SECONDARY_ACTIVITIES]
    person_trip_matrix, static_matrix, dynamic_matrix, person_static_scaler, person_dynamic_scaler, person_trip_cols = fit_person_trip_matrix(age=age, sex=sex, 
                                                                                 employed=employed, car_availability=car_availability, income_class=income_class,
                                                                                 daily_longest=daily_longest, daily_total=daily_total, daily_longest_work=daily_longest_work,
                                                                                 activity_chain_matrix=activity_chain_matrix,
                                                                                 consumed_before=consumed_before, trip_position=trip_position, departure_time=departure_time,
                                                                                 activity_duration_h=activity_duration_h,
                                                                                 purpose_series=df_trips["purpose"], origin_purpose_series=df_trips["origin_purpose"],
                                                                                 purpose_categories=purpose_categories)

    logger.info("\t Building candidate tensor with Numba...")
    candidate_tensor = build_coarse_candidate_batch_numba(home_x, home_y, work_x, work_y, has_work, origin_x, origin_y, centroid_x, centroid_y, statent_per_h3, employees_per_h3,
                                                          urban_core_per_h3, urban_per_h3, education_per_h3, shop_per_h3, leisure_per_h3, sport_per_h3, gastronomy_per_h3,
                                                          accommodation_per_h3, cultural_per_h3, ovgk_share_a_per_h3, ovgk_share_b_per_h3,
                                                          ovgk_share_c_per_h3, ovgk_share_d_per_h3, ovgk_share_none_per_h3, outside_fraction)
    valid_mask = np.ones((n_trips, num_h3), dtype=bool)
    candidate_tensor, candidate_static_scaler, candidate_dynamic_scaler = fit_candidate_tensor(candidate_tensor, valid_mask, random_state=int(context.config("random_seed")))

    ################ Training the models ################
    logger.info("\t Training coarse two-input model...")
    seed = int(context.config("random_seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidate_static_x  = candidate_tensor[0:1, :, N_CANDIDATE_DYNAMIC:]  # [1, num_h3, 13] — static cols are identical for all rows; broadcast avoids redundant storage
    candidate_dynamic_x = candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC]    # [n_trips, num_h3, 3] — per-trip distances

    model = NeuralChoiceModel(person_input_dim=person_trip_matrix.shape[1], candidate_input_dim=candidate_tensor.shape[2], person_hidden_dim=32, hidden_dim=64)
    train_choice_model(model=model, person_static_x=static_matrix, person_dynamic_x=dynamic_matrix, candidate_static_x=candidate_static_x, candidate_dynamic_x=candidate_dynamic_x,
                              y=y, valid_mask=valid_mask, logger_instance=logger, weights=weights,
                              epochs=int(context.config("regional_model_epochs")),
                              batch_size=int(context.config("regional_model_batch_size")),
                              lr=float(context.config("regional_model_learning_rate")),
                              weight_decay=1e-3,
                              num_threads=int(context.config("threads")))

    ########## Building wrapper and saving model ##########
    static_feature_map = {
        "num_statent": statent_per_h3,
        "employees": employees_per_h3,
        "urban_core": urban_core_per_h3,
        "urban": urban_per_h3,
        "education": education_per_h3,
        "shop": shop_per_h3,
        "leisure": leisure_per_h3,
        "sport": sport_per_h3,
        "gastronomy": gastronomy_per_h3,
        "accommodation": accommodation_per_h3,
        "cultural": cultural_per_h3,
        "ovgk_share_a": ovgk_share_a_per_h3,
        "ovgk_share_b": ovgk_share_b_per_h3,
        "ovgk_share_c": ovgk_share_c_per_h3,
        "ovgk_share_d": ovgk_share_d_per_h3,
        "ovgk_share_none": ovgk_share_none_per_h3,
        "outside_fraction": outside_fraction,
    }
    static_candidate_features = np.column_stack([static_feature_map[name] for name in STATIC_CANDIDATE_FEATURES]).astype(np.float64)

    wrapper = RegionalChoiceWrapper(model=model, person_static_scaler=person_static_scaler, person_dynamic_scaler=person_dynamic_scaler,
                                          candidate_static_scaler=candidate_static_scaler, candidate_dynamic_scaler=candidate_dynamic_scaler,
                                          person_trip_cols=person_trip_cols, candidate_cols=CANDIDATE_FEATURES,
                                          all_h3=all_h3, centroid_x=centroid_x, centroid_y=centroid_y,
                                          static_candidate_features=static_candidate_features, purpose_categories=purpose_categories)
    wrapper.save(model_path)

    joblib.dump(person_static_scaler,  person_static_scaler_path)
    joblib.dump(person_dynamic_scaler, person_dynamic_scaler_path)
    ########## analysis and plots ##########
    _ = plot_analysis(context, wrapper, person_trip_matrix, candidate_tensor, y, all_h3, h3_geo_level0, centroid_x, centroid_y, home_x, home_y, origin_x, origin_y, h3_to_index, n_trips)

    return model_path, person_static_scaler_path, person_dynamic_scaler_path


def plot_analysis(context, wrapper, person_trip_matrix, candidate_tensor, y, all_h3, h3_geo, centroid_x, centroid_y, home_x, home_y, origin_x, origin_y, h3_to_index, n_trips):
    logger.info("Predicting on training data and plotting...")
    pred_idx = wrapper.predict_from_inputs(person_trip_matrix, candidate_tensor[:, :, N_CANDIDATE_DYNAMIC:], candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC], rng=np.random)
    predicted_h3 = [all_h3[int(i)] for i in pred_idx]

    real_h3_list = [all_h3[i] for i in y]
    real_counts = pd.Series(real_h3_list).value_counts().rename("real_count")
    pred_counts = pd.Series(predicted_h3).value_counts().rename("pred_count")
    counts_df = pd.DataFrame({"real_count": real_counts, "pred_count": pred_counts}).fillna(0)
    h3_geo_counts = h3_geo.set_index("h3_index").join(counts_df, how="left").fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h3_geo_counts.plot(column="real_count", ax=axes[0], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[0].set_title("Real H3 Counts")
    h3_geo_counts.plot(column="pred_count", ax=axes[1], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[1].set_title("Predicted H3 Counts")
    plt.tight_layout()
    plot_path = os.path.join(context.path(), "h3_counts_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    pred_indices = [h3_to_index[h3] for h3 in predicted_h3]
    real_dist_home = []
    real_dist_last = []
    pred_dist_home = []
    pred_dist_last = []

    for i in range(n_trips):
        real_idx = y[i]
        real_cx = centroid_x[real_idx]
        real_cy = centroid_y[real_idx]
        real_dist_home.append(np.sqrt((real_cx - home_x[i]) ** 2 + (real_cy - home_y[i]) ** 2))
        real_dist_last.append(np.sqrt((real_cx - origin_x[i]) ** 2 + (real_cy - origin_y[i]) ** 2))

        pred_idx_i = pred_indices[i]
        pred_cx = centroid_x[pred_idx_i]
        pred_cy = centroid_y[pred_idx_i]
        pred_dist_home.append(np.sqrt((pred_cx - home_x[i]) ** 2 + (pred_cy - home_y[i]) ** 2))
        pred_dist_last.append(np.sqrt((pred_cx - origin_x[i]) ** 2 + (pred_cy - origin_y[i]) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    threshold = 80000
    real_dist_home = np.array(real_dist_home)
    real_dist_last = np.array(real_dist_last)
    pred_dist_home = np.array(pred_dist_home)
    pred_dist_last = np.array(pred_dist_last)
    real_dist_home = real_dist_home[real_dist_home <= threshold]
    real_dist_last = real_dist_last[real_dist_last <= threshold]
    pred_dist_home = pred_dist_home[pred_dist_home <= threshold]
    pred_dist_last = pred_dist_last[pred_dist_last <= threshold]

    axes[0].hist(real_dist_home, bins=50, histtype="step", color="black", linewidth=2, label="Real", density=True)
    axes[0].hist(pred_dist_home, bins=50, histtype="step", color="red", linewidth=1, label="Predicted", density=True, linestyle="dashed")
    axes[0].set_title("Distance from Home")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    axes[1].hist(real_dist_last, bins=50, histtype="step", color="black", linewidth=2, label="Real", density=True)
    axes[1].hist(pred_dist_last, bins=50, histtype="step", color="red", linewidth=1, label="Predicted", density=True, linestyle="dashed")
    axes[1].set_title("Distance from Last")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    dist_plot_path = os.path.join(context.path(), "distance_distributions.png")
    plt.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return h3_geo_counts