import os
import logging

import joblib
import numpy as np
import pandas as pd
import torch

from .h3 import H3_LEVEL_NAMES
from .hierarchical_utils import (
    SECONDARY_ACTIVITIES,
    build_level2_children_by_level1,
    build_level2_candidate_attributes_by_level1,
    sanitize_work_coordinates,
    build_hierarchical_candidate_batch_numba,
)
from .feature_encoding import (
    CANDIDATE_FEATURES,
    STATIC_CANDIDATE_FEATURES,
    N_CANDIDATE_DYNAMIC,
    ACTIVITY_CHAIN_N,
    fit_candidate_tensor,
    fit_person_trip_matrix,
    add_detour_factor_feature,
)
from .choice_model import NeuralChoiceModel, train_choice_model
from .model_wrappers import ShortRangeChoiceWrapper

logger = logging.getLogger("synpp: short_range_model")

MODEL_NAME = "short_range_model.pt"
DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION = 1.0
SHORT_RANGE_MODEL_THRESHOLD = 1500.0

def configure(context):
    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.stage("synthesis.population.spatial.secondary.detour_factors.factors")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    context.stage("synthesis.population.spatial.secondary_nn.regional_model")

    context.config("threads")
    context.config("random_seed")

    context.config("overwrite_short_range_model_if_exists", True)
    context.config("short_range_model_batch_size", 256)
    context.config("short_range_model_epochs", 60)
    context.config("short_range_model_learning_rate", 4e-3)
    context.config("short_range_trip_threshold_m", SHORT_RANGE_MODEL_THRESHOLD)
    context.config("short_range_trip_min_m", DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION)
    context.config("secondary_nn_distance_loss_weight", 0.07)
    context.config("secondary_nn_distance_loss_short_floor_m", 100.0)


def execute(context):
    logger.info("Training short-range level2 model (<= threshold distance)...")

    overwrite_model = context.config("overwrite_short_range_model_if_exists")
    model_path = os.path.join(context.working_directory, MODEL_NAME)
    if os.path.exists(model_path) and not overwrite_model:
        logger.info("Model %s already exists.", MODEL_NAME)
        return (model_path,)

    logger.info("\t Loading data...")
    mz_persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "car_availability", "employed", "income_class", "home_x", "home_y", "work_x", "work_y", "weekend", "person_weight"
    ]]
    constants = context.stage("data.constants")
    mz_persons["car_availability"] = (mz_persons["car_availability"] != constants.CAR_AVAILABILITY_NEVER)

    mz_trips = context.stage("data.microcensus.trips")[0]
    mz_trips = mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "purpose", "origin_purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")[[
        "person_id", "trip_id", "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip", "trip_position_class",
        "departure_time_normalized", "daily_longest_distance_from_work",
        "activity_duration_h", "target_distance", "trip_destination_distance_from_home", "activity_chain"
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    h3_data, h3_geo, h3_tree = context.stage("synthesis.population.spatial.secondary_nn.h3")
    trips_h3 = h3_data["microcensus_trips"][["person_id", "trip_id", f"destination_{H3_LEVEL_NAMES[0]}", f"destination_{H3_LEVEL_NAMES[1]}", f"destination_{H3_LEVEL_NAMES[-1]}"]]
    h3_geo_level2 = h3_geo[H3_LEVEL_NAMES[-1]]

    required_h3_cols = [
        "centroid", "outside_fraction", "num_statent", "employees", "urban_core", "urban", "education", "shop", "leisure",
        "sport", "gastronomy", "accommodation", "cultural",
        "ovgk_share_a", "ovgk_share_b", "ovgk_share_c", "ovgk_share_d", "ovgk_share_none",
    ]
    missing_h3_cols = [col for col in required_h3_cols if col not in h3_geo_level2.columns]
    if missing_h3_cols:
        raise RuntimeError("Missing required destination-derived H3 columns in h3 stage: " + ", ".join(missing_h3_cols))

    h3_level2_indexed = h3_geo_level2.set_index("h3_index")
    centroids = h3_level2_indexed["centroid"]
    centroid_x_by_l2 = centroids.x.to_dict()
    centroid_y_by_l2 = centroids.y.to_dict()
    outside_fraction_by_l2 = h3_level2_indexed["outside_fraction"].to_dict()
    statent_count_l2 = h3_level2_indexed["num_statent"].to_dict()
    employees_count_l2 = h3_level2_indexed["employees"].to_dict()
    urban_core_count_l2 = h3_level2_indexed["urban_core"].to_dict()
    urban_count_l2 = h3_level2_indexed["urban"].to_dict()
    education_count_l2 = h3_level2_indexed["education"].to_dict()
    shop_count_l2 = h3_level2_indexed["shop"].to_dict()
    leisure_count_l2 = h3_level2_indexed["leisure"].to_dict()
    sport_count_l2 = h3_level2_indexed["sport"].to_dict()
    gastronomy_count_l2 = h3_level2_indexed["gastronomy"].to_dict()
    accommodation_count_l2 = h3_level2_indexed["accommodation"].to_dict()
    cultural_count_l2 = h3_level2_indexed["cultural"].to_dict()
    ovgk_share_a_by_l2 = h3_level2_indexed["ovgk_share_a"].to_dict()
    ovgk_share_b_by_l2 = h3_level2_indexed["ovgk_share_b"].to_dict()
    ovgk_share_c_by_l2 = h3_level2_indexed["ovgk_share_c"].to_dict()
    ovgk_share_d_by_l2 = h3_level2_indexed["ovgk_share_d"].to_dict()
    ovgk_share_none_by_l2 = h3_level2_indexed["ovgk_share_none"].to_dict()

    if h3_tree is None:
        raise RuntimeError("H3 hierarchy tree is missing from H3 stage output. Cannot train short-range model.")
    children_by_level1 = build_level2_children_by_level1(h3_tree, centroid_x_by_l2, centroid_y_by_l2)
    if len(children_by_level1) == 0:
        raise RuntimeError("H3 hierarchy tree has no valid level2 children with centroids. Cannot train short-range model.")

    level2_candidate_attributes_by_level1 = build_level2_candidate_attributes_by_level1(
        children_by_level1,
        centroid_x_by_l2,
        centroid_y_by_l2,
        statent_count_l2,
        employees_count_l2,
        urban_core_count_l2,
        urban_count_l2,
        education_count_l2,
        shop_count_l2,
        leisure_count_l2,
        sport_count_l2,
        gastronomy_count_l2,
        accommodation_count_l2,
        cultural_count_l2,
        ovgk_share_a_by_l2,
        ovgk_share_b_by_l2,
        ovgk_share_c_by_l2,
        ovgk_share_d_by_l2,
        ovgk_share_none_by_l2,
        outside_fraction_by_l2,
    )

    logger.info("\t Preparing short-range training set...")
    df = mz_trips.merge(trips_h3, on=["person_id", "trip_id"], how="left")
    df = df.merge(mz_persons, on="person_id", how="left")
    df = df[
        df["purpose"].isin(SECONDARY_ACTIVITIES)
    ].dropna(
        subset=[f"destination_{H3_LEVEL_NAMES[0]}", f"destination_{H3_LEVEL_NAMES[1]}", f"destination_{H3_LEVEL_NAMES[-1]}"]
    ).reset_index(drop=True)

    short_max = float(context.config("short_range_trip_threshold_m"))
    short_min = float(context.config("short_range_trip_min_m"))
    td = df["target_distance"].to_numpy(dtype=np.float64)
    td_valid = np.isfinite(td)
    df = df.loc[td_valid].copy()
    df = df[(df["target_distance"] >= short_min) & (df["target_distance"] <= short_max)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No short-range training samples remain after target-distance filtering.")

    valid_rows = []
    for idx, row in df.iterrows():
        key = (row[f"destination_{H3_LEVEL_NAMES[0]}"], row[f"destination_{H3_LEVEL_NAMES[1]}"])
        children = children_by_level1.get(key, [])
        if len(children) < 2:
            continue
        if row[f"destination_{H3_LEVEL_NAMES[-1]}"] not in children:
            continue
        valid_rows.append(idx)

    if len(valid_rows) == 0:
        raise RuntimeError("No valid short-range samples after filtering by hierarchy.")
    df = df.iloc[valid_rows].reset_index(drop=True)

    max_children = max(
        len(children_by_level1[(l0, l1)])
        for l0, l1 in df[[f"destination_{H3_LEVEL_NAMES[0]}", f"destination_{H3_LEVEL_NAMES[1]}"]].itertuples(index=False)
    )
    n_samples = len(df)

    cand_x = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_y = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_statent = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_employees = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_urban_core = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_urban = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_education = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_shop = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_leisure = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_sport = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_gastronomy = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_accommodation = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_cultural = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_a = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_b = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_c = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_d = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_none = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_outside_fraction = np.zeros((n_samples, max_children), dtype=np.float64)
    valid_mask = np.zeros((n_samples, max_children), dtype=np.bool_)
    y = np.zeros(n_samples, dtype=np.int64)
    weights = df["person_weight"].to_numpy(dtype=np.float32)

    with context.progress(total=n_samples, label="Short-range model: building level2 choice sets") as progress:
        for i, row in df.iterrows():
            key = (row[f"destination_{H3_LEVEL_NAMES[0]}"], row[f"destination_{H3_LEVEL_NAMES[1]}"])
            chosen_level2 = row[f"destination_{H3_LEVEL_NAMES[-1]}"]
            attrs = level2_candidate_attributes_by_level1[key]
            children = attrs["children"]
            n_children = len(children)

            valid_mask[i, :n_children] = True
            cand_x[i, :n_children] = attrs["x"]
            cand_y[i, :n_children] = attrs["y"]
            cand_statent[i, :n_children] = attrs["num_statent"]
            cand_employees[i, :n_children] = attrs["employees"]
            cand_urban_core[i, :n_children] = attrs["urban_core"]
            cand_urban[i, :n_children] = attrs["urban"]
            cand_education[i, :n_children] = attrs["education"]
            cand_shop[i, :n_children] = attrs["shop"]
            cand_leisure[i, :n_children] = attrs["leisure"]
            cand_sport[i, :n_children] = attrs["sport"]
            cand_gastronomy[i, :n_children] = attrs["gastronomy"]
            cand_accommodation[i, :n_children] = attrs["accommodation"]
            cand_cultural[i, :n_children] = attrs["cultural"]
            cand_ovgk_share_a[i, :n_children] = attrs["ovgk_share_a"]
            cand_ovgk_share_b[i, :n_children] = attrs["ovgk_share_b"]
            cand_ovgk_share_c[i, :n_children] = attrs["ovgk_share_c"]
            cand_ovgk_share_d[i, :n_children] = attrs["ovgk_share_d"]
            cand_ovgk_share_none[i, :n_children] = attrs["ovgk_share_none"]
            cand_outside_fraction[i, :n_children] = attrs["outside_fraction"]

            y[i] = attrs["index_by_child"][chosen_level2]
            progress.update()

    home_x = df["home_x"].to_numpy(dtype=np.float64)
    home_y = df["home_y"].to_numpy(dtype=np.float64)
    work_x = df["work_x"].to_numpy(dtype=np.float64)
    work_y = df["work_y"].to_numpy(dtype=np.float64)
    has_work, work_x, work_y = sanitize_work_coordinates(work_x, work_y)
    origin_x = df["origin_x"].to_numpy(dtype=np.float64)
    origin_y = df["origin_y"].to_numpy(dtype=np.float64)

    age = df["age"].to_numpy(dtype=np.float64)
    income_class = df["income_class"].to_numpy(dtype=np.float64)
    daily_longest = df["daily_longest_distance_from_home"].to_numpy(dtype=np.float64)
    daily_longest = np.where(np.isfinite(daily_longest) & (daily_longest >= 0.0), daily_longest, 0.0)
    daily_total = df["daily_crowfly_total"].to_numpy(dtype=np.float64)
    daily_total = np.where(np.isfinite(daily_total) & (daily_total >= 0.0), daily_total, 0.0)
    consumed_before = df["crowfly_consumed_before_trip"].to_numpy(dtype=np.float64)
    consumed_before = np.where(np.isfinite(consumed_before) & (consumed_before >= 0.0), consumed_before, 0.0)
    trip_position = df["trip_position_class"].to_numpy(dtype=np.float64)
    trip_position = np.where(np.isfinite(trip_position), trip_position, 2.0)
    sex = df["sex"].to_numpy(dtype=np.float32)
    employed = df["employed"].to_numpy(dtype=np.float32)
    car_availability = df["car_availability"].to_numpy(dtype=np.float32)
    daily_longest_work = df["daily_longest_distance_from_work"].to_numpy(dtype=np.float64)
    daily_longest_work = np.where(np.isfinite(daily_longest_work) & (daily_longest_work >= 0.0), daily_longest_work, 0.0)
    departure_time = df["departure_time_normalized"].to_numpy(dtype=np.float64)
    departure_time = np.where(np.isfinite(departure_time), departure_time, 0.5)
    activity_duration_h = df["activity_duration_h"].to_numpy(dtype=np.float64)
    activity_duration_h = np.where(np.isfinite(activity_duration_h) & (activity_duration_h >= 0.0), activity_duration_h, 0.0)
    target_distance = df["target_distance"].to_numpy(dtype=np.float64)
    target_distance = np.where(np.isfinite(target_distance) & (target_distance >= 0.0), target_distance, 0.0)
    target_home_distance = df["trip_destination_distance_from_home"].to_numpy(dtype=np.float64)
    target_home_distance = np.where(np.isfinite(target_home_distance) & (target_home_distance >= 0.0), target_home_distance, 0.0)
    activity_chain_matrix = np.stack([
        np.asarray(v, dtype=np.float64)[:ACTIVITY_CHAIN_N] if isinstance(v, np.ndarray) else np.zeros(ACTIVITY_CHAIN_N, dtype=np.float64)
        for v in df["activity_chain"].to_numpy()
    ])
    activity_chain_matrix = np.where(np.isfinite(activity_chain_matrix) & (activity_chain_matrix >= 0.0), activity_chain_matrix, 0.0)

    _, person_static_scaler_path, person_dynamic_scaler_path = context.stage("synthesis.population.spatial.secondary_nn.regional_model")
    person_static_scaler = joblib.load(person_static_scaler_path)
    person_dynamic_scaler = joblib.load(person_dynamic_scaler_path)

    purpose_categories = [str(purpose) for purpose in SECONDARY_ACTIVITIES]
    person_trip_matrix, static_matrix, dynamic_matrix, person_static_scaler, person_dynamic_scaler, person_trip_cols = fit_person_trip_matrix(
        age=age,
        sex=sex,
        employed=employed,
        car_availability=car_availability,
        income_class=income_class,
        daily_longest=daily_longest,
        daily_total=daily_total,
        daily_longest_work=daily_longest_work,
        activity_chain_matrix=activity_chain_matrix,
        consumed_before=consumed_before,
        trip_position=trip_position,
        departure_time=departure_time,
        activity_duration_h=activity_duration_h,
        target_distance=target_distance,
        purpose_series=df["purpose"],
        origin_purpose_series=df["origin_purpose"],
        purpose_categories=purpose_categories,
        person_static_scaler=person_static_scaler,
        person_dynamic_scaler=person_dynamic_scaler,
    )

    logger.info("\t Computing candidate-hex features with Numba...")
    candidate_tensor = build_hierarchical_candidate_batch_numba(
        home_x,
        home_y,
        work_x,
        work_y,
        has_work,
        origin_x,
        origin_y,
        cand_x,
        cand_y,
        cand_statent,
        cand_employees,
        cand_urban_core,
        cand_urban,
        cand_education,
        cand_shop,
        cand_leisure,
        cand_sport,
        cand_gastronomy,
        cand_accommodation,
        cand_cultural,
        cand_ovgk_share_a,
        cand_ovgk_share_b,
        cand_ovgk_share_c,
        cand_ovgk_share_d,
        cand_ovgk_share_none,
        cand_outside_fraction,
        valid_mask,
    )
    candidate_tensor = add_detour_factor_feature(
        candidate_tensor, origin_x, origin_y, cand_x, cand_y, valid_mask,
        context.stage("synthesis.population.spatial.secondary.detour_factors.factors"),
    )
    candidate_dist_home_m = candidate_tensor[:, :, 0].astype(np.float32)
    candidate_dist_last_m = candidate_tensor[:, :, 2].astype(np.float32)
    candidate_tensor, candidate_static_scaler, candidate_dynamic_scaler = fit_candidate_tensor(candidate_tensor, valid_mask)

    logger.info("\t Training short-range neural choice model...")
    seed = int(context.config("random_seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidate_static_x = candidate_tensor[:, :, N_CANDIDATE_DYNAMIC:]
    candidate_dynamic_x = candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC]

    model = NeuralChoiceModel(
        person_input_dim=person_trip_matrix.shape[1],
        candidate_input_dim=candidate_tensor.shape[2],
        person_hidden_dim=32,
        hidden_dim=32,
    )
    train_choice_model(
        model=model,
        person_static_x=static_matrix,
        person_dynamic_x=dynamic_matrix,
        candidate_static_x=candidate_static_x,
        candidate_dynamic_x=candidate_dynamic_x,
        y=y,
        valid_mask=valid_mask,
        logger_instance=logger,
        weights=weights,
        epochs=int(context.config("short_range_model_epochs")),
        batch_size=int(context.config("short_range_model_batch_size")),
        lr=float(context.config("short_range_model_learning_rate")),
        num_threads=int(context.config("threads")),
        path=context.path(),
        distance_candidates=candidate_dist_last_m,
        distance_targets=target_distance.astype(np.float32),
        distance_candidates_home=candidate_dist_home_m,
        distance_targets_home=target_home_distance.astype(np.float32),
        distance_loss_weight=float(context.config("secondary_nn_distance_loss_weight")),
        distance_loss_short_floor_m=float(context.config("secondary_nn_distance_loss_short_floor_m")),
    )

    all_h3 = h3_geo_level2["h3_index"].astype(str).tolist()
    h3_all_indexed = h3_geo_level2.set_index("h3_index").loc[all_h3]
    all_centroids = h3_all_indexed["centroid"]
    centroid_x = all_centroids.x.to_numpy(dtype=np.float64)
    centroid_y = all_centroids.y.to_numpy(dtype=np.float64)

    static_candidate_features = np.column_stack([
        h3_all_indexed[name].to_numpy(dtype=np.float64) for name in STATIC_CANDIDATE_FEATURES
    ])

    wrapper = ShortRangeChoiceWrapper(
        model=model,
        person_static_scaler=person_static_scaler,
        person_dynamic_scaler=person_dynamic_scaler,
        candidate_static_scaler=candidate_static_scaler,
        candidate_dynamic_scaler=candidate_dynamic_scaler,
        person_trip_cols=person_trip_cols,
        candidate_cols=CANDIDATE_FEATURES,
        all_h3=all_h3,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        static_candidate_features=static_candidate_features,
        purpose_categories=purpose_categories,
    )
    wrapper.save(model_path)

    logger.info("Short-range model saved to %s", model_path)
    return (model_path,)
