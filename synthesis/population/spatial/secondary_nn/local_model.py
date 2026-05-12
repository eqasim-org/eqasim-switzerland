import os
import logging

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from .hierarchical_utils import SECONDARY_ACTIVITIES, build_level2_children_by_level1, build_level2_candidate_attributes_by_level1, sanitize_work_coordinates, build_hierarchical_candidate_batch_numba
from .feature_encoding import CANDIDATE_FEATURES, N_CANDIDATE_DYNAMIC, ACTIVITY_CHAIN_N, fit_candidate_tensor, fit_person_trip_matrix
from .choice_model import NeuralChoiceModel, train_choice_model
from .model_wrappers import LocalChoiceWrapper

logger = logging.getLogger("synpp: local_model")

MODEL_NAME = "local_model.pt"


def configure(context):
    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    context.stage("synthesis.population.spatial.secondary_nn.regional_model")

    context.config("threads")
    context.config("random_seed")

    # training params
    context.config("overwrite_local_model_if_exists", True)
    context.config("local_model_batch_size", 256)
    context.config("local_model_epochs", 50)
    context.config("local_model_learning_rate", 4e-3)


def execute(context):
    logger.info("Training local model (neural choice) for level2-within-level1 secondary location choice...")

    overwrite_model = context.config("overwrite_local_model_if_exists")
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

    mz_trips, _ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "purpose", "origin_purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")[[
        "person_id", "trip_id", "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip", "trip_position_class",
        "departure_time_normalized", "daily_longest_distance_from_work",
        "activity_duration_h", "activity_chain"
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    h3_data, h3_geo, h3_tree = context.stage("synthesis.population.spatial.secondary_nn.h3")
    trips_h3 = h3_data["microcensus_trips"][["person_id", "trip_id", "destination_level_0", "destination_level_1", "destination_level_2"]]
    h3_geo_level2 = h3_geo["level_2"]

    if "outside_fraction" not in h3_geo_level2.columns:
        raise RuntimeError("Missing outside_fraction in H3 level_2 geometry. Run h3 stage with outside_fraction enabled.")

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
        raise RuntimeError("H3 hierarchy tree is missing from H3 stage output. Cannot train detailed model.")
    children_by_level1 = build_level2_children_by_level1(h3_tree, centroid_x_by_l2, centroid_y_by_l2)
    if len(children_by_level1) == 0:
        raise RuntimeError("H3 hierarchy tree has no valid level2 children with centroids. Cannot train detailed model.")

    level2_candidate_attributes_by_level1 = build_level2_candidate_attributes_by_level1(children_by_level1, centroid_x_by_l2, centroid_y_by_l2, statent_count_l2, employees_count_l2,
                                                    urban_core_count_l2, urban_count_l2, education_count_l2, shop_count_l2, leisure_count_l2, sport_count_l2, gastronomy_count_l2,
                                                    accommodation_count_l2, cultural_count_l2, ovgk_share_a_by_l2, ovgk_share_b_by_l2,
                                                    ovgk_share_c_by_l2, ovgk_share_d_by_l2, ovgk_share_none_by_l2, outside_fraction_by_l2)

    logger.info("\t Preparing microcensus training set...")
    df = mz_trips.merge(trips_h3, on=["person_id", "trip_id"], how="left")
    df = df.merge(mz_persons, on="person_id", how="left")
    df = df[df["purpose"].isin(SECONDARY_ACTIVITIES)].dropna(subset=["destination_level_0", "destination_level_1", "destination_level_2"]).reset_index(drop=True)

    # Here we check that the level2 choice is consistent with the level1 choice and that there are at least 2 valid level2 alternatives for each level1. This ensures that the detailed model has a valid choice set to learn from for each sample. We filter out any samples that do not meet these criteria.
    valid_rows = []
    for idx, row in df.iterrows():
        key = (row["destination_level_0"], row["destination_level_1"])
        children = children_by_level1.get(key, [])
        if len(children) < 2:
            continue
        if row["destination_level_2"] not in children:
            continue
        valid_rows.append(idx)

    if len(valid_rows) == 0:
        raise RuntimeError("No valid samples for detailed model after filtering by level1-level2 hierarchy.")
    df = df.iloc[valid_rows].reset_index(drop=True)

    max_children = max(len(children_by_level1[(l0, l1)]) for l0, l1 in df[["destination_level_0", "destination_level_1"]].itertuples(index=False))
    n_samples = len(df)

    # Here we build candidate feature tensors with shape (n_samples, max_children, n_candidate_features) where max_children is the maximum number of level2 alternatives for any level1 in the training set. We also build a valid_mask tensor with shape (n_samples, max_children) that indicates which entries in the candidate tensor are valid alternatives for each sample. The target tensor y has shape (n_samples,) and contains the index of the chosen alternative among the valid ones for each sample.
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

    with context.progress(total=n_samples, label="Detailed model: building level2 choice sets") as progress:
        for i, row in df.iterrows():
            key = (row["destination_level_0"], row["destination_level_1"])
            chosen_level2 = row["destination_level_2"]
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
    activity_chain_matrix = np.stack([np.asarray(v, dtype=np.float64)[:ACTIVITY_CHAIN_N] if isinstance(v, np.ndarray) else np.zeros(ACTIVITY_CHAIN_N, dtype=np.float64) for v in df["activity_chain"].to_numpy()])
    activity_chain_matrix = np.where(np.isfinite(activity_chain_matrix) & (activity_chain_matrix >= 0.0), activity_chain_matrix, 0.0)
    
    _, person_static_scaler_path, person_dynamic_scaler_path = context.stage("synthesis.population.spatial.secondary_nn.regional_model")
    person_static_scaler  = joblib.load(person_static_scaler_path)
    person_dynamic_scaler = joblib.load(person_dynamic_scaler_path)

    ########### Building tensors and fitting scalers ###########
    purpose_categories = [str(purpose) for purpose in SECONDARY_ACTIVITIES]
    person_trip_matrix, static_matrix, dynamic_matrix, person_static_scaler, person_dynamic_scaler, person_trip_cols = fit_person_trip_matrix(
        age=age, sex=sex, employed=employed, car_availability=car_availability, income_class=income_class,
        daily_longest=daily_longest, daily_total=daily_total, daily_longest_work=daily_longest_work,
        activity_chain_matrix=activity_chain_matrix,
        consumed_before=consumed_before, trip_position=trip_position, departure_time=departure_time,
        activity_duration_h=activity_duration_h,
        purpose_series=df["purpose"], origin_purpose_series=df["origin_purpose"],
        purpose_categories=purpose_categories,
        person_static_scaler=person_static_scaler, person_dynamic_scaler=person_dynamic_scaler)

    logger.info("\t Computing candidate-hex features with Numba...")
    candidate_tensor = build_hierarchical_candidate_batch_numba(home_x, home_y, work_x, work_y, has_work, origin_x, origin_y, cand_x, cand_y, cand_statent, cand_employees,
                                                                cand_urban_core, cand_urban, cand_education, cand_shop, cand_leisure, cand_sport, cand_gastronomy,
                                                                cand_accommodation, cand_cultural, cand_ovgk_share_a, cand_ovgk_share_b,
                                                                cand_ovgk_share_c, cand_ovgk_share_d, cand_ovgk_share_none, cand_outside_fraction, valid_mask)
    candidate_tensor, candidate_static_scaler, candidate_dynamic_scaler = fit_candidate_tensor(candidate_tensor, valid_mask)

    ############ Training the model ###########
    logger.info("\t Training local neural choice model...")
    seed = int(context.config("random_seed"))
    np.random.seed(seed)
    torch.manual_seed(seed)

    candidate_static_x  = candidate_tensor[:, :, N_CANDIDATE_DYNAMIC:]
    candidate_dynamic_x = candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC]

    model = NeuralChoiceModel(person_input_dim=person_trip_matrix.shape[1], candidate_input_dim=candidate_tensor.shape[2], person_hidden_dim=32, hidden_dim=32)
    train_choice_model(model=model, person_static_x=static_matrix, person_dynamic_x=dynamic_matrix, candidate_static_x=candidate_static_x, candidate_dynamic_x=candidate_dynamic_x,
        y=y, valid_mask=valid_mask, logger_instance=logger, weights=weights,
        epochs=int(context.config("local_model_epochs")),
        batch_size=int(context.config("local_model_batch_size")),
        lr=float(context.config("local_model_learning_rate")),
        weight_decay=1e-3,
        num_threads=int(context.config("threads")))

    ########## Building wrapper and saving model ##########
    wrapper = LocalChoiceWrapper(model=model, person_static_scaler=person_static_scaler, person_dynamic_scaler=person_dynamic_scaler,
        candidate_static_scaler=candidate_static_scaler, candidate_dynamic_scaler=candidate_dynamic_scaler,
        person_trip_cols=person_trip_cols, candidate_cols=CANDIDATE_FEATURES,
        children_by_level1=children_by_level1, level2_candidate_attributes_by_level1=level2_candidate_attributes_by_level1, purpose_categories=purpose_categories)
    wrapper.save(model_path)

    ########### Plotting analysis of predictions ##########
    _ = plot_analysis(context=context, wrapper=wrapper, person_trip_matrix=person_trip_matrix, candidate_tensor=candidate_tensor, valid_mask=valid_mask, df=df,
                      children_by_level1=children_by_level1, h3_geo_level2=h3_geo_level2, centroid_x_by_l2=centroid_x_by_l2, centroid_y_by_l2=centroid_y_by_l2)

    logger.info("Detailed model saved to %s", model_path)
    return (model_path,)


def plot_analysis(context, wrapper, person_trip_matrix, candidate_tensor, valid_mask, df, children_by_level1, h3_geo_level2, centroid_x_by_l2, centroid_y_by_l2):
    logger.info("Predicting on training data and plotting level2 counts...")
    pred_idx = wrapper.predict_from_inputs(person_trip_matrix, candidate_tensor[:, :, N_CANDIDATE_DYNAMIC:], candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC], valid_mask, rng=None, return_probabilities=False)

    predicted_level2 = []
    for i, (level0, level1) in enumerate(df[["destination_level_0", "destination_level_1"]].itertuples(index=False)):
        children = children_by_level1[(level0, level1)]
        predicted_level2.append(children[int(pred_idx[i])])

    real_level2 = df["destination_level_2"].astype(str)
    real_counts = real_level2.value_counts().rename("real_count")
    pred_counts = pd.Series(predicted_level2).value_counts().rename("pred_count")
    counts_df = pd.DataFrame({"real_count": real_counts, "pred_count": pred_counts}).fillna(0)
    h3_geo_counts = h3_geo_level2.set_index("h3_index").join(counts_df, how="left").fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h3_geo_counts.plot(column="real_count", ax=axes[0], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[0].set_title("Real Level2 H3 Counts")
    h3_geo_counts.plot(column="pred_count", ax=axes[1], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[1].set_title("Predicted Level2 H3 Counts")
    plt.tight_layout()
    plot_path = os.path.join(context.path(), "detailed_level2_counts_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Plotting distance distributions for level2...")
    home_x = df["home_x"].to_numpy(dtype=np.float64)
    home_y = df["home_y"].to_numpy(dtype=np.float64)
    work_x = df["work_x"].to_numpy(dtype=np.float64)
    work_y = df["work_y"].to_numpy(dtype=np.float64)
    has_work = np.isfinite(work_x) & np.isfinite(work_y)

    real_dist_home = []
    pred_dist_home = []
    real_dist_work = []
    pred_dist_work = []

    real_level2_arr = df["destination_level_2"].astype(str).to_numpy()
    pred_level2_arr = np.asarray(predicted_level2, dtype=str)
    for i in range(len(df)):
        real_h3 = real_level2_arr[i]
        pred_h3 = pred_level2_arr[i]
        real_cx = centroid_x_by_l2.get(real_h3)
        real_cy = centroid_y_by_l2.get(real_h3)
        pred_cx = centroid_x_by_l2.get(pred_h3)
        pred_cy = centroid_y_by_l2.get(pred_h3)
        if real_cx is None or real_cy is None or pred_cx is None or pred_cy is None:
            continue

        real_dist_home.append(np.sqrt((real_cx - home_x[i]) ** 2 + (real_cy - home_y[i]) ** 2))
        pred_dist_home.append(np.sqrt((pred_cx - home_x[i]) ** 2 + (pred_cy - home_y[i]) ** 2))

        if has_work[i]:
            real_dist_work.append(np.sqrt((real_cx - work_x[i]) ** 2 + (real_cy - work_y[i]) ** 2))
            pred_dist_work.append(np.sqrt((pred_cx - work_x[i]) ** 2 + (pred_cy - work_y[i]) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    threshold = 80000
    real_dist_home = np.array(real_dist_home)
    pred_dist_home = np.array(pred_dist_home)
    real_dist_home = real_dist_home[real_dist_home <= threshold]
    pred_dist_home = pred_dist_home[pred_dist_home <= threshold]

    axes[0].hist(real_dist_home, bins=50, alpha=0.4, color="black", linewidth=2, label="Real", density=True, histtype="step")
    axes[0].hist(pred_dist_home, bins=50, alpha=0.4, color="red", linewidth=1, label="Predicted", density=True, histtype="step", linestyle="dashed")
    axes[0].set_title("Distance from Home")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    if len(real_dist_work) > 0:
        real_dist_work = np.array(real_dist_work)
        pred_dist_work = np.array(pred_dist_work)
        real_dist_work = real_dist_work[real_dist_work <= threshold]
        pred_dist_work = pred_dist_work[pred_dist_work <= threshold]
        axes[1].hist(real_dist_work, bins=50, alpha=0.4, color="black", linewidth=2, label="Real", density=True, histtype="step")
        axes[1].hist(pred_dist_work, bins=50, alpha=0.4, color="red", linewidth=1, label="Predicted", density=True, histtype="step", linestyle="dashed")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid work locations", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_title("Distance from Work (has work)")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    dist_plot_path = os.path.join(context.path(), "detailed_distance_distributions.png")
    plt.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return h3_geo_counts