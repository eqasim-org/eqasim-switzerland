import os
import logging
import json

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

from .h3 import H3_LEVEL_NAMES
from .hierarchical_utils import (
    SECONDARY_ACTIVITIES,
    DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION,
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

from .tools import (model_validation_options, training_mobility, enrich_candidate_tensor,
                    destination_offer_masks, destination_support_masks, origin_training_choice_sets)
from .training_split import expert_persons

logger = logging.getLogger("synpp: short_range_model")

MODEL_NAME = "short_range_model.pt"
SHORT_RANGE_MODEL_THRESHOLD = 1500.0

def configure(context):
    context.config("secondary_nn_model_validation_rotation_epochs", 3)
    context.config("secondary_nn_model_validation_holdout_fraction", 0.1)
    context.stage("synthesis.population.spatial.secondary_nn.mobility")
    context.config("secondary_nn_use_target_distance", True)
    context.config("secondary_nn_model_validation_folds", 5)
    context.config("secondary_nn_model_min_epochs", 30)
    context.config("secondary_nn_model_validation_strategy", "rotating_holdout")
    context.stage("data.spatial.ovgk")
    context.stage("synthesis.population.destinations")
    context.config("secondary_nn_distance_loss_kind", "crps_energy")
    context.config("secondary_nn_distribution_loss_weight", 0.1)
    context.config("secondary_nn_short_distribution_loss_weight", 0.05)
    context.config("secondary_nn_short_relative_distance_loss_weight", 0.05)
    context.config("secondary_nn_short_distance_loss_kind", "relative")
    context.config("secondary_nn_short_relative_underestimation_factor", 2.0)
    context.config("secondary_nn_distribution_scale_m", 1000.0)
    context.config("secondary_nn_energy_projections", 8)
    if not context.config("secondary_nn_use_target_distance"):
        context.config("secondary_nn_crossfit_fold", -1)
        context.stage("synthesis.population.spatial.secondary_nn.training_split")
    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.stage("synthesis.population.spatial.secondary.detour_factors.factors")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    context.stage("synthesis.population.spatial.secondary_nn.regional_model")

    context.config("threads", volatile=True)
    context.config("random_seed")

    context.config("overwrite_short_range_model_if_exists", True)
    context.config("short_range_model_batch_size", 256)
    context.config("short_range_model_epochs", 60)
    context.config("short_range_model_learning_rate", 4e-3)
    context.config("short_range_model_dropout_rate", 0.35)
    context.config("short_range_model_weight_decay", 1e-2)
    context.config("short_range_model_validation_folds", None)
    context.config("short_range_model_validation_strategy", None)
    context.config("short_range_model_min_epochs", None)
    context.config("short_range_model_validation_every", 2)
    context.config("short_range_model_early_stopping_patience", 5)
    context.config("short_range_model_early_stopping_min_delta", 1e-3)
    context.config("secondary_nn_short_trip_threshold_m", SHORT_RANGE_MODEL_THRESHOLD)
    context.config("secondary_nn_short_query_radius_m", 200.0)
    context.config("secondary_nn_short_min_candidates", 6)
    context.config("short_range_trip_min_m", DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION)
    context.config("secondary_nn_distance_loss_weight", 0.07)
    context.config("secondary_nn_distance_loss_short_floor_m", 100.0)


def execute(context):

    # Load expert inputs and prepare the fitting data.

    logger.info("Training short-range level2 model (<= threshold distance)...")

    overwrite_model = context.config("overwrite_short_range_model_if_exists")
    model_path = os.path.join(context.path(), MODEL_NAME)
    if os.path.exists(model_path) and not overwrite_model:
        logger.info("Model %s already exists.", MODEL_NAME)
        return (model_path,)

    logger.info("\t Loading data...")
    mz_persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "car_availability", "employed", "income_class", "home_x", "home_y", "work_x", "work_y", "weekend", "person_weight"
    ]]
    mz_persons = expert_persons(context, mz_persons)
    constants = context.stage("data.constants")
    mz_persons["car_availability"] = (mz_persons["car_availability"] != constants.CAR_AVAILABILITY_NEVER)

    mz_trips = context.stage("data.microcensus.trips")[0]
    mz_trips = mz_trips.loc[mz_trips.person_id.isin(mz_persons.person_id)].copy()
    mz_trips = mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "destination_x", "destination_y", "purpose", "origin_purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")[[
        "person_id", "trip_id", "daily_longest_distance_from_home", "daily_crowfly_total", "crowfly_consumed_before_trip", "trip_position_class",
        "departure_time_normalized", "daily_longest_distance_from_work",
        "activity_duration_h", "target_distance", "trip_destination_distance_from_home", "activity_chain", "chain_state"
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    # Build the same global leaf catalogue and facility eligibility as prediction.

    h3_data, h3_geo, _ = context.stage("synthesis.population.spatial.secondary_nn.h3")
    leaf_name = H3_LEVEL_NAMES[-1]
    target_column = f"destination_{leaf_name}"
    trips_h3 = h3_data["microcensus_trips"][["person_id", "trip_id", target_column]]
    h3_geo_level2 = h3_geo[leaf_name]
    required = ["centroid", *STATIC_CANDIDATE_FEATURES]
    missing = [name for name in required if name not in h3_geo_level2]
    if missing:
        raise RuntimeError("Missing destination-derived H3 columns: " + ", ".join(missing))
    all_h3 = h3_geo_level2["h3_index"].astype(str).tolist()
    h3_all_indexed = h3_geo_level2.set_index("h3_index").loc[all_h3]
    centroid_x = h3_all_indexed["centroid"].x.to_numpy(dtype=np.float64)
    centroid_y = h3_all_indexed["centroid"].y.to_numpy(dtype=np.float64)
    tree = KDTree(np.column_stack([centroid_x, centroid_y]), leaf_size=32)
    leaf_index = {cell: i for i, cell in enumerate(all_h3)}
    destinations = context.stage("synthesis.population.destinations")
    offer_columns = [name for name in destinations if name.startswith("offers_")]
    mapped = h3_data["destinations"][["destination_id", leaf_name]].merge(
        destinations[["destination_id", *offer_columns]], on="destination_id", how="left", validate="one_to_one")
    offers = destination_offer_masks(mapped)
    pools = {purpose: dict.fromkeys(mapped.loc[mask, leaf_name].dropna().astype(str).unique(), True)
             for purpose, mask in offers.items()}
    support = destination_support_masks(all_h3, pools, SECONDARY_ACTIVITIES)

    # Query origin neighbourhoods without conditioning alternatives on the observed destination.

    df = mz_trips.merge(trips_h3, on=["person_id", "trip_id"], how="left", validate="one_to_one")
    df = df.merge(mz_persons, on="person_id", how="left", validate="many_to_one")
    short_max = float(context.config("secondary_nn_short_trip_threshold_m"))
    short_min = float(context.config("short_range_trip_min_m"))
    if not np.isfinite([short_min, short_max]).all() or not 0 <= short_min < short_max:
        raise ValueError("Short-range training distance bounds must be finite and ordered")
    td = df["target_distance"].to_numpy(dtype=float)
    eligible = (df.purpose.isin(SECONDARY_ACTIVITIES) & np.isfinite(td)
                & (td >= short_min) & (td <= short_max))
    df = df.loc[eligible].reset_index(drop=True)
    radius_step = float(context.config("secondary_nn_short_query_radius_m"))
    radius = short_max + radius_step * 1.5
    minimum = int(context.config("secondary_nn_short_min_candidates"))
    valid_rows, choices, audit = origin_training_choice_sets(
        df, all_h3, tree, support, target_column=target_column,
        radius_m=radius, min_candidates=minimum, expand_step_m=radius_step)
    audit_path = os.path.join(context.path(), "short_range_choice_sets.parquet")
    audit.to_parquet(audit_path, index=False)
    os.chmod(audit_path, 0o600)
    counts = audit.reason.value_counts().to_dict()
    with open(os.path.join(context.path(), "short_range_choice_sets.json"), "w") as stream:
        json.dump(dict(scope="origin_neighbourhood", initial_radius_m=radius,
                       expansion_step_m=radius_step, minimum_candidates=minimum, rows=counts), stream, indent=2)
    logger.info("Short-range origin choice sets: %s", counts)
    if not valid_rows:
        raise RuntimeError("No short-range samples have an observed eligible leaf in their origin neighbourhood")
    df = df.iloc[valid_rows].reset_index(drop=True)

    # Pack variable neighbourhoods with masked padding for model training.

    n_samples, max_children = len(df), max(map(len, choices))
    cand_x = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_y = np.zeros_like(cand_x)
    cand_statent = np.zeros_like(cand_x)
    cand_employees = np.zeros_like(cand_x)
    cand_urban_core = np.zeros_like(cand_x)
    cand_urban = np.zeros_like(cand_x)
    cand_education = np.zeros_like(cand_x)
    cand_shop = np.zeros_like(cand_x)
    cand_leisure = np.zeros_like(cand_x)
    cand_sport = np.zeros_like(cand_x)
    cand_gastronomy = np.zeros_like(cand_x)
    cand_accommodation = np.zeros_like(cand_x)
    cand_cultural = np.zeros_like(cand_x)
    cand_ovgk_share_a = np.zeros_like(cand_x)
    cand_ovgk_share_b = np.zeros_like(cand_x)
    cand_ovgk_share_c = np.zeros_like(cand_x)
    cand_ovgk_share_d = np.zeros_like(cand_x)
    cand_ovgk_share_none = np.zeros_like(cand_x)
    cand_outside_fraction = np.zeros_like(cand_x)
    valid_mask = np.zeros(cand_x.shape, dtype=bool)
    y = np.zeros(n_samples, dtype=np.int64)
    weights = df["person_weight"].to_numpy(dtype=np.float32)
    attributes = h3_all_indexed[STATIC_CANDIDATE_FEATURES]
    attribute_arrays = {name: attributes[name].to_numpy(dtype=float) for name in STATIC_CANDIDATE_FEATURES}
    for i, indices in enumerate(choices):
        n = len(indices)
        cand_x[i, :n], cand_y[i, :n] = centroid_x[indices], centroid_y[indices]
        valid_mask[i, :n] = True
        chosen = leaf_index[str(df.iloc[i][target_column])]
        y[i] = int(np.flatnonzero(indices == chosen)[0])
        cand_statent[i, :n] = attribute_arrays["num_statent"][indices]
        cand_employees[i, :n] = attribute_arrays["employees"][indices]
        cand_urban_core[i, :n] = attribute_arrays["urban_core"][indices]
        cand_urban[i, :n] = attribute_arrays["urban"][indices]
        cand_education[i, :n] = attribute_arrays["education"][indices]
        cand_shop[i, :n] = attribute_arrays["shop"][indices]
        cand_leisure[i, :n] = attribute_arrays["leisure"][indices]
        cand_sport[i, :n] = attribute_arrays["sport"][indices]
        cand_gastronomy[i, :n] = attribute_arrays["gastronomy"][indices]
        cand_accommodation[i, :n] = attribute_arrays["accommodation"][indices]
        cand_cultural[i, :n] = attribute_arrays["cultural"][indices]
        cand_ovgk_share_a[i, :n] = attribute_arrays["ovgk_share_a"][indices]
        cand_ovgk_share_b[i, :n] = attribute_arrays["ovgk_share_b"][indices]
        cand_ovgk_share_c[i, :n] = attribute_arrays["ovgk_share_c"][indices]
        cand_ovgk_share_d[i, :n] = attribute_arrays["ovgk_share_d"][indices]
        cand_ovgk_share_none[i, :n] = attribute_arrays["ovgk_share_none"][indices]
        cand_outside_fraction[i, :n] = attribute_arrays["outside_fraction"][indices]

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
        mobility_matrix=training_mobility(context, df),
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
        use_target_distance=bool(context.config("secondary_nn_use_target_distance")),
        chain_state_matrix=np.stack(df["chain_state"]),
        random_state=int(context.config("random_seed")),
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
    candidate_tensor = enrich_candidate_tensor(candidate_tensor, cand_x, cand_y, valid_mask, h3_geo_level2)
    candidate_dist_home_m = candidate_tensor[:, :, 0].astype(np.float32)
    candidate_dist_last_m = candidate_tensor[:, :, 2].astype(np.float32)
    candidate_tensor, candidate_static_scaler, candidate_dynamic_scaler = fit_candidate_tensor(candidate_tensor, valid_mask, random_state=int(context.config("random_seed")))

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
        dropout_rate=float(context.config("short_range_model_dropout_rate")),
    )

    # Select the destination distribution loss.

    loss_kind = context.config("secondary_nn_distance_loss_kind")
    if context.config("secondary_nn_use_target_distance"):
        loss_kind = context.config("secondary_nn_short_distance_loss_kind") or loss_kind
    validation = model_validation_options(context,
        folds=context.config("short_range_model_validation_folds"),
        strategy=context.config("short_range_model_validation_strategy"),
        min_epochs=context.config("short_range_model_min_epochs"),
        val_every=context.config("short_range_model_validation_every"))
    distribution_weight = float(context.config("secondary_nn_distribution_loss_weight"))
    short_weight = context.config("secondary_nn_short_distribution_loss_weight")
    if context.config("secondary_nn_use_target_distance") and short_weight is not None:
        distribution_weight = float(short_weight)
        if not np.isfinite(distribution_weight) or distribution_weight < 0:
            raise ValueError("Short-range distribution weight must be finite and nonnegative")
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
        weight_decay=float(context.config("short_range_model_weight_decay")),
        num_threads=int(context.config("threads")),
        path=context.path(),
        **validation,
        validation_groups=df["person_id"].to_numpy(),
        rotate_validation=False,
        early_stopping_patience=int(context.config("short_range_model_early_stopping_patience")),
        early_stopping_min_delta=float(context.config("short_range_model_early_stopping_min_delta")),
        restore_best_weights=True,
        distance_candidates=candidate_dist_last_m,
        distance_targets=target_distance.astype(np.float32),
        distance_candidates_home=candidate_dist_home_m,
        distance_targets_home=target_home_distance.astype(np.float32),
        distance_loss_weight=(distribution_weight if loss_kind == "crps_energy"
                              else float(context.config("secondary_nn_distance_loss_weight"))),
        distance_loss_kind=loss_kind,
        distribution_scale_m=float(context.config("secondary_nn_distribution_scale_m")),
        energy_projections=int(context.config("secondary_nn_energy_projections")),
        candidate_xy=np.stack([cand_x, cand_y], axis=-1) if loss_kind == "crps_energy" else None,
        observed_xy=df[["destination_x", "destination_y"]].to_numpy(dtype=np.float32, copy=True)
            if loss_kind == "crps_energy" else None,
        distance_loss_short_floor_m=float(context.config("secondary_nn_distance_loss_short_floor_m")),
        relative_distance_underestimation_factor=(
            float(context.config("secondary_nn_short_relative_underestimation_factor"))
            if context.config("secondary_nn_use_target_distance") and loss_kind == "relative" else 1.0),
        auxiliary_relative_distance_loss_weight=(
            float(context.config("secondary_nn_short_relative_distance_loss_weight"))
            if context.config("secondary_nn_use_target_distance") and loss_kind == "crps_energy"
            else 0.0),
    )

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

    if context.config("secondary_nn_use_target_distance") or int(context.config("secondary_nn_crossfit_fold")) < 0:
        plot_distance_analysis(
            context=context,
            wrapper=wrapper,
            person_trip_matrix=person_trip_matrix,
            candidate_tensor=candidate_tensor,
            valid_mask=valid_mask,
            df=df,
            choices=choices,
            centroid_x_by_l2=dict(zip(all_h3, centroid_x)),
            centroid_y_by_l2=dict(zip(all_h3, centroid_y)),
        )

    logger.info("Short-range model saved to %s", model_path)
    return (model_path,)


def plot_distance_analysis(context, wrapper, person_trip_matrix, candidate_tensor,
                           valid_mask, df, choices,
                           centroid_x_by_l2, centroid_y_by_l2):
    """Plot real and predicted short-range distances without count maps."""
    logger.info("Plotting short-range distance distributions...")
    pred_idx = wrapper.predict_from_inputs(
        person_trip_matrix,
        candidate_tensor[:, :, N_CANDIDATE_DYNAMIC:],
        candidate_tensor[:, :, :N_CANDIDATE_DYNAMIC],
        valid_mask,
        rng=np.random.default_rng(int(context.config("random_seed"))),
        return_probabilities=False,
    )

    predicted_level2 = [wrapper.all_h3[indices[int(pred_idx[i])]]
                        for i, indices in enumerate(choices)]

    real_level2 = df[f"destination_{H3_LEVEL_NAMES[-1]}"].astype(str).to_numpy()
    predicted_level2 = np.asarray(predicted_level2, dtype=str)
    origin_x = df["origin_x"].to_numpy(dtype=np.float64)
    origin_y = df["origin_y"].to_numpy(dtype=np.float64)
    home_x = df["home_x"].to_numpy(dtype=np.float64)
    home_y = df["home_y"].to_numpy(dtype=np.float64)

    real_dist_origin = []
    pred_dist_origin = []
    real_dist_home = []
    pred_dist_home = []
    for i, (real_h3, pred_h3) in enumerate(zip(real_level2, predicted_level2)):
        real_cx = centroid_x_by_l2.get(real_h3)
        real_cy = centroid_y_by_l2.get(real_h3)
        pred_cx = centroid_x_by_l2.get(pred_h3)
        pred_cy = centroid_y_by_l2.get(pred_h3)
        if real_cx is None or real_cy is None or pred_cx is None or pred_cy is None:
            continue

        real_dist_origin.append(np.hypot(real_cx - origin_x[i], real_cy - origin_y[i]))
        pred_dist_origin.append(np.hypot(pred_cx - origin_x[i], pred_cy - origin_y[i]))
        real_dist_home.append(np.hypot(real_cx - home_x[i], real_cy - home_y[i]))
        pred_dist_home.append(np.hypot(pred_cx - home_x[i], pred_cy - home_y[i]))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    _plot_distance_histogram(
        axes[0], real_dist_origin, pred_dist_origin,
        "Distance from Previous Location",
    )
    _plot_distance_histogram(
        axes[1], real_dist_home, pred_dist_home,
        "Distance from Home",
    )
    plt.tight_layout()
    output_path = os.path.join(context.path(), "short_range_distance_distributions.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Short-range distance plot saved to %s", output_path)


def _plot_distance_histogram(axis, real_distances, predicted_distances, title):
    real_distances = np.asarray(real_distances, dtype=np.float64)
    predicted_distances = np.asarray(predicted_distances, dtype=np.float64)
    real_distances = real_distances[np.isfinite(real_distances)]
    predicted_distances = predicted_distances[np.isfinite(predicted_distances)]

    if len(real_distances) == 0 or len(predicted_distances) == 0:
        axis.text(0.5, 0.5, "No valid distances", ha="center", va="center", transform=axis.transAxes)
    else:
        maximum = max(float(real_distances.max()), float(predicted_distances.max()), 1.0)
        bins = np.linspace(0.0, maximum, 51)
        axis.hist(real_distances, bins=bins, color="black", linewidth=2,
                  label="Real", density=True, histtype="step")
        axis.hist(predicted_distances, bins=bins, color="red", linewidth=1,
                  label="Predicted", density=True, histtype="step", linestyle="dashed")
        axis.legend()

    axis.set_title(title)
    axis.set_xlabel("Distance (m)")
    axis.set_ylabel("Density")
