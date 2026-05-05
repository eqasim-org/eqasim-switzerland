import os
import logging
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .NNModel import MNLModel, train_with_mask, MediumLevel1Wrapper
from .hierarchical_model_utils import (
    SECONDARY_ACTIVITIES,
    build_hierarchical_numerical_batch_numba,
    get_h3_stage_outputs,
    build_level1_children_by_level0,
    build_level1_candidate_attributes_by_level0,
    make_purpose_one_hot,
    sanitize_work_coordinates,
)

logger = logging.getLogger("synpp: medium_model")

MODEL_NAME = "medium_model.pt"


def configure(context):
    context.stage("synthesis.population.spatial.secondary.locations_v2.h3")
    context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("data.spatial.swiss_border")
    context.stage("synthesis.population.destinations")

    context.config("overwrite_medium_model_if_exists", True)
    context.config("medium_model_batch_size", 256)
    context.config("medium_model_epochs", 40)
    context.config("medium_model_torch_num_threads", 16)
    context.config("random_seed")


def execute(context):
    logger.info("Training medium model for level1-within-level0 secondary location choice...")

    overwrite_model = context.config("overwrite_medium_model_if_exists")
    model_path = os.path.join(context.working_directory, MODEL_NAME)
    if os.path.exists(model_path) and not overwrite_model:
        logger.info("Model %s already exists.", MODEL_NAME)
        return model_path

    logger.info("\t Loading data...")
    mz_persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "car_availability", "employed", "income_class", "home_x", "home_y", "work_x", "work_y", "weekend", "person_weight"
    ]]
    c = context.stage("data.constants")
    mz_persons["car_availability"] = (mz_persons["car_availability"] != c.CAR_AVAILABILITY_NEVER)

    mz_trips, _ = context.stage("data.microcensus.trips")
    mz_trips = mz_trips[["person_id", "trip_id", "origin_x", "origin_y", "purpose"]]
    mz_chain_trips = context.stage("synthesis.population.spatial.secondary.locations_v2.mz_chains")[[
        "person_id",
        "trip_id",
        "daily_longest_distance_from_home",
        "daily_crowfly_total",
        "crowfly_consumed_before_trip",
        "trip_position_class",
    ]]
    mz_trips = mz_trips.merge(mz_chain_trips, on=["person_id", "trip_id"], how="left")

    weekend_persons = mz_persons[mz_persons["weekend"]]["person_id"].unique()
    mz_trips = mz_trips[~mz_trips["person_id"].isin(weekend_persons)].reset_index(drop=True)
    mz_persons = mz_persons[~mz_persons["person_id"].isin(weekend_persons)].drop(columns=["weekend"]).reset_index(drop=True)

    h3_data, h3_geo, h3_tree = get_h3_stage_outputs(context)
    trips_h3 = h3_data["microcensus_trips"][["person_id", "trip_id", "destination_level_0", "destination_level_1"]]
    h3_geo_level1 = h3_geo["level_1"]

    if "outside_fraction" not in h3_geo_level1.columns:
        raise RuntimeError("Missing outside_fraction in H3 level_1 geometry. Run h3 stage with outside_fraction enabled.")

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
    missing_h3_cols = [c for c in required_h3_cols if c not in h3_geo_level1.columns]
    if missing_h3_cols:
        raise RuntimeError("Missing required destination-derived H3 columns in h3 stage: " + ", ".join(missing_h3_cols))

    h3_level1_indexed = h3_geo_level1.set_index("h3_index")
    centroids = h3_level1_indexed["centroid"]
    centroid_x_by_l1 = centroids.x.to_dict()
    centroid_y_by_l1 = centroids.y.to_dict()
    outside_fraction_by_l1 = h3_level1_indexed["outside_fraction"].to_dict()
    statent_count = h3_level1_indexed["num_statent"].to_dict()
    employees_count = h3_level1_indexed["employees"].to_dict()
    urban_core_count = h3_level1_indexed["urban_core"].to_dict()
    urban_count = h3_level1_indexed["urban"].to_dict()
    education_count = h3_level1_indexed["education"].to_dict()
    shop_count = h3_level1_indexed["shop"].to_dict()
    leisure_count = h3_level1_indexed["leisure"].to_dict()
    ovgk_share_a_by_l1 = h3_level1_indexed["ovgk_share_a"].to_dict()
    ovgk_share_b_by_l1 = h3_level1_indexed["ovgk_share_b"].to_dict()
    ovgk_share_c_by_l1 = h3_level1_indexed["ovgk_share_c"].to_dict()
    ovgk_share_d_by_l1 = h3_level1_indexed["ovgk_share_d"].to_dict()
    ovgk_share_none_by_l1 = h3_level1_indexed["ovgk_share_none"].to_dict()

    if h3_tree is None:
        raise RuntimeError("H3 hierarchy tree is missing from H3 stage output. Cannot train medium model.")

    children_by_level0 = build_level1_children_by_level0(h3_tree, centroid_x_by_l1, centroid_y_by_l1)
    if len(children_by_level0) == 0:
        raise RuntimeError("H3 hierarchy tree has no valid level1 children with centroids. Cannot train medium model.")

    level1_candidate_attributes_by_level0 = build_level1_candidate_attributes_by_level0(
        children_by_level0,
        centroid_x_by_l1,
        centroid_y_by_l1,
        statent_count,
        employees_count,
        urban_core_count,
        urban_count,
        education_count,
        shop_count,
        leisure_count,
        ovgk_share_a_by_l1,
        ovgk_share_b_by_l1,
        ovgk_share_c_by_l1,
        ovgk_share_d_by_l1,
        ovgk_share_none_by_l1,
        outside_fraction_by_l1,
    )

    logger.info("\t Preparing microcensus training set...")
    df = mz_trips.merge(trips_h3, on=["person_id", "trip_id"], how="left")
    df = df.merge(mz_persons, on="person_id", how="left")
    df = df[df["purpose"].isin(SECONDARY_ACTIVITIES)].dropna(subset=["destination_level_0", "destination_level_1"]).reset_index(drop=True)

    valid_rows = []
    for i, row in df.iterrows():
        children = children_by_level0.get(row["destination_level_0"], [])
        if len(children) < 2:
            continue
        if row["destination_level_1"] not in children:
            continue
        valid_rows.append(i)

    if len(valid_rows) == 0:
        raise RuntimeError("No valid samples for medium model after filtering by level0-level1 hierarchy.")
    df = df.iloc[valid_rows].reset_index(drop=True)

    max_children = max(len(children_by_level0[l0]) for l0 in df["destination_level_0"].unique())
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
    cand_ovgk_share_a = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_b = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_c = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_d = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_ovgk_share_none = np.zeros((n_samples, max_children), dtype=np.float64)
    cand_outside_fraction = np.zeros((n_samples, max_children), dtype=np.float64)
    valid_mask = np.zeros((n_samples, max_children), dtype=np.bool_)
    y = np.zeros(n_samples, dtype=np.int64)
    weights = df["person_weight"].to_numpy(dtype=np.float32)

    with context.progress(total=n_samples, label="Medium model: building level1 choice sets") as progress:
        for i, row in df.iterrows():
            l0 = row["destination_level_0"]
            chosen_l1 = row["destination_level_1"]
            l0_attributes = level1_candidate_attributes_by_level0[l0]
            children = l0_attributes["children"]

            valid_mask[i, :len(children)] = True
            for j in range(len(children)):
                cand_x[i, j] = l0_attributes["x"][j]
                cand_y[i, j] = l0_attributes["y"][j]
                cand_statent[i, j] = l0_attributes["num_statent"][j]
                cand_employees[i, j] = l0_attributes["employees"][j]
                cand_urban_core[i, j] = l0_attributes["urban_core"][j]
                cand_urban[i, j] = l0_attributes["urban"][j]
                cand_education[i, j] = l0_attributes["education"][j]
                cand_shop[i, j] = l0_attributes["shop"][j]
                cand_leisure[i, j] = l0_attributes["leisure"][j]
                cand_ovgk_share_a[i, j] = l0_attributes["ovgk_share_a"][j]
                cand_ovgk_share_b[i, j] = l0_attributes["ovgk_share_b"][j]
                cand_ovgk_share_c[i, j] = l0_attributes["ovgk_share_c"][j]
                cand_ovgk_share_d[i, j] = l0_attributes["ovgk_share_d"][j]
                cand_ovgk_share_none[i, j] = l0_attributes["ovgk_share_none"][j]
                cand_outside_fraction[i, j] = l0_attributes["outside_fraction"][j]

            y[i] = children.index(chosen_l1)
            progress.update()

    home_x = df["home_x"].to_numpy(dtype=np.float64)
    home_y = df["home_y"].to_numpy(dtype=np.float64)
    work_x = df["work_x"].to_numpy(dtype=np.float64)
    work_y = df["work_y"].to_numpy(dtype=np.float64)
    has_work, work_x, work_y = sanitize_work_coordinates(work_x, work_y)
    origin_x = df["origin_x"].to_numpy(dtype=np.float64)
    origin_y = df["origin_y"].to_numpy(dtype=np.float64)
    age = df["age"].to_numpy(dtype=np.float64)
    daily_longest_distance_from_home = df["daily_longest_distance_from_home"].to_numpy(dtype=np.float64)
    daily_longest_distance_from_home = np.where(np.isfinite(daily_longest_distance_from_home) & (daily_longest_distance_from_home >= 0.0), daily_longest_distance_from_home, 0.0)
    daily_crowfly_total = df["daily_crowfly_total"].to_numpy(dtype=np.float64)
    daily_crowfly_total = np.where(np.isfinite(daily_crowfly_total) & (daily_crowfly_total >= 0.0), daily_crowfly_total, 0.0)
    crowfly_consumed_before_trip = df["crowfly_consumed_before_trip"].to_numpy(dtype=np.float64)
    crowfly_consumed_before_trip = np.where(np.isfinite(crowfly_consumed_before_trip) & (crowfly_consumed_before_trip >= 0.0), crowfly_consumed_before_trip, 0.0)
    trip_position_class = df["trip_position_class"].to_numpy(dtype=np.float64)
    trip_position_class = np.where(np.isfinite(trip_position_class), trip_position_class, 2.0)
    sex = df["sex"].to_numpy(dtype=np.float32)
    employed = df["employed"].to_numpy(dtype=np.float32)
    car_availability = df["car_availability"].to_numpy(dtype=np.float32)
    income_class = df["income_class"].to_numpy(dtype=np.float32)

    purpose_categories = [str(p) for p in SECONDARY_ACTIVITIES]
    purpose_one_hot = make_purpose_one_hot(df["purpose"], purpose_categories)

    logger.info("\t Computing medium-model numerical features with Numba...")
    full_numerical = build_hierarchical_numerical_batch_numba(
        home_x,
        home_y,
        work_x,
        work_y,
        has_work,
        origin_x,
        origin_y,
        age,
        daily_longest_distance_from_home,
        daily_crowfly_total,
        crowfly_consumed_before_trip,
        trip_position_class,
        income_class,
        cand_x,
        cand_y,
        cand_statent,
        cand_employees,
        cand_urban_core,
        cand_urban,
        cand_education,
        cand_shop,
        cand_leisure,
        cand_ovgk_share_a,
        cand_ovgk_share_b,
        cand_ovgk_share_c,
        cand_ovgk_share_d,
        cand_ovgk_share_none,
        cand_outside_fraction,
        valid_mask,
    )

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
        ("dist", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, validate=False)),
            ("scale", QuantileTransformer(output_distribution="uniform")),
        ]), distance_cols),
        ("count", QuantileTransformer(output_distribution="normal"), count_cols),
        ("positive", QuantileTransformer(output_distribution="uniform"), positive_cols),
        ("passthrough", "passthrough", passthrough_cols),
    ])

    scaler.fit(full_numerical.reshape(-1, len(numerical))[valid_mask.reshape(-1)])

    scaled_flat = scaler.transform(full_numerical.reshape(-1, len(numerical)))
    scaled_numerical = scaled_flat.reshape(n_samples, max_children, len(numerical)).astype(np.float32)

    purpose_features = [f"purpose_{p}" for p in purpose_categories]
    features = numerical + ["sex", "employed", "car_availability"] + purpose_features
    X = np.zeros((n_samples, max_children, len(features)), dtype=np.float32)
    X[:, :, :len(numerical)] = scaled_numerical
    X[:, :, len(numerical)] = sex[:, None]
    X[:, :, len(numerical) + 1] = employed[:, None]
    X[:, :, len(numerical) + 2] = car_availability[:, None]
    X[:, :, len(numerical) + 3:] = np.broadcast_to(purpose_one_hot[:, None, :], (n_samples, max_children, purpose_one_hot.shape[1]))
    X[~valid_mask] = 0.0

    logger.info("\t Training medium model...")
    seed = context.config("random_seed")
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = MNLModel(input_dim=len(features), num_h3=max_children)
    train_with_mask(
        model=model,
        X=X,
        y=y,
        valid_mask=valid_mask,
        epochs=int(context.config("medium_model_epochs")),
        batch_size=int(context.config("medium_model_batch_size")),
        lr=1e-2,
        weight_decay=1e-4,
        num_threads=int(context.config("medium_model_torch_num_threads")),
        logger_instance=logger,
        weights=weights,
    )

    wrapper = MediumLevel1Wrapper(
        model=model,
        scaler=scaler,
        numerical_cols=numerical,
        features=features,
        children_by_level0=children_by_level0,
        level1_candidate_attributes_by_level0=level1_candidate_attributes_by_level0,
        purpose_categories=purpose_categories,
    )
    wrapper.save(model_path)

    h3_geo_counts = plot_analysis(
        context=context,
        wrapper=wrapper,
        X=X,
        valid_mask=valid_mask,
        df=df,
        children_by_level0=children_by_level0,
        h3_geo_level1=h3_geo_level1,
        centroid_x_by_l1=centroid_x_by_l1,
        centroid_y_by_l1=centroid_y_by_l1,
    )

    logger.info("Medium model saved to %s", model_path)
    return wrapper, X, features, model_path, h3_geo_counts


def plot_analysis(context, wrapper, X, valid_mask, df, children_by_level0, h3_geo_level1, centroid_x_by_l1, centroid_y_by_l1):
    logger.info("Predicting on training data and plotting level1 counts...")
    pred_idx = wrapper.predict_from_X(X, valid_mask, max_utility=False)

    predicted_level1 = []
    for i, l0 in enumerate(df["destination_level_0"].to_numpy()):
        children = children_by_level0[l0]
        predicted_level1.append(children[int(pred_idx[i])])

    real_level1 = df["destination_level_1"].astype(str)
    real_counts = real_level1.value_counts().rename("real_count")
    pred_counts = pd.Series(predicted_level1).value_counts().rename("pred_count")
    counts_df = pd.DataFrame({"real_count": real_counts, "pred_count": pred_counts}).fillna(0)

    h3_geo_counts = h3_geo_level1.set_index("h3_index").join(counts_df, how="left").fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    h3_geo_counts.plot(column="real_count", ax=axes[0], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[0].set_title("Real Level1 H3 Counts")
    h3_geo_counts.plot(column="pred_count", ax=axes[1], legend=True, cmap="viridis", legend_kwds={"shrink": 0.5})
    axes[1].set_title("Predicted Level1 H3 Counts")
    plt.tight_layout()
    plot_path = os.path.join(context.path(), "medium_level1_counts_comparison.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    logger.info("Plot saved to %s", plot_path)
    plt.close(fig)

    logger.info("Plotting distance distributions for level1...")
    real_dist_home = []
    pred_dist_home = []
    real_dist_work = []
    pred_dist_work = []

    home_x = df["home_x"].to_numpy(dtype=np.float64)
    home_y = df["home_y"].to_numpy(dtype=np.float64)
    work_x = df["work_x"].to_numpy(dtype=np.float64)
    work_y = df["work_y"].to_numpy(dtype=np.float64)
    has_work = np.isfinite(work_x) & np.isfinite(work_y)

    real_level1_arr = df["destination_level_1"].astype(str).to_numpy()
    pred_level1_arr = np.asarray(predicted_level1, dtype=str)

    for i in range(len(df)):
        real_h3 = real_level1_arr[i]
        pred_h3 = pred_level1_arr[i]

        real_cx = centroid_x_by_l1.get(real_h3)
        real_cy = centroid_y_by_l1.get(real_h3)
        pred_cx = centroid_x_by_l1.get(pred_h3)
        pred_cy = centroid_y_by_l1.get(pred_h3)

        if real_cx is None or real_cy is None or pred_cx is None or pred_cy is None:
            continue

        real_dist_home.append(np.sqrt((real_cx - home_x[i]) ** 2 + (real_cy - home_y[i]) ** 2))
        pred_dist_home.append(np.sqrt((pred_cx - home_x[i]) ** 2 + (pred_cy - home_y[i]) ** 2))

        if has_work[i]:
            real_dist_work.append(np.sqrt((real_cx - work_x[i]) ** 2 + (real_cy - work_y[i]) ** 2))
            pred_dist_work.append(np.sqrt((pred_cx - work_x[i]) ** 2 + (pred_cy - work_y[i]) ** 2))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    THRESHOLD = 80000  # 80 km
    real_dist_home = np.array(real_dist_home)
    pred_dist_home = np.array(pred_dist_home)
    real_dist_home = real_dist_home[real_dist_home <= THRESHOLD]
    pred_dist_home = pred_dist_home[pred_dist_home <= THRESHOLD]

    axes[0].hist(real_dist_home, bins=50, alpha=0.4, color="black", linewidth=2, label="Real", density=True, histtype="step")
    axes[0].hist(pred_dist_home, bins=50, alpha=0.4, color="red", linewidth=1, label="Predicted", density=True, histtype="step", linestyle="dashed")
    axes[0].set_title("Distance from Home")
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    if len(real_dist_work) > 0:
        real_dist_work = np.array(real_dist_work)
        pred_dist_work = np.array(pred_dist_work)
        real_dist_work = real_dist_work[real_dist_work <= THRESHOLD]
        pred_dist_work = pred_dist_work[pred_dist_work <= THRESHOLD]

        axes[1].hist(real_dist_work, bins=50, alpha=0.4, color="black", linewidth=2, label="Real", density=True, histtype="step")
        axes[1].hist(pred_dist_work, bins=50, alpha=0.4, color="red", linewidth=1, label="Predicted", density=True, histtype="step", linestyle="dashed")
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid work locations", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_title("Distance from Work (has work)")
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Density")

    plt.tight_layout()
    dist_plot_path = os.path.join(context.path(), "medium_distance_distributions.png")
    plt.savefig(dist_plot_path, dpi=200, bbox_inches="tight")
    logger.info("Distance distribution plot saved to %s", dist_plot_path)
    plt.close(fig)

    return h3_geo_counts
