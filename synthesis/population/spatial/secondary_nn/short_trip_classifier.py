"""Learn which purpose-masked expert has lower destination loss on unseen people.

The historical stage name is retained for pipeline compatibility. Labels now
compare sampled destination predictions, never a 1.5 km trip-distance threshold.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

from .feature_encoding import fit_person_trip_matrix, transform_person_static_vector, transform_person_dynamic_vector
from .hierarchical_utils import (
    SECONDARY_ACTIVITIES, ORIGIN_PURPOSE_REMAP,
    DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION,
)
from .tools import MOBILITY_FEATURES, OriginContext, training_mobility

logger = logging.getLogger("synpp: expert_selector")
MODEL_NAME = "expert_selector.pkl"


def configure(context):
    context.config("secondary_nn_use_target_distance", True)
    if context.config("secondary_nn_use_target_distance"):
        return
    for stage in ["data.microcensus.trips", "data.microcensus.persons", "data.constants",
                  "data.spatial.ovgk", "synthesis.population.destinations",
                  "synthesis.population.spatial.secondary.detour_factors.factors"]:
        context.stage(stage)
    for stage in ["mz_chains", "h3", "training_split", "regional_model", "subregional_model", "local_model", "short_range_model"]:
        context.stage("synthesis.population.spatial.secondary_nn." + stage)
    context.config("random_seed")
    context.config("threads", volatile=True)
    context.config("secondary_nn_short_trip_threshold_m", 1500.0)
    context.config("secondary_nn_short_query_radius_m", 200.0)
    context.config("secondary_nn_short_min_candidates", 6)
    context.config("secondary_nn_selector_draws", 10)
    context.config("secondary_nn_selector_context_radii_m", [1500.0, 5000.0])
    context.config("secondary_nn_selector_validation_fraction", 0.2)
    context.config("secondary_nn_selector_spatial_loss_weight", 0.5)
    context.config("secondary_nn_selector_loss_scale_m", 1000.0)
    context.config("secondary_nn_selector_max_iter", 150)
    context.config("secondary_nn_selector_min_samples_leaf", 40)
    context.config("secondary_nn_selector_l2", 10.0)


def execute(context):

    # Train only when the expert cannot be selected from target distance.

    if context.config("secondary_nn_use_target_distance"):
        return None

    # Load experts with the same pool-backed masks used during assignment.

    from .model_wrappers import HierarchicalLocationChoiceModel
    torch.set_num_threads(max(1, int(context.config("threads"))))
    frame = prepare_training_data(context)
    wrapper = HierarchicalLocationChoiceModel.build(context, optimize=False, load_selector=False)
    seed = int(context.config("random_seed"))
    draws = int(context.config("secondary_nn_selector_draws"))
    weight = float(context.config("secondary_nn_selector_spatial_loss_weight"))
    scale = float(context.config("secondary_nn_selector_loss_scale_m"))

    # Compare predictions without exposing the observed destination to the models.

    logger.info("Comparing %s draws per expert on %s trips from %s reserved persons", draws, len(frame), frame.person_id.nunique())
    with context.progress(label="Comparing secondary destination experts", total=len(frame)) as progress:
        comparisons = compare_experts(frame, wrapper, draws=draws, seed=seed,
                                      spatial_weight=weight, scale_m=scale, progress=progress)

    # Fit the selector with separate person groups for fitting and validation.

    comparisons.to_parquet(Path(context.path()) / "expert_comparisons.parquet", index=False)
    cells = context.stage("synthesis.population.spatial.secondary_nn.h3")[1]["level_2"]
    classifier, audit = train_selector(
        frame, comparisons, OriginContext(cells, radii=context.config("secondary_nn_selector_context_radii_m")), seed=seed,
        validation_fraction=float(context.config("secondary_nn_selector_validation_fraction")),
        max_iter=int(context.config("secondary_nn_selector_max_iter")),
        min_samples_leaf=int(context.config("secondary_nn_selector_min_samples_leaf")),
        l2=float(context.config("secondary_nn_selector_l2")),
    )
    classifier.metrics.update(draws_per_expert=draws, spatial_loss_weight=weight, loss_scale_m=scale,
                              experts_exclude_all_selector_persons=True)

    # Save the selector and validation audit.

    classifier.origin_context.cache.clear()
    path = Path(context.path()) / MODEL_NAME
    joblib.dump(classifier, path)
    audit.to_parquet(Path(context.path()) / "selector_partitions.parquet", index=False)
    with open(Path(context.path()) / "validation.json", "w") as stream:
        json.dump(classifier.metrics, stream, indent=2)
    logger.info("Expert selector held-out validation: %s", classifier.metrics)
    return (str(path),)


def prepare_training_data(context):
    trips = context.stage("data.microcensus.trips")[0][[
        "person_id", "trip_id", "purpose", "origin_purpose", "origin_x", "origin_y",
        "destination_x", "destination_y",
    ]]
    chains = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    valid_trips = context.stage("synthesis.population.spatial.secondary_nn.h3")[0]["microcensus_trips"][["person_id", "trip_id"]]
    persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "employed", "car_availability", "income_class",
        "weekend", "person_weight", "home_x", "home_y", "work_x", "work_y",
    ]].copy()
    persons = persons.loc[~persons.weekend].drop(columns="weekend")
    split = context.stage("synthesis.population.spatial.secondary_nn.training_split")
    persons = persons.loc[persons.person_id.isin(split.loc[split.selector, "person_id"])]
    persons["car_availability"] = persons.car_availability != context.stage("data.constants").CAR_AVAILABILITY_NEVER
    frame = trips.merge(chains, on=["person_id", "trip_id"], validate="one_to_one")
    frame = frame.merge(valid_trips, on=["person_id", "trip_id"], validate="one_to_one")
    frame = frame.merge(persons, on="person_id", validate="many_to_one")
    coordinates = frame[["origin_x", "origin_y", "destination_x", "destination_y", "home_x", "home_y"]]
    distance = frame.target_distance.to_numpy(dtype=float)
    valid = (frame.purpose.isin(SECONDARY_ACTIVITIES) & np.isfinite(coordinates).all(axis=1)
             & np.isfinite(distance) & (distance >= DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION))
    frame = frame.loc[valid].sort_values(["person_id", "trip_id"]).reset_index(drop=True)
    frame["has_work"] = np.isfinite(frame.work_x) & np.isfinite(frame.work_y)
    frame.loc[~frame.has_work, ["work_x", "work_y"]] = 0.0
    frame[MOBILITY_FEATURES] = training_mobility(context, frame)
    return frame


def prediction_loss(predictions, origin, target, *, spatial_weight=.5, scale_m=1000.0):
    """Score both directional destination error and the error of mean trip length.

    Mean spatial error prevents predictions on opposite sides of the target from
    cancelling. Log scaling limits the influence of very long excursion outliers.
    """
    predictions = np.asarray(predictions, dtype=float)
    origin, target = np.asarray(origin, dtype=float), np.asarray(target, dtype=float)
    if (predictions.ndim != 2 or predictions.shape[1] != 2 or len(predictions) == 0
            or not np.isfinite(predictions).all() or not np.isfinite(origin).all()
            or not np.isfinite(target).all() or scale_m <= 0 or not 0 <= spatial_weight <= 1):
        raise ValueError("Invalid expert prediction loss inputs")
    spatial_error = np.linalg.norm(predictions - target, axis=1).mean()
    predicted_distance = np.linalg.norm(predictions - origin, axis=1).mean()
    observed_distance = np.linalg.norm(target - origin)
    distance_error = abs(predicted_distance - observed_distance)
    loss = spatial_weight * np.log1p(spatial_error / scale_m) + (1 - spatial_weight) * np.log1p(distance_error / scale_m)
    return dict(loss=float(loss), mean_destination_error_m=float(spatial_error),
                mean_predicted_distance_m=float(predicted_distance),
                mean_distance_error_m=float(distance_error), observed_distance_m=float(observed_distance))


def prediction_inputs(row):
    return dict(
        person_id=row.person_id, home_x=row.home_x, home_y=row.home_y,
        work_x=row.work_x, work_y=row.work_y, origin_x=row.origin_x, origin_y=row.origin_y,
        age=row.age, daily_longest_distance_from_home=row.daily_longest_distance_from_home,
        daily_crowfly_total=row.daily_crowfly_total,
        daily_longest_distance_from_work=row.daily_longest_distance_from_work,
        crowfly_consumed_before_trip=row.crowfly_consumed_before_trip,
        trip_position_class=row.trip_position_class, departure_time_normalized=row.departure_time_normalized,
        activity_duration_h=row.activity_duration_h, target_distance=0.0,
        activity_chain_vector=row.activity_chain, sex=row.sex, employed=row.employed,
        car_availability=row.car_availability, income_class=row.income_class,
        purpose=row.purpose, origin_purpose=row.origin_purpose, has_work=row.has_work,
        has_education=False, mobility_vector=np.asarray([getattr(row, c) for c in MOBILITY_FEATURES]),
    )


def compare_experts(frame, wrapper, *, draws, seed, spatial_weight=.5, scale_m=1000.0, progress=None):
    if draws < 2:
        raise ValueError("Expert comparison requires at least two destination draws")
    records = []
    for row in frame.itertuples(index=False):
        inputs = prediction_inputs(row)
        record = dict(person_id=row.person_id, trip_id=row.trip_id)
        for expert_index, expert in enumerate(["short", "hierarchy"]):
            predictions = []
            for draw in range(draws):
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(row.person_id), int(row.trip_id), expert_index, draw]))
                _, point = wrapper.predict(**inputs, rng=rng, force_expert=expert)
                predictions.append([point.x, point.y])
            scores = prediction_loss(predictions, [row.origin_x, row.origin_y],
                                     [row.destination_x, row.destination_y],
                                     spatial_weight=spatial_weight, scale_m=scale_m)
            record.update({expert + "_" + key: value for key, value in scores.items()})
        records.append(record)
        if progress is not None:
            progress.update()
    return pd.DataFrame(records)


def _feature_matrix(frame, context, static_scaler=None, dynamic_scaler=None, seed=None):
    def numeric(column):
        values = frame[column].to_numpy(dtype=float)
        return np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    chains = np.stack(frame.activity_chain.map(lambda v: np.asarray(v, dtype=float)[:len(SECONDARY_ACTIVITIES)]))
    chains = np.maximum(np.nan_to_num(chains, nan=0, posinf=0, neginf=0), 0)
    matrix, _, _, static, dynamic, names = fit_person_trip_matrix(
        age=numeric("age"), sex=numeric("sex"), employed=numeric("employed"),
        car_availability=numeric("car_availability"), income_class=numeric("income_class"),
        daily_longest=numeric("daily_longest_distance_from_home"), daily_total=numeric("daily_crowfly_total"),
        daily_longest_work=numeric("daily_longest_distance_from_work"), activity_chain_matrix=chains,
        consumed_before=numeric("crowfly_consumed_before_trip"), trip_position=numeric("trip_position_class"),
        departure_time=numeric("departure_time_normalized"), activity_duration_h=numeric("activity_duration_h"),
        target_distance=np.zeros(len(frame)), purpose_series=frame.purpose, origin_purpose_series=frame.origin_purpose,
        purpose_categories=SECONDARY_ACTIVITIES, person_static_scaler=static_scaler,
        person_dynamic_scaler=dynamic_scaler, use_target_distance=False, random_state=seed,
        mobility_matrix=frame[MOBILITY_FEATURES].to_numpy(),
    )
    origins = np.stack([context.vector(r.origin_x, r.origin_y, r.home_x, r.home_y, r.work_x, r.work_y, r.has_work)
                        for r in frame.itertuples(index=False)])
    return np.concatenate([matrix, origins], axis=1), static, dynamic, names + context.names


@dataclass
class ExpertSelector:
    model: object
    static_scaler: object
    dynamic_scaler: object
    origin_context: OriginContext
    feature_names: list
    metrics: dict

    def predict_loss_difference(self, *, age, income_class, daily_longest, daily_total,
                                daily_longest_work, activity_chain_vector, sex, employed,
                                car_availability, consumed_before, trip_position, departure_time,
                                activity_duration_h, purpose, origin_purpose, mobility_vector,
                                origin_x, origin_y, home_x, home_y, work_x, work_y, has_work):
        static = transform_person_static_vector(
            age, income_class, daily_longest, daily_total, daily_longest_work,
            activity_chain_vector, sex, employed, car_availability, self.static_scaler,
            mobility_vector=mobility_vector)
        identity = np.eye(len(SECONDARY_ACTIVITIES), dtype=np.float32)
        purpose_hot = identity[SECONDARY_ACTIVITIES.index(purpose)]
        origin = ORIGIN_PURPOSE_REMAP.get(origin_purpose, origin_purpose)
        origin_hot = identity[SECONDARY_ACTIVITIES.index(origin)] if origin in SECONDARY_ACTIVITIES else np.zeros(len(identity))
        dynamic = transform_person_dynamic_vector(consumed_before, trip_position, departure_time,
                                                 activity_duration_h, 0, purpose_hot, origin_hot, self.dynamic_scaler)
        context = self.origin_context.vector(origin_x, origin_y, home_x, home_y, work_x, work_y, has_work)
        return float(self.model.predict(np.concatenate([static, dynamic, context[None, :]], axis=1))[0])

    def predict_short(self, **features):
        # Regress the cost difference so expensive mistakes matter more than close ties.
        return self.predict_loss_difference(**features) < 0.0


def train_selector(frame, comparisons, origin_context, *, seed, validation_fraction=.2,
                   max_iter=150, min_samples_leaf=40, l2=10.0):
    if not 0 < validation_fraction < .5 or max_iter < 1 or min_samples_leaf < 2 or l2 < 0:
        raise ValueError("Invalid expert-selector training controls")
    frame = frame.merge(comparisons, on=["person_id", "trip_id"], validate="one_to_one")
    if frame.person_id.nunique() < 10:
        raise ValueError("Expert selector requires at least ten independent persons")
    y = (frame.short_loss - frame.hierarchy_loss).to_numpy()
    weights = frame.person_weight.to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Expert losses and positive person weights must be finite")
    # Outer validation is never used for scaler fitting, early stopping, or tuning.
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=validation_fraction, random_state=seed).split(frame, groups=frame.person_id))
    inner_train, inner_validation = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed + 1).split(frame.iloc[train], groups=frame.iloc[train].person_id))
    fit, stopping = train[inner_train], train[inner_validation]
    x_fit, static, dynamic, names = _feature_matrix(frame.iloc[fit], origin_context, seed=seed)
    x_stopping, _, _, _ = _feature_matrix(frame.iloc[stopping], origin_context, static, dynamic)
    x_validation, _, _, _ = _feature_matrix(frame.iloc[validation], origin_context, static, dynamic)
    model = HistGradientBoostingRegressor(
        max_iter=max_iter, max_leaf_nodes=7, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2, learning_rate=.05, early_stopping=True,
        n_iter_no_change=12, random_state=seed,
    )
    model.fit(x_fit, y[fit], sample_weight=weights[fit] / weights[fit].mean(),
              X_val=x_stopping, y_val=y[stopping], sample_weight_val=weights[stopping] / weights[fit].mean())
    predicted = model.predict(x_validation)
    short, hierarchy = frame.short_loss.to_numpy()[validation], frame.hierarchy_loss.to_numpy()[validation]
    chosen = np.where(predicted < 0, short, hierarchy)
    oracle = np.minimum(short, hierarchy)
    average = lambda value: float(np.average(value, weights=weights[validation]))
    fitted = model.predict(x_fit)
    fit_short, fit_hierarchy = frame.short_loss.to_numpy()[fit], frame.hierarchy_loss.to_numpy()[fit]
    training_regret = np.where(fitted < 0, fit_short, fit_hierarchy) - np.minimum(fit_short, fit_hierarchy)
    metrics = dict(
        training_regret=float(np.average(training_regret, weights=weights[fit])),
        training_loss_difference_mae=float(np.average(abs(fitted - y[fit]), weights=weights[fit])),
        training_rows=len(fit), stopping_rows=len(stopping), validation_rows=len(validation),
        training_persons=int(frame.iloc[fit].person_id.nunique()),
        stopping_persons=int(frame.iloc[stopping].person_id.nunique()),
        validation_persons=int(frame.iloc[validation].person_id.nunique()),
        validation_selected_loss=average(chosen), validation_always_short_loss=average(short),
        validation_always_hierarchy_loss=average(hierarchy), validation_oracle_loss=average(oracle),
        validation_regret=average(chosen - oracle), validation_expert_accuracy=average((predicted < 0) == (y[validation] < 0)),
        validation_short_share=average(predicted < 0), iterations=int(model.n_iter_),
        validation_loss_difference_mae=average(abs(predicted - y[validation])),
    )
    audit = frame[["person_id", "trip_id"]].copy()
    audit["partition"] = ""
    audit.loc[fit, "partition"] = "train"
    audit.loc[stopping, "partition"] = "stopping"
    audit.loc[validation, "partition"] = "validation"
    audit["observed_loss_difference"] = y
    audit["predicted_loss_difference"] = np.nan
    audit.loc[validation, "predicted_loss_difference"] = predicted
    return ExpertSelector(model, static, dynamic, origin_context, names, metrics), audit
