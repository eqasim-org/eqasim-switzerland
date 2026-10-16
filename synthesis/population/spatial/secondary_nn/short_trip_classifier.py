"""Train the distance-free selector between short-range and hierarchical choice."""

from dataclasses import dataclass
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from .feature_encoding import (
    ACTIVITY_CHAIN_N, fit_person_trip_matrix, transform_person_dynamic_vector,
    transform_person_static_vector,
)
from .hierarchical_utils import (
    ORIGIN_PURPOSE_REMAP, SECONDARY_ACTIVITIES,
    DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION,
)

logger = logging.getLogger("synpp: short_trip_classifier")
MODEL_NAME = "short_trip_classifier.pkl"


def configure(context):
    context.config("secondary_nn_use_target_distance", True)
    if context.config("secondary_nn_use_target_distance"):
        return
    context.stage("data.microcensus.trips")
    context.stage("data.microcensus.persons")
    context.stage("data.constants")
    context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    context.stage("synthesis.population.spatial.secondary_nn.h3")
    context.config("random_seed")
    context.config("secondary_nn_short_trip_threshold_m", 1500.0)
    context.config("secondary_nn_short_classifier_validation_fraction", 0.2)
    # Small, L2-regularized model; no identity features or validation-driven tuning.
    context.config("secondary_nn_short_classifier_C", 0.1)


def prepare_training_data(context):
    trips = context.stage("data.microcensus.trips")[0][
        ["person_id", "trip_id", "purpose", "origin_purpose"]
    ]
    chains = context.stage("synthesis.population.spatial.secondary_nn.mz_chains")
    valid_trips = context.stage("synthesis.population.spatial.secondary_nn.h3")[0][
        "microcensus_trips"
    ][["person_id", "trip_id"]]
    persons = context.stage("data.microcensus.persons")[[
        "person_id", "age", "sex", "employed", "car_availability", "income_class",
        "weekend", "person_weight",
    ]].copy()
    persons = persons.loc[~persons["weekend"]].drop(columns="weekend")
    constants = context.stage("data.constants")
    persons["car_availability"] = persons["car_availability"] != constants.CAR_AVAILABILITY_NEVER
    frame = trips.merge(chains, on=["person_id", "trip_id"], validate="one_to_one")
    frame = frame.merge(valid_trips, on=["person_id", "trip_id"], validate="one_to_one")
    frame = frame.merge(persons, on="person_id", validate="many_to_one")
    distance = frame["target_distance"].to_numpy(dtype=float)
    # Assignment handles stationary trips before calling the selector.
    valid = (frame["purpose"].isin(SECONDARY_ACTIVITIES)
             & np.isfinite(distance)
             & (distance >= DISTANCE_THRESHOLD_FOR_STAYING_AT_PREVIOUS_LOCATION))
    return frame.loc[valid].reset_index(drop=True)


def _feature_matrix(frame, static_scaler=None, dynamic_scaler=None, random_state=None):
    """Use the choice-model feature vocabulary; distance is never an input."""
    def numeric(column, default=0.0):
        values = frame[column].to_numpy(dtype=float)
        return np.where(np.isfinite(values), np.maximum(values, 0.0), default)

    chains = np.stack(frame["activity_chain"].map(
        lambda value: np.asarray(value, dtype=float)[:ACTIVITY_CHAIN_N]
    ))
    chains = np.where(np.isfinite(chains), np.maximum(chains, 0.0), 0.0)
    matrix, _, _, static_scaler, dynamic_scaler, names = fit_person_trip_matrix(
        age=numeric("age"), sex=numeric("sex"), employed=numeric("employed"),
        car_availability=numeric("car_availability"), income_class=numeric("income_class"),
        daily_longest=numeric("daily_longest_distance_from_home"),
        daily_total=numeric("daily_crowfly_total"),
        daily_longest_work=numeric("daily_longest_distance_from_work"),
        activity_chain_matrix=chains, consumed_before=numeric("crowfly_consumed_before_trip"),
        trip_position=numeric("trip_position_class"),
        departure_time=numeric("departure_time_normalized", 0.5),
        activity_duration_h=numeric("activity_duration_h"),
        target_distance=np.zeros(len(frame)), purpose_series=frame["purpose"],
        origin_purpose_series=frame["origin_purpose"], purpose_categories=SECONDARY_ACTIVITIES,
        person_static_scaler=static_scaler, person_dynamic_scaler=dynamic_scaler,
        use_target_distance=False, random_state=random_state,
    )
    return matrix, static_scaler, dynamic_scaler, names


@dataclass
class ShortTripClassifier:
    model: LogisticRegression
    static_scaler: object
    dynamic_scaler: object
    feature_names: list
    short_trip_threshold_m: float
    metrics: dict

    def predict_probability(self, *, age, income_class, daily_longest, daily_total,
                            daily_longest_work, activity_chain_vector, sex, employed,
                            car_availability, consumed_before, trip_position, departure_time,
                            activity_duration_h, purpose, origin_purpose):
        static = transform_person_static_vector(
            age, income_class, daily_longest, daily_total, daily_longest_work,
            activity_chain_vector, sex, employed, car_availability, self.static_scaler,
        )
        identity = np.eye(len(SECONDARY_ACTIVITIES), dtype=np.float32)
        purpose_hot = identity[SECONDARY_ACTIVITIES.index(purpose)]
        origin = ORIGIN_PURPOSE_REMAP.get(origin_purpose, origin_purpose)
        origin_hot = identity[SECONDARY_ACTIVITIES.index(origin)] if origin in SECONDARY_ACTIVITIES else np.zeros(len(identity))
        dynamic = transform_person_dynamic_vector(
            consumed_before, trip_position, departure_time, activity_duration_h, 0.0,
            purpose_hot, origin_hot, self.dynamic_scaler,
        )
        matrix = np.concatenate([static, dynamic], axis=1)
        return float(self.model.predict_proba(matrix)[0, 1])

    def predict_short(self, **features):
        return self.predict_probability(**features) >= 0.5


def train_classifier(frame, *, threshold_m, seed, validation_fraction=0.2, regularization_c=0.1):
    if not 0 < validation_fraction < 1 or regularization_c <= 0 or threshold_m <= 0:
        raise ValueError("Classifier requires a positive threshold and C, and validation fraction between 0 and 1")
    y = (frame["target_distance"].to_numpy(dtype=float) <= threshold_m).astype(int)
    if len(np.unique(y)) != 2 or frame["person_id"].nunique() < 2:
        raise ValueError("Short-trip classifier needs both classes and multiple persons for validation")
    splitter = GroupShuffleSplit(n_splits=20, test_size=validation_fraction, random_state=seed)
    for train, validation in splitter.split(frame, y, groups=frame["person_id"]):
        if len(np.unique(y[train])) == len(np.unique(y[validation])) == 2:
            break
    else:
        raise ValueError("Cannot form person-disjoint classifier training and validation sets containing both classes")

    # Neither scalers nor coefficients see the held-out persons. Keep that model
    # rather than refitting on the validation set after reporting its scores.
    x_train, static, dynamic, names = _feature_matrix(frame.iloc[train], random_state=seed)
    x_validation, _, _, _ = _feature_matrix(frame.iloc[validation], static, dynamic)
    weights = frame["person_weight"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Classifier person weights must be finite and positive")
    model = LogisticRegression(C=regularization_c, max_iter=1000, random_state=seed)
    # Normalize so the amount of regularization does not depend on weight units.
    model.fit(x_train, y[train], sample_weight=weights[train] / weights[train].mean())
    p_train = model.predict_proba(x_train)[:, 1]
    p_validation = model.predict_proba(x_validation)[:, 1]
    baseline = np.average(y[train], weights=weights[train])
    metrics = dict(
        training_rows=len(train), validation_rows=len(validation),
        training_persons=int(frame.iloc[train].person_id.nunique()),
        validation_persons=int(frame.iloc[validation].person_id.nunique()),
        training_log_loss=float(log_loss(y[train], p_train, sample_weight=weights[train])),
        validation_log_loss=float(log_loss(y[validation], p_validation, sample_weight=weights[validation])),
        validation_baseline_log_loss=float(log_loss(y[validation], np.full(len(validation), baseline), sample_weight=weights[validation])),
        validation_balanced_accuracy=float(balanced_accuracy_score(y[validation], p_validation >= .5, sample_weight=weights[validation])),
        validation_roc_auc=float(roc_auc_score(y[validation], p_validation, sample_weight=weights[validation])),
    )
    return ShortTripClassifier(model, static, dynamic, names, threshold_m, metrics)


def execute(context):
    if context.config("secondary_nn_use_target_distance"):
        return None
    classifier = train_classifier(
        prepare_training_data(context),
        threshold_m=float(context.config("secondary_nn_short_trip_threshold_m")),
        seed=int(context.config("random_seed")),
        validation_fraction=float(context.config("secondary_nn_short_classifier_validation_fraction")),
        regularization_c=float(context.config("secondary_nn_short_classifier_C")),
    )
    path = os.path.join(context.working_directory, MODEL_NAME)
    joblib.dump(classifier, path)
    with open(os.path.join(context.working_directory, "validation.json"), "w") as stream:
        json.dump(classifier.metrics, stream, indent=2)
    logger.info("Short-trip classifier held-out validation: %s", classifier.metrics)
    return (path,)
