"""Pipeline option declarations and typed settings for the reusable V2 estimator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.utils import coerce_boolean_series


def configure(context):
    context.config("random_seed")
    context.config("matching_embedding_top_k", 20)
    context.config("matching_embedding_temperature", 0.5)
    context.config("matching_embedding_epochs", 100)
    context.config("matching_embedding_batch_size", 128)
    context.config("matching_embedding_lr", 5.0e-3)
    context.config("matching_embedding_lr_decay_step", 15)
    context.config("matching_embedding_lr_decay_gamma", 0.5)
    context.config("matching_embedding_dropout", 0.2)
    context.config("matching_embedding_weight_decay", 3.0e-3)
    context.config("matching_embedding_max_sequence_len", 12)
    context.config("matching_loss_weight_activity", 2.0)
    context.config("matching_loss_weight_trip_count", 1.5)
    context.config("matching_loss_weight_mode", 0.1)
    context.config("matching_loss_weight_departure", 0.1)
    context.config("matching_loss_weight_distance", 0.3)
    context.config("matching_loss_weight_reconstruction", 0.2)
    context.config("matching_loss_weight_latent_norm", 0.05)
    context.config("matching_loss_weight_latent_spread", 0.2)
    context.config("matching_embedding_geo_proj_dim", 8)
    context.config("matching_embedding_cars_proj_dim", 4)
    context.config("matching_embedding_emp_proj_dim", 6)
    context.config("matching_embedding_misc_proj_dim", 8)
    context.config("matching_use_class_weighting", False)
    context.config("matching_latent_noise_std", 0.08)
    context.config("matching_trip_count_scale", 5.0)
    context.config("matching_trip_distance_scale", 30000.0)
    context.config("matching_loss_weight_duration", 0.5)
    context.config("matching_loss_weight_activity_duration", 0.5)
    context.config("matching_embedding_validation_fraction", 0.15)
    context.config("matching_embedding_patience", 12)
    context.config("matching_v2_raw_feature_weight", 0.0)
    context.config("matching_v2_balance_strength", 0.0)
    context.config("matching_v2_query_batch_size", 4096)
    context.config("specific_day_scenario", default="workday")
    context.config("overwrite_matching_embedding_model", default=True)


def validate_options(context):
    positive_ints = ("matching_embedding_top_k", "matching_embedding_epochs", "matching_embedding_batch_size",
                     "matching_embedding_max_sequence_len", "matching_embedding_lr_decay_step",
                     "matching_embedding_patience", "matching_v2_query_batch_size",
                     "matching_embedding_geo_proj_dim", "matching_embedding_cars_proj_dim",
                     "matching_embedding_emp_proj_dim", "matching_embedding_misc_proj_dim")
    for name in positive_ints:
        value = context.config(name)
        if not np.isfinite(float(value)) or int(value) != float(value) or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("matching_embedding_temperature", "matching_trip_count_scale", "matching_trip_distance_scale",
                 "matching_embedding_lr", "matching_embedding_lr_decay_gamma"):
        if not np.isfinite(float(context.config(name))) or float(context.config(name)) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("matching_v2_raw_feature_weight", "matching_v2_balance_strength",
                 "matching_embedding_weight_decay", "matching_latent_noise_std"):
        if not np.isfinite(float(context.config(name))) or float(context.config(name)) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0 <= float(context.config("matching_embedding_dropout")) < 1:
        raise ValueError("matching_embedding_dropout must be in [0, 1)")


def as_bool(value):
    return bool(coerce_boolean_series(pd.Series([value])).iloc[0])


@dataclass(frozen=True)
class MatchingOptions:
    """Resolved estimator settings; pipeline defaults are declared by configure()."""

    random_seed: int
    training: dict
    targets: dict
    validation_fraction: float
    retrieval: dict
    raw_feature_weight: float
    overwrite: bool

    @classmethod
    def from_context(cls, context, *, random_seed=None):
        validate_options(context)
        seed = int(context.config("random_seed") if random_seed is None else random_seed)
        losses = {key: float(context.config("matching_loss_weight_" + key)) for key in (
            "activity", "trip_count", "mode", "departure", "distance", "duration", "activity_duration",
            "reconstruction", "latent_norm", "latent_spread")}
        if not all(np.isfinite(value) and value >= 0 for value in losses.values()):
            raise ValueError("Matching loss weights must be finite and non-negative")
        if not any(losses[key] for key in ("activity", "trip_count", "mode", "departure", "distance", "duration", "activity_duration")):
            raise ValueError("At least one behavioural matching loss must be positive")
        training = {"random_seed": seed, "loss_weights": losses,
                    "proj_dims": {key: int(context.config("matching_embedding_" + key + "_proj_dim"))
                                  for key in ("geo", "cars", "emp", "misc")},
                    "use_class_weighting": as_bool(context.config("matching_use_class_weighting")),
                    "latent_noise_std": float(context.config("matching_latent_noise_std")),
                    "patience": int(context.config("matching_embedding_patience"))}
        for key in ("epochs", "batch_size", "lr_decay_step"):
            training[key] = int(context.config("matching_embedding_" + key))
        for key in ("lr", "dropout", "weight_decay", "lr_decay_gamma"):
            training[key] = float(context.config("matching_embedding_" + key))
        targets = {"max_sequence_len": int(context.config("matching_embedding_max_sequence_len")),
                   "trip_count_scale": float(context.config("matching_trip_count_scale")),
                   "trip_distance_scale": float(context.config("matching_trip_distance_scale"))}
        return cls(seed, training, targets,
                   float(context.config("matching_embedding_validation_fraction")),
                   {"top_k": int(context.config("matching_embedding_top_k")),
                    "temperature": float(context.config("matching_embedding_temperature")),
                    "balance_strength": float(context.config("matching_v2_balance_strength")),
                    "query_batch_size": int(context.config("matching_v2_query_batch_size"))},
                   float(context.config("matching_v2_raw_feature_weight")),
                   as_bool(context.config("overwrite_matching_embedding_model")))

    @classmethod
    def from_config(cls, values):
        """Build settings outside Synpp, using the same option declarations/defaults."""
        context = _Configuration(dict(values))
        configure(context)
        return cls.from_context(context)


class _Configuration:
    def __init__(self, values):
        self.values = values

    def config(self, name, default=...):
        if name not in self.values:
            if default is ...:
                raise ValueError(f"Missing required V2 option: {name}")
            self.values[name] = default
        return self.values[name]
