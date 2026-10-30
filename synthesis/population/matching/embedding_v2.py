"""Compatibility imports; neural components now live in the V2 package."""

from .V2.runtime import torch, nn, F, to_tensors as _to_tensors
from .V2.features import (CONTINUOUS, CATEGORICAL, CATEGORICAL_GROUPS,
                          fit_feature_processors, transform_inputs)
from .V2.targets import build_sequence_targets
from .V2.model import EmbeddingGRUModel, encode
from .V2.training import split_training_rows, train_embedding_model, _class_weights
from .V2.losses import MatchingLoss

_loss_per_person = MatchingLoss.per_person
