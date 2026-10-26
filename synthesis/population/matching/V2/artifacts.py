"""Fingerprinting and atomic storage of locally trusted V2 model artifacts."""

import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile

import pandas as pd

from . import configuration, features, losses, model, runtime, targets, training
from .features import FeatureProcessor, CONTINUOUS, CATEGORICAL, source_token
from .targets import SequenceTargetBuilder
from .training import EmbeddingTrainer, split_training_rows

logger = logging.getLogger("synpp")
ARTIFACT_VERSION = 4


def implementation_token():
    """Include every source file affecting training or serialization compatibility."""
    return source_token(configuration, features, losses, model, runtime, targets, training)


class ModelStore:
    """Load only compatible local artifacts; fit and save atomically otherwise."""

    def __init__(self, directory=None):
        self.directory = None if directory is None else Path(directory)

    def fit_or_load(self, donors, trips, options):
        if runtime.torch is None:
            raise RuntimeError("PyTorch is required for matching V2")
        train, validation = split_training_rows(donors, options.validation_fraction, options.random_seed)
        identity = {"version": ARTIFACT_VERSION, "training": options.training, "targets": options.targets,
                    "train_rows": train.tolist(), "validation_rows": validation.tolist(),
                    "torch": str(runtime.torch.__version__), "implementation": implementation_token()}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
        for frame in (donors[["mz_id", "person_weight"] + CONTINUOUS + CATEGORICAL],
                      trips[["person_id", "trip_id", "departure_time", "arrival_time", "crowfly_distance", "purpose", "mode"]]):
            digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
        fingerprint = digest.hexdigest()
        path = None if self.directory is None else self.directory / "matching_embedding_model.pkl"
        if path is not None and path.exists() and not options.overwrite:
            # Stage artifacts contain sklearn state and are trusted local files, never external downloads.
            payload = runtime.torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("version") == ARTIFACT_VERSION and payload.get("fingerprint") == fingerprint:
                with runtime.torch.random.fork_rng(devices=[]):
                    fitted = model.EmbeddingGRUModel(**payload["model_kwargs"])
                fitted.load_state_dict(payload["model_state_dict"])
                fitted.eval()
                processor = FeatureProcessor.from_fitted(payload["scaler"], payload["mappings"])
                fitted.training_report = payload["training_report"]
                logger.info("Loaded compatible V2 model from stage cache")
                return fitted, processor, payload["training_report"]
            logger.info("V2 model inputs/options/implementation changed; retraining")
        processor = FeatureProcessor().fit(donors.iloc[train])
        inputs = processor.transform(donors)
        observed = SequenceTargetBuilder(**options.targets).build(donors, trips)
        trainer = EmbeddingTrainer(**options.training).fit(inputs, observed, train, validation)
        report = {**trainer.report_, "truncated_training_diaries": observed["truncated_diaries"]}
        if path is not None:
            payload = {"fingerprint": fingerprint, "version": ARTIFACT_VERSION,
                       "model_state_dict": {key: value.cpu() for key, value in trainer.model_.state_dict().items()},
                       "model_kwargs": trainer.model_kwargs_, "scaler": processor.scaler_,
                       "mappings": processor.mappings_, "training_report": report}
            self._save(payload, path)
        return trainer.model_, processor, report

    @staticmethod
    def _save(payload, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as temporary:
            temporary_path = temporary.name
        try:
            runtime.torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
