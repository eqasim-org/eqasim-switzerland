"""Reusable fit/transform/predict interface for whole-diary matching."""

from dataclasses import dataclass
import copy

import numpy as np
import pandas as pd

from .artifacts import ModelStore
from .donors import DonorSelector, retrieval_space, validate_ids
from .features import CONTINUOUS, CATEGORICAL
from .model import encode


@dataclass
class MatchingPrediction:
    """Donor IDs/row indices in recipient order, with explicit unmatched values (-1).

    Arrays have shape (people,) for one draw and (people, draws) for ensembles.
    Donor IDs refer to the survey's mz_id column; row indices refer to donors_.
    """

    person_ids: np.ndarray
    donor_ids: np.ndarray
    donor_rows: np.ndarray
    diagnostics: pd.DataFrame
    pools: list

    @property
    def mapping(self):
        if self.donor_ids.ndim != 1:
            raise ValueError("mapping is available for single-draw predictions only")
        return pd.DataFrame({"person_id": self.person_ids, "mz_person_id": self.donor_ids})

    @property
    def unmatched_person_ids(self):
        missing = self.donor_ids < 0
        if missing.ndim == 2:
            missing = missing.all(axis=1)
        return set(self.person_ids[missing])


class DiaryMatcher:
    """Fit harmonized survey donors once, then match any compatible population.

    Raw source/census semantics are prepared by inputs.prepare_inputs. Training
    and prediction never mutate caller frames. Preprocessors, fitted model,
    donors and normalized donor embeddings are owned by this wrapper.
    """

    def __init__(self, options, *, cache_directory=None):
        self.options = options
        self.store = ModelStore(cache_directory)

    def fit(self, donors, trips):
        self._validate_frame(donors, "mz_id", "Survey donors")
        if donors.empty:
            raise ValueError("At least one survey donor is required")
        source = donors.copy(deep=True)
        fitted, processor, report = self.store.fit_or_load(source, trips, self.options)
        embeddings = encode(fitted, processor.transform(source))
        self.model_, self.processor_ = fitted, processor
        self.training_report_, self.donors_, self.donor_embeddings_ = report, source, embeddings
        self.selector_ = DonorSelector(**self.options.retrieval)
        from .training import split_training_rows
        train, validation = split_training_rows(source, self.options.validation_fraction, self.options.random_seed)
        self.training_ids_ = source.iloc[train].mz_id.to_numpy(copy=True)
        self.validation_ids_ = source.iloc[validation].mz_id.to_numpy(copy=True)
        return self

    def with_donors(self, donor_ids):
        """Restrict donor availability without refitting or changing the encoder."""
        self._require_fitted()
        ids = pd.Index(donor_ids)
        if ids.has_duplicates or not ids.isin(self.donors_.mz_id).all() or ids.empty:
            raise ValueError("Donor IDs must be a non-empty unique subset of fitted donors")
        rows = pd.Series(np.arange(len(self.donors_)), index=self.donors_.mz_id).loc[ids].to_numpy()
        view = copy.copy(self)
        view.donors_ = self.donors_.iloc[rows].reset_index(drop=True).copy()
        view.donor_embeddings_ = self.donor_embeddings_[rows].copy()
        return view

    def match(self, population, *, donor_ids=None, n_draws=1, random_seed=None):
        """Match using the fitted encoder and, optionally, a restricted donor pool."""
        matcher = self if donor_ids is None else self.with_donors(donor_ids)
        return matcher.predict(population, n_draws=n_draws, random_seed=random_seed)

    def transform(self, frame):
        self._require_fitted()
        return encode(self.model_, self.processor_.transform(frame))

    def predict(self, population, *, n_draws=1, random_seed=None):
        self._require_fitted()
        self._validate_frame(population, "person_id", "Prediction population")
        target = population.copy(deep=True)
        target_embeddings = self.transform(target)
        spaces = retrieval_space(self.donors_, target, self.donor_embeddings_, target_embeddings,
                                 self.options.raw_feature_weight)
        rows, diagnostics, pools = self.selector_.predict(self.donors_, target, *spaces,
            random_seed=self.options.random_seed if random_seed is None else random_seed, n_draws=n_draws)
        ids = np.full(rows.shape, -1, dtype=np.int64)
        valid = rows >= 0
        ids[valid] = self.donors_["mz_id"].to_numpy(dtype=np.int64)[rows[valid]]
        return MatchingPrediction(target["person_id"].to_numpy(copy=True), ids, rows, diagnostics, pools)

    def _require_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("DiaryMatcher.fit must be called before transform or predict")

    @staticmethod
    def _validate_frame(frame, id_column, label):
        required = set(CONTINUOUS + CATEGORICAL + [id_column])
        if id_column == "mz_id":
            required.add("person_weight")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} lack harmonized columns: {', '.join(missing)}")
        validate_ids(frame, id_column, label)
        for key in ("employed", "is_student"):
            if frame[key].isna().any() or not frame[key].isin([0, 1]).all():
                raise ValueError(f"{label} must have non-missing binary {key} status")
