"""Feature harmonization and constrained whole-diary donor selection for V2."""

import logging

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.neighbors import NearestNeighbors

from .features import prepare_features

logger = logging.getLogger("synpp")


def validate_ids(frame, column, label):
    if frame[column].isna().any() or frame[column].duplicated().any():
        raise ValueError(f"{label} must have unique, non-missing {column} values")


def retrieval_space(source, target, source_embedding, target_embedding, raw_weight):
    """Learned behaviour is the default; optional raw-space regularization is explicit."""
    spaces = [source_embedding / np.sqrt(2), target_embedding / np.sqrt(2)]
    if raw_weight:
        from .features import fit_feature_processors, transform_inputs
        scaler, mappings = fit_feature_processors(source)
        for i, frame in enumerate((source, target)):
            x = transform_inputs(frame, scaler, mappings)
            raw = np.column_stack([x["cont"]] + [pd.get_dummies(
                pd.Categorical(x[col], categories=range(x["cardinalities"][col]))).to_numpy(dtype=float)
                for col in mappings])
            spaces[i] = np.column_stack([spaces[i], raw * np.sqrt(raw_weight / raw.shape[1])])
    return tuple(spaces)


def transport_probabilities(indices, costs, weights, temperature, balance_strength=0.0, iterations=60):
    """Sparse semi-relaxed entropic transport; each recipient has mass exactly one.

    Minimize <P,C> + tau*KL(P|a*b) + rho*KL(P.T*1|b), with P*1=a.
    The donor marginal is soft because local support and synthetic demographics
    need not permit the survey's marginal exactly. Zero rho is weighted hot deck.
    """
    if temperature <= 0 or not np.isfinite(temperature) or balance_strength < 0 or not np.isfinite(balance_strength):
        raise ValueError("Temperature must be positive and balance strength non-negative")
    weights = np.asarray(weights, dtype=float)
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise ValueError("Donor survey weights must be finite and positive")
    reachable = np.bincount(indices.ravel(), minlength=len(weights)) > 0
    prior = np.where(reachable, weights, 0)
    prior = prior / prior.sum() * len(indices)
    log_kernel = -np.asarray(costs, dtype=float) / temperature + np.log(prior[indices])
    log_v = np.zeros(len(weights))
    exponent = balance_strength / (balance_strength + temperature)
    for _ in range(iterations if balance_strength else 0):
        log_p = log_kernel + log_v[indices]
        log_p -= logsumexp(log_p, axis=1, keepdims=True)
        mass = np.bincount(indices.ravel(), weights=np.exp(log_p).ravel(), minlength=len(weights))
        updated = np.zeros_like(log_v)
        updated[reachable] = exponent * (np.log(prior[reachable]) - np.log(np.maximum(mass[reachable], 1e-300))
                                         + log_v[reachable])
        delta = np.max(np.abs(updated - log_v))
        log_v = updated
        if delta < 1e-7:
            break
    log_p = log_kernel + log_v[indices]
    return np.exp(log_p - logsumexp(log_p, axis=1, keepdims=True))


def select_donors(source, target, source_space, target_space, *, rng, top_k, temperature,
                   balance_strength, query_batch_size, n_draws=1):
    """Only employment and student status define hard inference masks.

    No exact or hierarchical age, income, sex, children, geography, car-access,
    collective-housing or work-type cells constrain the learned neighbourhood.
    Optional marginal balancing is off by default; n_draws supports evaluation.
    """
    assigned = np.full((len(target), n_draws), -1, dtype=np.int64)
    distances = np.full(len(target), np.nan)
    ess = np.full(len(target), np.nan)
    support = np.zeros(len(target), dtype=int)
    keys = ["employed", "is_student"]
    source_groups = source.groupby(keys, sort=True).indices
    diagnostics = []
    for key, target_rows in target.groupby(keys, sort=True).indices.items():
        source_rows = source_groups.get(key, np.array([], dtype=int))
        if not len(source_rows):
            continue
        neighbors = min(top_k, len(source_rows))
        knn = NearestNeighbors(n_neighbors=neighbors, metric="euclidean", n_jobs=1).fit(source_space[source_rows])
        idx = np.empty((len(target_rows), neighbors), dtype=np.int32)
        costs = np.empty_like(idx, dtype=np.float64)
        for start in range(0, len(target_rows), query_batch_size):
            batch_rows = target_rows[start:start + query_batch_size]
            distance, local = knn.kneighbors(target_space[batch_rows])
            idx[start:start + len(batch_rows)] = local
            costs[start:start + len(batch_rows)] = distance ** 2
        weights = source.iloc[source_rows]["person_weight"].to_numpy(dtype=float)
        probabilities = transport_probabilities(idx, costs, weights, temperature, balance_strength)
        cdf = np.cumsum(probabilities, axis=1)
        cdf[:, -1] = 1.0
        uniforms = rng.random_sample((len(target_rows), n_draws))
        draws = np.empty_like(uniforms, dtype=int)
        for draw in range(n_draws):
            draws[:, draw] = (cdf < uniforms[:, draw, None]).sum(axis=1)
        chosen = np.take_along_axis(idx, draws, axis=1)
        assigned[target_rows] = source_rows[chosen]
        distances[target_rows] = np.take_along_axis(costs, draws, axis=1).mean(axis=1)
        ess[target_rows] = 1 / (probabilities ** 2).sum(axis=1)
        support[target_rows] = len(source_rows)
        expected = np.bincount(idx.ravel(), weights=probabilities.ravel(), minlength=len(weights))
        prior = weights / weights.sum()
        diagnostics.append({"employed": int(key[0]), "is_student": int(key[1]),
                            "recipients": len(target_rows), "donors": len(source_rows),
                            "reachable_donors": int(np.count_nonzero(expected)),
                            "expected_donor_tvd": float(np.abs(expected / len(target_rows) - prior).sum() / 2)})
    details = pd.DataFrame({"person_id": target["person_id"].to_numpy(),
                            "donor_pool_size": support, "distance_squared": distances, "candidate_ess": ess})
    return (assigned[:, 0] if n_draws == 1 else assigned), details, diagnostics


class DonorSelector:
    """Survey-weighted nearest donors constrained only by employment/student status."""

    def __init__(self, *, top_k, temperature, balance_strength, query_batch_size):
        self.options = dict(top_k=top_k, temperature=temperature,
                            balance_strength=balance_strength, query_batch_size=query_batch_size)

    def predict(self, donors, population, donor_space, population_space, *, random_seed, n_draws=1):
        if not isinstance(n_draws, (int, np.integer)) or n_draws < 1:
            raise ValueError("n_draws must be a positive integer")
        return select_donors(donors, population, donor_space, population_space,
                            rng=np.random.RandomState(random_seed), n_draws=n_draws, **self.options)
