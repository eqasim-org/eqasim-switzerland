"""Evaluate canonical DMC utilities and propagate generated passenger history."""
from dataclasses import dataclass

import jax
import numpy as np
from scipy.special import log_softmax

from dmc.models import model_jax


@dataclass
class Prediction:
    probabilities: np.ndarray
    conditional: np.ndarray
    prior_states: np.ndarray
    passenger_index: int | None

    @property
    def prior_passenger(self):
        """Probability that the immediately preceding trip was passenger."""
        return self.prior_states[:, 2]

    @property
    def prior_any_passenger(self):
        """Probability of any earlier passenger trip during the day."""
        return self.prior_states[:, 1:].sum(axis=1)


def layout(keys):
    """Rows of each person's selected day, in trip order, padded with -1."""
    if keys.duplicated(['person_id', 'trip_id']).any():
        raise ValueError('Duplicate trip keys in replay')
    if not keys.reset_index(drop=True).equals(keys.sort_values(['person_id', 'trip_id']).reset_index(drop=True)):
        raise ValueError('Replay keys must be sorted by person and trip')
    lengths = keys.groupby('person_id', sort=False).size().to_numpy(int)
    if len(lengths) == 0:
        raise ValueError('No trips to replay')
    rows = np.full((len(lengths), lengths.max()), -1, dtype=int)
    person = np.repeat(np.arange(len(lengths)), lengths)
    starts = np.repeat(np.r_[0, np.cumsum(lengths)[:-1]], lengths)
    rows[person, np.arange(len(keys)) - starts] = np.arange(len(keys))
    return rows


def predictor(context, coefficients, modes):
    """Compile production utilities once, reusing them for interventions."""
    return jax.jit(lambda arrays: model_jax.build_utilities(context, arrays, coefficients, modes, 'car_passenger' not in modes))


def predict(arrays, evaluate, coefficients, modes, rows):
    inputs = dict(arrays)
    inputs['passenger_history'] = np.zeros(len(inputs['age']))
    inputs['passenger_history_any'] = np.zeros(len(inputs['age']))
    utility, available = evaluate(inputs)
    utility, available = np.asarray(utility), np.asarray(available)
    # States: never passenger; earlier passenger, last trip another mode; last trip passenger.
    logits = np.repeat(utility[:, None, :], 3, axis=1)
    cp = modes.index('car_passenger') if 'car_passenger' in modes else None
    if cp is not None:
        logits[:, 1:, cp] += coefficients['beta_passenger_history_any']
        logits[:, 2, cp] += coefficients['beta_passenger_history']
    log_q = log_softmax(np.where(available[:, None, :] > 0, logits, -np.inf), axis=2)
    conditional = np.exp(log_q)
    probabilities = np.empty_like(utility)
    prior = np.empty((len(utility), 3))
    state = np.zeros((len(rows), 3)); state[:, 0] = 1.
    for step in range(rows.shape[1]):
        active = rows[:, step] >= 0; index = rows[active, step]
        old = state[active].copy()
        prior[index] = old
        probabilities[index] = np.einsum('ns,nsm->nm', old, conditional[index])
        if cp is not None:
            state[active, 0] = old[:, 0] * (1 - conditional[index, 0, cp])
            state[active, 1] = (old[:, 1] * (1 - conditional[index, 1, cp])
                                + old[:, 2] * (1 - conditional[index, 2, cp]))
            state[active, 2] = probabilities[index, cp]
    if (not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-12)
            or np.any(probabilities[available <= 0] != 0)):
        raise ValueError('Invalid generated-history probabilities or changed availability')
    return Prediction(probabilities, conditional, prior, cp)


def sample(prediction, rows, seed):
    """One seeded sequential replay, using generated choices only."""
    return sample_many(prediction, rows, np.random.default_rng(seed).random((1, len(prediction.probabilities))))[0]


def sample_many(prediction, rows, draws):
    cp = prediction.passenger_index if prediction.passenger_index is not None else -1
    result = np.empty(draws.shape, dtype=np.int8)
    state = np.zeros((len(draws), len(rows)), dtype=np.int8)
    for step in range(rows.shape[1]):
        active = rows[:, step] >= 0; index = rows[active, step]
        cumulative = prediction.conditional[index[None, :], state[:, active], :].cumsum(axis=2)
        cumulative[:, :, -1] = 1.
        chosen = (draws[:, index, None] >= cumulative).sum(axis=2).astype(np.int8)
        result[:, index] = chosen
        state[:, active] = np.where(chosen == cp, 2, np.where(state[:, active] > 0, 1, 0)).astype(np.int8)
    return result


def sampling_shares(prediction, rows, weights, seed, repetitions, batch_size=20):
    """Bound memory by batching draws; same seed gives common draws across models."""
    if repetitions < 2:
        raise ValueError('At least two sampling repetitions are required')
    rng = np.random.default_rng(seed)
    shares = []; first = None
    for start in range(0, repetitions, batch_size):
        draws = rng.random((min(batch_size, repetitions - start), len(weights)))
        choices = sample_many(prediction, rows, draws)
        if first is None:
            first = choices[0].copy()
        shares.extend(np.bincount(row, weights=weights, minlength=prediction.probabilities.shape[1]) / weights.sum() for row in choices)
    return np.asarray(shares), first
