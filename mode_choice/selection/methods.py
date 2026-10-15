"""Stable categorical choice and Gumbel-max on conditional total utilities."""
import numpy as np
from mode_choice.defaults import SAMPLING_METHODS


def select(log_weights, rng, method):
    # -inf represents infeasible paths. Subtracting the maximum preserves both
    # methods and avoids overflow without clipping meaningful utility differences.
    scores = np.asarray(log_weights, dtype=np.float64)
    maximum = scores.max()
    if not np.isfinite(maximum):
        raise ValueError("No feasible alternative with finite total utility")
    scores = scores - maximum
    if method == "gumbel":
        return int(np.argmax(scores + rng.gumbel(size=len(scores))))
    if method == "probability":
        weights = np.exp(scores)
        cumulative = weights.cumsum()
        return min(int(cumulative.searchsorted(rng.random() * cumulative[-1], side="right")), len(scores) - 1)
    raise ValueError(f"Sampling method must be one of {SAMPLING_METHODS}")
