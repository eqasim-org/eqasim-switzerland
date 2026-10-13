"""Damped, PT-relative log-ratio ASC updates from the original Kai optimizer."""
from dataclasses import dataclass
import logging

import numpy as np

from mode_choice.defaults import (
    MODES, CALIBRATION_DEFAULTS, CALIBRATION_REFERENCE_INDEX, CALIBRATION_SHARE_FLOOR,
    CALIBRATION_MAX_INERTIA, CALIBRATION_DAMPING_RATE,
)

logger = logging.getLogger(__name__)


def share_vector(shares):
    values = np.asarray(shares, dtype=float)
    if values.shape != (len(MODES),) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Mode shares must contain five finite, nonnegative values")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mode shares must have a positive total")
    return values / total


def update_alphas(alphas, simulated, targets, iteration):
    """Keep PT fixed; floor zero shares so all updates remain finite.

    alpha_m += (1 - beta) * [log(target_m / simulated_m)
                            - log(target_pt / simulated_pt)]
    beta = min(0.8, 1 - 1 / (0.5 * iteration + 1)).
    """
    simulated, targets = share_vector(simulated), share_vector(targets)
    correction = np.log(np.maximum(targets, CALIBRATION_SHARE_FLOOR)) - np.log(
        np.maximum(simulated, CALIBRATION_SHARE_FLOOR))
    correction -= correction[CALIBRATION_REFERENCE_INDEX]
    inertia = min(CALIBRATION_MAX_INERTIA, 1 - 1 / (CALIBRATION_DAMPING_RATE * iteration + 1))
    return np.asarray(alphas, dtype=float) + (1 - inertia) * correction


@dataclass
class CalibrationResult:
    alphas: np.ndarray
    shares: np.ndarray
    choices: object
    converged: bool
    history: list
    best_evaluation: int


def calibrate(evaluate, initial_alphas, targets, *,
              max_evaluations=CALIBRATION_DEFAULTS["mode_choice_calibration_max_evaluations"],
              tolerance=CALIBRATION_DEFAULTS["mode_choice_calibration_tolerance"]):
    """Evaluate complete population choices, returning the best evaluated state.

    evaluate(alphas) returns (trip shares in MODES order, choices). The caller
    reuses base utilities, routing and the same person/tour random streams.
    Convergence measures target error, never just a lack of change in shares.
    """
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, (int, np.integer)) or max_evaluations <= 0:
        raise ValueError("Calibration max_evaluations must be a positive integer")
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("Calibration tolerance must be finite and between zero and one")
    targets = share_vector(targets)
    if targets[CALIBRATION_REFERENCE_INDEX] <= 0:
        raise ValueError("Calibration requires a positive PT target share")
    alphas = np.asarray(initial_alphas, dtype=float).copy()
    if alphas.shape != targets.shape or not np.isfinite(alphas).all():
        raise ValueError("Initial alphas must contain five finite values")
    history, best, best_error = [], None, float("inf")
    for iteration in range(max_evaluations):
        shares, choices = evaluate(alphas.copy())
        shares = share_vector(shares)
        error = float(np.max(np.abs(shares - targets)))
        history.append({"evaluation": iteration + 1, "max_share_error": error,
                        "alphas": dict(zip(MODES, alphas.tolist())),
                        "shares": dict(zip(MODES, shares.tolist()))})
        logger.info("ASC calibration %d/%d: max share error %.6f; shares %s",
                    iteration + 1, max_evaluations, error, history[-1]["shares"])
        if error < best_error:
            best_error = error
            best = CalibrationResult(alphas.copy(), shares.copy(), choices, error <= tolerance,
                                     history, iteration + 1)
        if error <= tolerance:
            break
        if iteration + 1 < max_evaluations:
            alphas = update_alphas(alphas, shares, targets, iteration)
    if not best.converged:
        logger.warning("ASC calibration did not reach tolerance %.6f after %d evaluations; "
                       "using evaluated state %d with max share error %.6f. "
                       "Check sample size, mode availability and target compatibility.",
                       tolerance, len(history), best.best_evaluation, best_error)
    return best
