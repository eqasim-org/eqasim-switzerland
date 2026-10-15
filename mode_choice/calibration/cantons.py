"""Calibrate additive residence-canton ASCs, retaining global mode constants."""
import logging

import numpy as np
import polars as pl

from mode_choice.defaults import MODES, CANTON_ALPHA_KEYS, CALIBRATION_REFERENCE_MODE
from mode_choice.calibration.evaluation import calibrate_model, result_report
from mode_choice.calibration.shares import load_canton_targets

logger = logging.getLogger(__name__)


def calibrate_cantons(context, model, parameters, target_path):
    targets = load_canton_targets(target_path)
    cantons = model.cantons()
    missing = set(cantons) - set(targets)
    if missing:
        raise ValueError(f"Missing mode-share targets for residence cantons: {sorted(missing)}")
    reports, choices = {}, []
    # Each canton is independent: stop converged cantons and retain the best
    # evaluated state per canton instead of compromising with a national loss.
    for canton in cantons:
        logger.info("Calibrating residence canton %s", canton)
        initial = np.array([parameters.get(CANTON_ALPHA_KEYS[mode], {}).get(canton, 0.0) for mode in MODES])
        try:
            result = calibrate_model(context, model, initial, targets[canton], canton=canton)
        except ValueError as exc:
            raise ValueError(f"Canton {canton}: {exc}") from exc
        for mode, alpha in zip(MODES, result.alphas):
            if mode != CALIBRATION_REFERENCE_MODE:
                parameters.setdefault(CANTON_ALPHA_KEYS[mode], {})[canton] = float(alpha)
        reports[canton] = result_report(result, targets[canton])
        choices.append(result.choices)
    return pl.concat(choices).sort("trip_id"), {
        "converged": all(report["converged"] for report in reports.values()),
        "evaluations": sum(report["evaluations"] for report in reports.values()),
        "cantons": reports,
        "skipped_target_cantons": sorted(set(targets) - set(reports)),
    }
