"""Reuse compact utility batches and one topology cache across calibration runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from dmc.parameters import Parameters

from mode_choice.defaults import (
    ALPHA_KEYS, MODES, CALIBRATION_REFERENCE_MODE,
    CALIBRATION_LEVELS,
)
from mode_choice.calibration.evaluation import calibrate_model, result_report
from mode_choice.calibration.shares import load_targets
from mode_choice.calibration.cantons import calibrate_cantons


def calibration_level(context):
    level = context.config("mode_choice_calibration_level")
    if level not in CALIBRATION_LEVELS:
        raise ValueError(f"mode_choice_calibration_level must be one of {CALIBRATION_LEVELS}")
    return level


def run(context, model, parameters):
    level = calibration_level(context)
    target_paths = context.stage("data.microcensus.shares")
    calibrated = deepcopy(parameters)
    if level == "global":
        targets = load_targets(target_paths[0])
        initial = np.array([parameters[ALPHA_KEYS[mode]] for mode in MODES], dtype=float)
        result = calibrate_model(context, model, initial, targets)
        calibrated.update({ALPHA_KEYS[mode]: float(alpha) for mode, alpha in zip(MODES, result.alphas)})
        choices, report = result.choices, result_report(result, targets)
    else:
        choices, report = calibrate_cantons(context, model, calibrated, target_paths[1])
    report.update({
        "level": level, "reference_mode": CALIBRATION_REFERENCE_MODE,
        "tolerance": context.config("mode_choice_calibration_tolerance"),
    })
    directory = Path(context.path())
    # Final artifacts only: no per-iteration disk cache or modification of fits.
    Parameters(calibrated).to_yaml(directory / "calibrated_parameters.yml", sort_keys=True)
    (directory / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return choices
