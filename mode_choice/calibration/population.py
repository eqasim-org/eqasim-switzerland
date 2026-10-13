"""Reuse compact utility batches and one topology cache across calibration runs."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from mode_choice.defaults import (
    ALPHA_KEYS, MODES, CALIBRATION_COLUMNS, CALIBRATION_REFERENCE_MODE,
    CALIBRATION_DEFAULTS, CALIBRATION_LEVELS, CALIBRATION_CANTON_COLUMN,
)
from mode_choice.calibration.evaluation import calibrate_batches, result_report
from mode_choice.calibration.shares import load_targets
from mode_choice.calibration.cantons import calibrate_cantons


def calibration_level(context):
    level = context.config("mode_choice_calibration_level", CALIBRATION_DEFAULTS["mode_choice_calibration_level"])
    if level not in CALIBRATION_LEVELS:
        raise ValueError(f"mode_choice_calibration_level must be one of {CALIBRATION_LEVELS}")
    return level


def run(context, batches, parameters, sampler):
    level = calibration_level(context)
    target_paths = context.stage("data.microcensus.shares")
    columns = [*CALIBRATION_COLUMNS]
    if level == "canton":
        columns.append(CALIBRATION_CANTON_COLUMN)
    # Base utilities already include global and residence-canton ASCs. Keep only
    # compact sampling inputs in memory, with no repeated routing/cost work.
    batches = (batch.select(columns) for batch in batches)
    calibrated = deepcopy(parameters)
    if level == "global":
        targets = load_targets(target_paths[0])
        initial = np.array([parameters[ALPHA_KEYS[mode]] for mode in MODES], dtype=float)
        result = calibrate_batches(context, batches, initial, targets, sampler)
        calibrated.update({ALPHA_KEYS[mode]: float(alpha) for mode, alpha in zip(MODES, result.alphas)})
        choices, report = result.choices, result_report(result, targets)
    else:
        choices, report = calibrate_cantons(context, batches, calibrated, sampler, target_paths[1])
    report.update({
        "level": level, "reference_mode": CALIBRATION_REFERENCE_MODE,
        "tolerance": context.config("mode_choice_calibration_tolerance"),
    })
    directory = Path(context.path())
    # Final artifacts only: no per-iteration disk cache or modification of fits.
    (directory / "calibrated_parameters.yml").write_text(yaml.safe_dump(calibrated, sort_keys=True))
    (directory / "calibration.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return choices
