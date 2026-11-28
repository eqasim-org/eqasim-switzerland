"""Fit national ASCs without canton effects, then fit additive canton effects."""
import logging

import numpy as np

from mode_choice.defaults import MODES, ALPHA_KEYS, CANTON_ALPHA_KEYS
from mode_choice.calibration.cantons import calibrate_cantons
from mode_choice.calibration.evaluation import calibrate_model, result_report
from mode_choice.calibration.shares import load_targets

logger = logging.getLogger(__name__)


def calibrate_mixed(context, model, parameters, target_paths):

    # Cancel fitted canton effects already included in the immutable batch files.

    cantons = model.cantons()
    canton_deltas = {
        canton: [-parameters.get(CANTON_ALPHA_KEYS[mode], {}).get(canton, 0.0) for mode in MODES]
        for canton in cantons
    }
    initial = np.array([parameters[ALPHA_KEYS[mode]] for mode in MODES], dtype=float)
    targets = load_targets(target_paths[0])

    try:

        # Fit national constants with every canton-specific ASC set to zero.

        logger.info("Mixed ASC calibration: fitting global constants with zero canton ASCs")
        model.set_asc_offsets(canton_deltas=canton_deltas)
        result = calibrate_model(context, model, initial, targets)
        global_report = result_report(result, targets)
        parameters.update({ALPHA_KEYS[mode]: float(alpha) for mode, alpha in zip(MODES, result.alphas)})
        for key in CANTON_ALPHA_KEYS.values():
            parameters[key] = dict.fromkeys(set(parameters.get(key, {})) | set(cantons), 0.0)

        # Hold national constants fixed and calibrate each canton's additions.

        logger.info("Mixed ASC calibration: fitting canton additions to the calibrated global constants")
        model.set_asc_offsets(result.alphas - initial, canton_deltas=canton_deltas)
        del result  # The national choices are superseded; retain only their diagnostics.
        choices, report = calibrate_cantons(context, model, parameters, target_paths[1])
        report['global'] = global_report
        report['converged'] = global_report['converged'] and report['converged']
        report['evaluations'] += global_report['evaluations']
        return choices, report
    finally:

        # Release phase-specific arrays and leave the reusable model unchanged.

        model.set_asc_offsets()
