"""Calibrate additive residence-canton ASCs, retaining global mode constants."""
from collections import defaultdict
import logging

import numpy as np
import polars as pl

from mode_choice.defaults import MODES, CANTON_ALPHA_KEYS, CALIBRATION_CANTON_COLUMN, CALIBRATION_REFERENCE_MODE
from mode_choice.calibration.evaluation import calibrate_batches, result_report
from mode_choice.calibration.shares import load_canton_targets

logger = logging.getLogger(__name__)


def calibrate_cantons(context, batches, parameters, sampler, target_path):
    targets = load_canton_targets(target_path)
    groups = defaultdict(list)
    for batch in batches:
        # Persons are never split between batches. Verify residence is constant
        # within each complete person's chain before partitioning by canton.
        if batch[CALIBRATION_CANTON_COLUMN].null_count():
            raise ValueError("Canton calibration requires every person's residence canton")
        inconsistent = batch.group_by("person_id").agg(pl.col(CALIBRATION_CANTON_COLUMN).n_unique())
        if (inconsistent[CALIBRATION_CANTON_COLUMN] != 1).any():
            raise ValueError("A person's trips have inconsistent residence cantons")
        for (canton,), part in batch.partition_by(CALIBRATION_CANTON_COLUMN, as_dict=True, maintain_order=True).items():
            groups[canton].append(part)
    missing = set(groups) - set(targets)
    if missing:
        raise ValueError(f"Missing mode-share targets for residence cantons: {sorted(missing)}")
    reports, choices = {}, []
    # Each canton is independent: stop converged cantons and retain the best
    # evaluated state per canton instead of compromising with a national loss.
    for canton in sorted(groups):
        logger.info("Calibrating residence canton %s", canton)
        initial = np.array([parameters.get(CANTON_ALPHA_KEYS[mode], {}).get(canton, 0.0) for mode in MODES])
        try:
            result = calibrate_batches(context, iter(groups.pop(canton)), initial, targets[canton], sampler)
        except ValueError as exc:
            raise ValueError(f"Canton {canton}: {exc}") from exc
        for mode, alpha in zip(MODES, result.alphas):
            if mode != CALIBRATION_REFERENCE_MODE:
                parameters.setdefault(CANTON_ALPHA_KEYS[mode], {})[canton] = float(alpha)
        reports[canton] = result_report(result, targets[canton])
        choices.append(result.choices)
    return pl.concat(choices).sort("person_id", "trip_index"), {
        "converged": all(report["converged"] for report in reports.values()),
        "evaluations": sum(report["evaluations"] for report in reports.values()),
        "cantons": reports,
        "skipped_target_cantons": sorted(set(targets) - set(reports)),
    }
