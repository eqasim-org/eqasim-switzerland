"""Shared evaluator for global ASCs or one residence canton's offsets."""
from contextlib import ExitStack
import numpy as np
import polars as pl

from mode_choice.defaults import MODES
from mode_choice.calibration.optimizer import calibrate, share_vector
from mode_choice.tours.prepared import PreparedBatch


def calibrate_batches(context, batches, initial, targets, sampler):
    seed = context.config("random_seed")
    # Retain preparation for every iteration, but release random reservations
    # after each calibrated population/canton, including on failures.
    with ExitStack() as stack:
        prepared = [stack.enter_context(PreparedBatch(batch, seed, sampler, assume_sorted=True)) for batch in batches]

        def evaluate(alphas):
            counts = np.zeros(len(MODES), dtype=np.int64)
            codes = []
            delta = alphas - initial
            for batch in prepared:
                selected = batch.sample_codes(delta)
                counts += batch.counts(selected)
                codes.append(selected)
            if not counts.sum():
                raise ValueError("No trips longer than the minimum distance for ASC calibration")
            # The optimizer retains only the best evaluated int8 arrays. No
            # per-iteration Polars frames, string modes, groupbys or joins.
            return share_vector(counts), codes

        result = calibrate(
            evaluate, initial, targets,
            max_evaluations=context.config("mode_choice_calibration_max_evaluations"),
            tolerance=context.config("mode_choice_calibration_tolerance"),
        )
        result.choices = pl.concat([batch.materialize(codes) for batch, codes in zip(prepared, result.choices)])
        return result


def result_report(result, targets):
    return {
        "converged": result.converged, "evaluations": len(result.history),
        "best_evaluation": result.best_evaluation,
        "target_shares": dict(zip(MODES, targets.tolist())),
        "final_shares": dict(zip(MODES, result.shares.tolist())),
        "history": result.history,
    }
