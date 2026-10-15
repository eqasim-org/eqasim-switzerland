"""Calibration evaluates the same public ModeChoice.run used for selection."""
from mode_choice.defaults import MODES
from mode_choice.calibration.optimizer import calibrate, share_vector


def calibrate_model(context, model, initial, targets, *, canton=None):
    def evaluate(alphas):
        choices = model.run(alphas - initial, canton=canton)
        if not model.last_counts.sum():
            raise ValueError("No trips longer than the minimum distance for ASC calibration")
        return share_vector(model.last_counts), choices

    try:
        return calibrate(evaluate, initial, targets,
                         max_evaluations=context.config("mode_choice_calibration_max_evaluations"),
                         tolerance=context.config("mode_choice_calibration_tolerance"))
    finally:
        model.clear_cache()


def result_report(result, targets):
    return {
        "converged": result.converged, "evaluations": len(result.history),
        "best_evaluation": result.best_evaluation,
        "target_shares": dict(zip(MODES, targets.tolist())),
        "final_shares": dict(zip(MODES, result.shares.tolist())),
        "history": result.history,
    }
