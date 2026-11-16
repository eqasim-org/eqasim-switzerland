"""Provide fixed endpoint walking-speed parameters, or estimate them from MOBIS."""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger("synpp")


# Full MOBIS fit: 791,900 filtered legs, seed 518745; metres and seconds.
ESTIMATED_PARAMETERS = {
    "v0_m_s": 1.430,
    "k": 3.159,
    "uphill_penalty": 0.870,
    "flat_grade": 0.002,
    "detour_factor": 1.311,
}


def configure(context):
    estimate = context.config("estimate_walking_model", default=False)
    if estimate:
        context.stage("calibration.walking_speed.dataset")
        context.config("random_seed")
        context.config("calibration.walking_speed.test_fraction", default=0.1)
        context.config("calibration.walking_speed.detour_factor", default=None)
        context.config("calibration.walking_speed.flat_grade", default=0.002)


def execute(context):

    # Return the stored endpoint estimates without loading or training on MOBIS.

    if not context.config("estimate_walking_model"):
        return ESTIMATED_PARAMETERS.copy()

    # Split participants so repeated legs cannot leak into held-out evaluation.

    data = (
        context.stage("calibration.walking_speed.dataset").copy().reset_index(drop=True)
    )
    fraction = context.config("calibration.walking_speed.test_fraction")
    if not 0 < fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=fraction, random_state=context.config("random_seed")
    )
    train, test = next(splitter.split(data, groups=data.participant_id))
    factor = context.config("calibration.walking_speed.detour_factor")
    if factor is None:
        factor = max(1.0, float(data.iloc[train].observed_detour.median()))
    if not np.isfinite(factor) or factor < 1:
        raise ValueError("detour_factor must be finite and >=1")
    data["model_distance_m"] = factor * data.euclidean_distance_m
    data["model_grade"] = data.elevation_change_m / data.model_distance_m
    data["effective_speed_m_s"] = data.model_distance_m / data.duration

    # Fit the flat-zone speed and the slope reductions on training participants.
    parameters, _ = fit_equation(
        data.model_grade.to_numpy()[train],
        data.effective_speed_m_s.to_numpy()[train],
        flat_grade=context.config("calibration.walking_speed.flat_grade"),
    )
    data["split"] = "train"
    data.loc[test, "split"] = "test"
    data["predicted_speed_m_s"] = speed(data.model_grade.to_numpy(), **parameters)
    data["predicted_duration_s"] = data.model_distance_m / data.predicted_speed_m_s
    scores = []
    for label in ["train", "test"]:
        subset = data.loc[data.split.eq(label)]
        scores.append(
            dict(
                split=label,
                legs=len(subset),
                **metrics(
                    subset.duration.to_numpy(), subset.predicted_duration_s.to_numpy()
                ),
            )
        )

    # Save diagnostics and the speed curve, then return only named parameters.

    result = dict(parameters, detour_factor=float(factor))
    output = Path(context.path())
    report = {
        "parameters": result,
        "equations": {
            "grade": "g=(z_destination-z_origin)/(detour_factor*euclidean_distance)",
            "speed_m_s": "v=v0_m_s*exp(-k*max(abs(g)-flat_grade,0)-uphill_penalty*max(g-flat_grade,0))",
            "time_seconds": "t=detour_factor*euclidean_distance/v",
        },
        "flat_training_legs": int(
            data.model_grade.iloc[train].abs().le(parameters["flat_grade"]).sum()
        ),
        "scores": scores,
    }
    (output / "model.json").write_text(json.dumps(report, indent=2))
    (output / "equation.txt").write_text(
        "".join(f"{name} = {value:.10g}\n" for name, value in result.items())
        + "\n".join(report["equations"].values())
        + "\n"
    )
    data.to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(scores).to_csv(output / "metrics.csv", index=False)
    plot_diagnostics(data, parameters, output)
    plot_speed_curve(result, output)
    logger.info("Estimated endpoint walking-speed parameters: %s", result)
    return result


def speed(grade, v0_m_s, k, uphill_penalty=0.0, flat_grade=0.002):
    """Constant speed in the flat zone, with continuous reductions beyond it."""
    if not (
        np.isfinite([v0_m_s, k, uphill_penalty, flat_grade]).all()
        and v0_m_s > 0
        and k >= 0
        and uphill_penalty >= 0
        and flat_grade >= 0
    ):
        raise ValueError("Invalid walking-speed parameters")
    grade = np.asarray(grade, dtype=float)
    return v0_m_s * np.exp(
        -k * np.maximum(np.abs(grade) - flat_grade, 0.0)
        - uphill_penalty * np.maximum(grade - flat_grade, 0.0)
    )


def travel_time(
    origin,
    destination,
    *,
    detour_factor,
    v0_m_s,
    k,
    uphill_penalty=0.0,
    flat_grade=0.002,
):
    """Vectorized xyz evaluator; seconds, XY horizontal metres and signed grade."""
    if not (
        np.isfinite([detour_factor, v0_m_s, k, flat_grade, uphill_penalty]).all()
        and detour_factor >= 1
        and v0_m_s > 0
        and k >= 0
        and uphill_penalty >= 0
        and flat_grade >= 0
    ):
        raise ValueError("Invalid walking-speed parameters")
    origin, destination = np.asarray(origin, dtype=float), np.asarray(
        destination, dtype=float
    )
    if (
        origin.shape[-1] != 3
        or destination.shape[-1] != 3
        or not np.isfinite(origin).all()
        or not np.isfinite(destination).all()
    ):
        raise ValueError("Expected finite xyz coordinates")
    delta = destination - origin
    length = detour_factor * np.hypot(delta[..., 0], delta[..., 1])
    if np.any((length == 0) & (delta[..., 2] != 0)):
        raise ValueError(
            "Endpoint-grade equation cannot represent purely vertical walking"
        )
    grade = np.divide(
        delta[..., 2], length, out=np.zeros_like(length), where=length > 0
    )
    return length / speed(
        grade, v0_m_s, k, uphill_penalty=uphill_penalty, flat_grade=flat_grade
    )


def fit_equation(grade, observed_speed, *, flat_grade=0.002):
    grade, observed_speed = np.asarray(grade, dtype=float), np.asarray(
        observed_speed, dtype=float
    )
    if grade.ndim != 1 or observed_speed.shape != grade.shape or not len(grade):
        raise ValueError("Expected one grade and speed per walking leg")
    if (
        not np.isfinite(grade).all()
        or not np.isfinite(observed_speed).all()
        or np.any(observed_speed <= 0)
    ):
        raise ValueError("Fitting requires finite grades and positive speeds")
    if not np.isfinite(flat_grade) or flat_grade < 0:
        raise ValueError("flat_grade must be a finite nonnegative rise/run ratio")
    flat = np.abs(grade) <= flat_grade
    if not flat.any():
        raise ValueError("No training legs in the flat zone to estimate average speed")
    if not np.any(grade > flat_grade) or not np.any(grade < -flat_grade):
        raise ValueError(
            "Need training legs beyond both flat-zone boundaries to fit slope reductions"
        )
    v0 = float(observed_speed[flat].mean())
    outside = ~flat
    excess = np.maximum(np.abs(grade[outside]) - flat_grade, 0.0)
    uphill = np.maximum(grade[outside] - flat_grade, 0.0)

    def residual(p):
        return (
            np.log(v0) - p[0] * excess - p[1] * uphill - np.log(observed_speed[outside])
        )

    fit = least_squares(
        residual,
        [3.0, 1.0],
        bounds=([0.0, 0.0], [30.0, 30.0]),
        loss="soft_l1",
        f_scale=0.2,
    )
    if not fit.success:
        raise RuntimeError("Walking-speed fit did not converge: " + fit.message)
    return (
        dict(
            v0_m_s=v0,
            k=float(fit.x[0]),
            uphill_penalty=float(fit.x[1]),
            flat_grade=float(flat_grade),
        ),
        fit,
    )


def metrics(observed, predicted):
    error = predicted - observed
    denominator = np.sum((observed - observed.mean()) ** 2)
    return dict(
        mae_s=float(np.mean(abs(error))),
        rmse_s=float(np.sqrt(np.mean(error**2))),
        median_absolute_percentage_error=float(np.median(abs(error) / observed) * 100),
        r2=float(1 - np.sum(error**2) / denominator) if denominator > 0 else None,
    )


def plot_diagnostics(data, parameters, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    held = data.loc[data.split.eq("test")]
    axes[0].hexbin(
        held.model_grade,
        held.effective_speed_m_s,
        gridsize=40,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    grades = np.linspace(data.model_grade.min(), data.model_grade.max(), 300)
    axes[0].plot(
        grades,
        speed(grades, **parameters),
        color="darkorange",
        label="Selected local speed equation",
    )
    axes[0].set(
        xlabel="Signed endpoint grade (rise/run)",
        ylabel="Whole-leg effective walking speed (m/s)",
    )
    axes[0].legend()
    axes[1].hexbin(
        held.duration / 60,
        held.predicted_duration_s / 60,
        gridsize=40,
        mincnt=1,
        bins="log",
        cmap="Blues",
    )
    limit = max(held.duration.max(), held.predicted_duration_s.max()) / 60
    axes[1].plot([0, limit], [0, limit], color="gray", ls="--")
    axes[1].set(
        xlabel="Observed time (min)",
        ylabel="Predicted time (min)",
        title="Held-out participants",
    )
    fig.savefig(output / "validation.png", dpi=170)
    plt.close(fig)


def plot_speed_curve(parameters, output):
    """Plot endpoint walking speed for signed grades from -30% to +30%."""
    coefficients = {
        name: parameters[name]
        for name in ("v0_m_s", "k", "uphill_penalty", "flat_grade")
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for axis, extent, title in zip(
        axes,
        [0.3, max(0.02, parameters["flat_grade"] * 2)],
        ["Endpoint walking speed", "Near-flat detail"],
    ):
        grades = np.unique(
            np.r_[
                np.linspace(-extent, extent, 1001),
                -parameters["flat_grade"],
                parameters["flat_grade"],
            ]
        )
        axis.plot(grades, speed(grades, **coefficients), color="#4777b3", linewidth=2)
        axis.axvspan(
            -parameters["flat_grade"],
            parameters["flat_grade"],
            alpha=0.2,
            color="green",
            label="Flat zone",
        )
        axis.legend()
        axis.set(
            xlabel="Slope g (rise/run; 0.01 = 1%)",
            ylabel="Walking speed (m/s)",
            title=title,
        )
        axis.grid(alpha=0.25)
    fig.savefig(Path(output) / "speed_vs_slope.png", dpi=170)
    plt.close(fig)
