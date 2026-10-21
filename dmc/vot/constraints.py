"""Constrain car and combined-PT marginal WTP during JAX estimation."""
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dmc.constants import constants
from dmc.parameters import NAMES_CONVERSION

logger = logging.getLogger(__name__)


def create_vot_function(context, df, parameters, modes, free_parameters):
    """Return differentiable observed-trip means, matching the VOT reporter.

    These are conditional means over the observed car/PT samples, with optional
    survey weights. PT saves one minute proportionally across physical durations.
    """
    from dmc.models.model_jax import _safe_power

    samples = {}
    for mode in ("car", "pt"):
        sample = df.loc[(df["mode"] == modes.index(mode)) & (df[f"{mode}_availability"] > 0)]
        if sample.empty:
            raise ValueError(f"VOT constraints require observed available {mode} trips")
        samples[mode] = sample
    car = samples["car"]
    pt = samples["pt"]
    car_time = (car.car_travel_time_min + car.parking_searching_duration_min).to_numpy(float)
    if not np.isfinite(car_time).all() or (car_time <= 0).any():
        raise ValueError("VOT constraints require positive finite durations for observed car trips")
    durations = np.maximum(pt[["pt_in_vehicle_time_min", "pt_access_egress_time_min",
                               "pt_transfer_time_min"]].to_numpy(float), 0.)
    if not np.isfinite(durations).all() or (durations.sum(axis=1) <= 0).any():
        raise ValueError("VOT constraints require positive finite total durations for observed PT trips")
    weighted = context.config("use_weights_for_vot")
    weights = {}
    cost_features = {}
    distance_interaction = context.config("distance_cost_interaction")
    income_interaction = context.config("income_cost_interaction") and context.config("use_income_in_dmc")
    for mode, sample in samples.items():
        weight = sample.person_weight.to_numpy(float) if weighted else np.ones(len(sample))
        if not np.isfinite(weight).all() or (weight < 0).any() or weight.sum() <= 0:
            raise ValueError(f"Invalid weights for {mode} VOT constraints")
        weights[mode] = jnp.asarray(weight / weight.sum())
        features = {}
        for enabled, column, reference in (
            (distance_interaction, "euclidean_distance_km", "reference_euclidean_distance_km"),
            (income_interaction, "income", "reference_income_chf"),
        ):
            if enabled:
                values = sample[column].to_numpy(float) / context.config(reference)
                if not np.isfinite(values).all() or (values <= 0).any():
                    raise ValueError(f"VOT constraints require positive finite {column} for {mode}")
                features[column] = jnp.asarray(values)
        cost_features[mode] = features
    free_index = {parameter.name: i for i, parameter in enumerate(free_parameters)}
    scale = constants.TIME_SCALE_MIN
    car_time = jnp.asarray(car_time / scale)
    durations = jnp.asarray(durations)

    def means(x):
        beta = {name: x[free_index[name]] if name in free_index else parameter.init_value
                for name, parameter in parameters.items()}

        def cost_derivative(mode):
            cost = beta["beta_cost_CHF"]
            features = cost_features[mode]
            if distance_interaction:
                cost = cost * _safe_power(features["euclidean_distance_km"], beta["lambda_cost_distance"])
            if income_interaction:
                cost = cost * _safe_power(features["income"], beta["lambda_cost_income"])
            return cost

        car_vot = (60 * beta["beta_car_travel_time_min"] * beta["lambda_car_travel_time"]
                   * _safe_power(car_time, beta["lambda_car_travel_time"] - 1)
                   / scale / cost_derivative("car"))
        pt_marginal = (
            beta["beta_pt_in_vehicle_time_min"] * beta["lambda_pt_in_vehicle_time"]
            * _safe_power(durations[:, 0] / scale, beta["lambda_pt_in_vehicle_time"])
            + beta["beta_pt_access_egress_time_min"] * beta["lambda_pt_access_egress_time"]
            * _safe_power(durations[:, 1] / scale, beta["lambda_pt_access_egress_time"])
            + beta["beta_pt_transfer_time_min"] * beta["lambda_pt_transfer_time"]
            * _safe_power(durations[:, 2] / scale, beta["lambda_pt_transfer_time"])
        ) / durations.sum(axis=1)
        pt_vot = 60 * pt_marginal / cost_derivative("pt")
        return jnp.array([jnp.sum(weights["car"] * car_vot),
                          jnp.sum(weights["pt"] * pt_vot)])

    return jax.jit(means)


def constrain(context, df, parameters, modes, free_parameters, objective, initial):
    """Maximize the same likelihood with two equality constraints using SLSQP."""
    from dmc.models.model_jax import projected_gradient_norm

    targets = np.array([context.config("dmc_vot_car_target"), context.config("dmc_vot_pt_target")], float)
    tolerance = context.config("dmc_vot_constraint_tolerance")
    if not np.isfinite(targets).all() or (targets <= 0).any() or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("VOT targets and constraint tolerance must be finite and positive")
    remaining = context.config("dmc_jax_max_iterations") - initial.nit
    if remaining <= 0:
        raise RuntimeError("No iterations remain for VOT constraints; increase dmc_jax_max_iterations")
    means = create_vot_function(context, df, parameters, modes, free_parameters)
    residual = jax.jit(lambda x: means(x) / jnp.asarray(targets) - 1.)
    derivative = jax.jit(jax.jacrev(residual))
    # A mean likelihood keeps SLSQP's objective and the normalized constraints
    # numerically comparable. The reported likelihood is restored to a sum.
    weight_sum = float(df.person_weight.sum())

    def scaled_objective(x):
        value, gradient = objective(x)
        return value / weight_sum, gradient / weight_sum

    bounds = [(p.lower_bound, p.upper_bound) for p in free_parameters]
    logger.info("Constraining observed-trip VOT means to car %.2f, combined PT %.2f CHF/hour", *targets)
    optimization = minimize(
        scaled_objective, initial.x, method="SLSQP", jac=True, bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda x: np.asarray(residual(x)),
                      "jac": lambda x: np.asarray(derivative(x))}],
        options={"maxiter": remaining, "ftol": 1e-12},
    )
    achieved = np.asarray(means(optimization.x), dtype=float)
    errors = achieved - targets
    # Constrained stationarity concerns the Lagrangian, not the likelihood's
    # raw gradient, which is generally nonzero at the constrained optimum.
    jacobian = np.asarray(derivative(optimization.x))
    multipliers = getattr(optimization, "multipliers", None)
    if multipliers is None:
        raise RuntimeError("This SciPy SLSQP version must expose equality multipliers for convergence checks")
    final_value, final_gradient = objective(optimization.x)
    lagrangian_gradient = final_gradient / weight_sum - jacobian.T @ np.asarray(multipliers[:2])
    norm = projected_gradient_norm(optimization.x, lagrangian_gradient, bounds)
    gradient_tolerance = context.config("dmc_jax_gradient_tolerance")
    if (not optimization.success or not np.isfinite(achieved).all()
            or np.max(np.abs(errors)) > tolerance or not np.isfinite(norm) or norm > gradient_tolerance):
        raise RuntimeError(f"VOT-constrained estimation did not converge: {optimization.message}; "
                           f"achieved means {achieved.tolist()}, targets {targets.tolist()}, "
                           f"projected Lagrangian gradient {norm:.3g}")
    optimization.fun, optimization.jac = final_value, final_gradient
    optimization.projected_gradient_norm = norm
    optimization.gradient_converged = True
    optimization.gradient_definition = "projected Lagrangian gradient of mean negative log likelihood"
    optimization.unconstrained = initial
    optimization.parameter_comparison = parameter_comparison(parameters, free_parameters, initial, optimization)
    optimization.vot_constraints = {
        "car_target_chf_h": float(targets[0]), "pt_combined_target_chf_h": float(targets[1]),
        "car_achieved_chf_h": float(achieved[0]), "pt_combined_achieved_chf_h": float(achieved[1]),
        "absolute_tolerance_chf_h": float(tolerance),
        "survey_weighted": bool(context.config("use_weights_for_vot")),
        "unconstrained_log_likelihood": float(-initial.fun),
        "log_likelihood_loss": float(optimization.fun - initial.fun),
    }
    optimization.refinement_history = [*initial.refinement_history, {
        "method": "SLSQP with VOT constraints", "iterations": int(optimization.nit),
        "log_likelihood": float(-optimization.fun), "projected_gradient_norm": norm,
        "gradient_definition": optimization.gradient_definition,
        "message": str(optimization.message),
    }]
    optimization.nit += initial.nit
    optimization.nfev += initial.nfev
    return optimization


def parameter_comparison(parameters, free_parameters, unconstrained, constrained):
    """All coefficients in estimation and exported names, including fixed ones."""
    baseline = {name: p.init_value for name, p in parameters.items()}
    fitted = baseline.copy()
    baseline.update({p.name: float(value) for p, value in zip(free_parameters, unconstrained.x, strict=True)})
    fitted.update({p.name: float(value) for p, value in zip(free_parameters, constrained.x, strict=True)})
    rows = []
    for name, p in parameters.items():
        before, after = baseline[name], fitted[name]
        rows.append({"parameter": name, "exported_parameter": NAMES_CONVERSION[name],
                     "unconstrained": before, "constrained": after, "change": after - before,
                     "change_over_abs_baseline_pct": 100 * (after - before) / abs(before)
                     if abs(before) > 1e-10 else np.nan, "fixed": p.fixed})
    return pd.DataFrame(rows)
