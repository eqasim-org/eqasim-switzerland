"""Estimate the DMC multinomial-logit model with JAX and SciPy.

This is a Biogeme-independent implementation of `dmc.models.model_biogeme`.  It keeps
the same data preparation, utility specification, fixed parameters, bounds,
weights, MATSim YAML output and value-of-time calculations.  JAX compiles the
likelihood and its exact gradient; SciPy's L-BFGS-B handles the bounds.

Run this Synpp stage as ``dmc.models.model_jax``.
"""

import logging
import os
import time

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
import yaml

from dmc.constants import constants
from dmc.models.declarations import Parameter, define_parameters
from dmc.models.data import preprocess_data
from dmc.models.utility import build as build_shared_utilities
from dmc.models.validation import validate
from dmc.vot.functions import vot_utils
from dmc.parameters import writer


logger = logging.getLogger(__name__)

MODES = ["car", "pt", "bike", "walk", "car_passenger"]
TIME_SCALE_MIN = constants.TIME_SCALE_MIN
DISTANCE_SCALE_KM = constants.DISTANCE_SCALE_KM
PT_REGIONAL_RADIUS_KM = constants.PT_REGIONAL_RADIUS_KM
AGE_SCALE_YEAR = constants.AGE_SCALE_YEAR


def configure(context):
    context.stage("dmc.data.training_data")
    context.stage("data.constants")
    context.config("ignore_car_passenger", False)
    context.config("distance_cost_interaction", True)
    context.config("income_cost_interaction", True)
    context.config("use_exponents", True)
    context.config("use_income_in_dmc", True)
    context.config("use_weights_for_vot", False)
    context.config("dmc_transfer_peak_km", 7.0)
    context.config("dmc_jax_max_iterations", 10000)
    context.config("dmc_jax_gradient_tolerance", 1e-05)
    context.config("dmc_jax_function_tolerance", 1e-15)
    context.config("dmc_jax_refinement_passes", 2)
    context.config("dmc_jax_newton_steps", 2)
    context.config("dmc_vot_constraints", True)
    context.config("dmc_vot_car_target", 32.0)
    context.config("dmc_vot_pt_target", 18.0)
    context.config("dmc_vot_constraint_tolerance", 0.0001)

    # These are consumed by the existing MATSim parameter writer.
    context.config("urbancore_parking_search_min", constants.URBANCORE_PARKING_SEARCH_MIN)
    context.config("urban_parking_search_min", constants.URBAN_PARKING_SEARCH_MIN)
    context.config("suburban_parking_search_min", constants.SUBURBAN_PARKING_SEARCH_MIN)
    context.config("car_cost_per_km", constants.CAR_COST_PER_KM)
    context.config("parking_cost_per_hour_CHF_urban", constants.PARKING_COST_PER_HOUR_CHF_URBAN)
    context.config("parking_cost_per_hour_CHF_urbancore", constants.PARKING_COST_PER_HOUR_CHF_URBANCORE)
    context.config("parking_cost_per_hour_CHF_suburban", constants.PARKING_COST_PER_HOUR_CHF_SUBURBAN)
    context.config("parking_price_reduction_for_work", constants.PARKING_PRICE_REDUCTION_FOR_WORK)
    context.config("car_cost_model", constants.CAR_COST_MODEL)
    context.config("reference_euclidean_distance_km", constants.REF_EUCLIDEAN_DISTANCE_KM)
    context.config("reference_income_chf", constants.REF_INCOME_CHF)
    context.config("pt_regional_radius_km", constants.PT_REGIONAL_RADIUS_KM)


def execute(context):

    # Prepare conditional histories and the common model covariates.

    df = context.stage("dmc.data.training_data")
    ignore_car_passenger = context.config("ignore_car_passenger")
    df, modes = preprocess_data(df, ignore_car_passenger)
    log_trip_stats(df, modes)

    # Estimate the shared specification with the configured VOT assumptions.

    parameters = define_parameters(
        ignore_car_passenger,
        context.config("use_exponents"),
        context.config("use_income_in_dmc"),
        income_reference=float(np.average(df.income, weights=df.person_weight)),
        transfer_peak_km=context.config("dmc_transfer_peak_km"),
        distance_interaction=context.config("distance_cost_interaction"),
        income_interaction=context.config("income_cost_interaction"),
    )
    result = estimate(context, df, parameters, modes, ignore_car_passenger)
    logger.info(result.short_summary())
    if not result.optimization.success or not result.optimization.gradient_converged:
        logger.warning("Optimizer stopped before satisfying all convergence checks; "
                       "projected gradient %.3g (target %.3g). Outputs contain its best solution.",
                       result.optimization.projected_gradient_norm,
                       context.config("dmc_jax_gradient_tolerance"))

    # Export the exact fitted utility parameters.

    mode_params_path, cost_params_path = writer(context, result, parameters).write()
    free_parameters = [parameter for parameter in parameters.values() if not parameter.fixed]
    csv_path, report_path = _write_estimation_outputs(context, result, free_parameters)
    logger.info("Estimated parameters saved to %s and %s", csv_path, report_path)

    # Report time valuations under the fitted utility.

    vot_car, mean_vot_car = vot_utils.get_car_vot(context, df, result, MODES, parameters)
    vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer = vot_utils.get_pt_vot(
        context, df, result, MODES, parameters
    )
    logger.info("Mean marginal WTP for observed car trips: %.2f CHF/hour", mean_vot_car)
    logger.info("Mean marginal WTP for observed PT trips: %.2f CHF/hour", mean_vot_pt)
    logger.info("PT component WTP (CHF/hour): in-vehicle %.2f, access/egress %.2f, waiting %.2f",
                *[vot_utils._mean(values, df["person_weight"], context.config("use_weights_for_vot"))
                  for values in (vot_in_vehicle, vot_access_egress, vot_transfer)])

    figure_path = os.path.join(context.path(), "vot_distribution.png")
    use_vot_weights = context.config("use_weights_for_vot")
    logger.info(
        "WTP summaries and histogram use %s trip observations",
        "survey-weighted" if use_vot_weights else "unweighted",
    )
    vot_utils.plot_vot(
        vot_car,
        vot_pt,
        figure_path=figure_path,
        car_weights=df.loc[vot_car.index, "person_weight"] if use_vot_weights else None,
        pt_weights=df.loc[vot_pt.index, "person_weight"] if use_vot_weights else None,
    )
    # Return the existing estimation-stage contract.

    return (
        result,
        df,
        (mode_params_path, cost_params_path),
        figure_path,
        (vot_car, vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer),
    )


class EstimationResult:
    """Small result API shared by ``writer`` and ``vot_utils``."""

    def __init__(
        self,
        beta_values: dict[str, float],
        optimization: OptimizeResult,
        elapsed_seconds: float,
    ):
        self._beta_values = beta_values
        self.optimization = optimization
        self.elapsed_seconds = elapsed_seconds

    def get_beta_values(self) -> dict[str, float]:
        return self._beta_values.copy()

    def short_summary(self) -> str:
        return (
            "JAX/SciPy DMC estimation: "
            f"success={self.optimization.success}, "
            f"log_likelihood={-self.optimization.fun:.6f}, "
            f"iterations={self.optimization.nit}, "
            f"evaluations={self.optimization.nfev}, "
            f"projected_gradient={self.optimization.projected_gradient_norm:.3g}, "
            f"gradient_converged={self.optimization.gradient_converged}, "
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"message={self.optimization.message}"
        )


def dataframe_to_arrays(df: pd.DataFrame) -> dict[str, jax.Array]:
    """Copy the numeric training frame into contiguous JAX float64 arrays."""
    arrays = {
        column: jnp.asarray(df[column].to_numpy(dtype=np.float64, copy=True))
        for column in df.columns
        if column != "mode"
    }
    arrays["mode"] = jnp.asarray(df["mode"].to_numpy(dtype=np.int32, copy=True))
    return arrays


def _safe_power(base, exponent):
    """Power for non-negative features with finite exponent gradients at zero."""
    positive = base > 0
    safe_base = jnp.where(positive, base, 1.0)
    return jnp.where(positive, jnp.exp(exponent * jnp.log(safe_base)), 0.0)


def build_utilities(context, arrays, beta, modes, ignore_car_passenger):
    by_mode = build_shared_utilities(context, arrays, beta, modes,
                                     power=_safe_power, maximum=jnp.maximum, exp=jnp.exp)
    return (jnp.stack([by_mode[mode] for mode in modes], axis=1),
            jnp.stack([arrays[f"{mode}_availability"] for mode in modes], axis=1))


def create_objective(context, df, parameters, modes, ignore_car_passenger):
    """Return a compiled negative log-likelihood/gradient and optimizer inputs."""
    arrays = dataframe_to_arrays(df)
    free_parameters = [parameter for parameter in parameters.values() if not parameter.fixed]
    free_index = {parameter.name: index for index, parameter in enumerate(free_parameters)}

    def negative_log_likelihood(x):
        beta = {
            name: (x[free_index[name]] if name in free_index else parameter.init_value)
            for name, parameter in parameters.items()
        }
        utilities, availability = build_utilities(context, arrays, beta, modes, ignore_car_passenger)
        masked_utilities = jnp.where(availability > 0, utilities, -jnp.inf)
        chosen_utility = jnp.take_along_axis(masked_utilities, arrays["mode"][:, None], axis=1)[:, 0]
        log_probability = chosen_utility - logsumexp(masked_utilities, axis=1)
        return -jnp.sum(arrays["person_weight"] * log_probability)

    value_and_gradient = jax.jit(jax.value_and_grad(negative_log_likelihood))

    def scipy_objective(x):
        value, gradient = value_and_gradient(jnp.asarray(x))
        return float(value), np.asarray(gradient, dtype=np.float64)

    x0 = np.asarray([parameter.init_value for parameter in free_parameters], dtype=np.float64)
    bounds = [(parameter.lower_bound, parameter.upper_bound) for parameter in free_parameters]
    lower, upper = _bound_arrays(bounds)
    x0 = np.clip(x0, lower, upper)
    return scipy_objective, x0, bounds, free_parameters


def estimate(context, df, parameters, modes, ignore_car_passenger) -> EstimationResult:
    objective, x0, bounds, free_parameters = create_objective(
        context, df, parameters, modes, ignore_car_passenger
    )
    logger.info("Compiling JAX likelihood and gradient for %d observations", len(df))
    compile_started = time.perf_counter()
    initial_value, initial_gradient = objective(x0)
    logger.info(
        "Initial log likelihood %.6f; compilation/evaluation took %.2fs; gradient norm %.3g",
        -initial_value,
        time.perf_counter() - compile_started,
        np.linalg.norm(initial_gradient),
    )

    iteration = 0

    def callback(intermediate_result):
        nonlocal iteration
        iteration += 1
        if iteration == 1 or iteration % 50 == 0:
            logger.info("Iteration %d: log likelihood %.6f", iteration, -intermediate_result.fun)

    started = time.perf_counter()
    if not np.isfinite(initial_value) or not np.isfinite(initial_gradient).all():
        raise ValueError("Initial likelihood or gradient is not finite; check the training inputs")
    optimization = optimize(context, objective, x0, bounds, callback)
    if context.config("dmc_vot_constraints"):
        from dmc.vot.constraints import constrain
        optimization = constrain(context, df, parameters, modes, free_parameters, objective, optimization)
    elapsed = time.perf_counter() - started
    beta_values = {
        parameter.name: float(value)
        for parameter, value in zip(free_parameters, optimization.x, strict=True)
    }
    return EstimationResult(beta_values, optimization, elapsed)


def _bound_arrays(bounds):
    lower = np.asarray([-np.inf if low is None else low for low, _ in bounds])
    upper = np.asarray([np.inf if high is None else high for _, high in bounds])
    return lower, upper


def projected_gradient_norm(x, gradient, bounds):
    """Infinity norm of the gradient projected onto the feasible box."""
    lower, upper = _bound_arrays(bounds)
    projected = x - np.clip(x - gradient, lower, upper)
    return float(np.max(np.abs(projected)))


def optimize(context, objective, x0, bounds, callback=None):
    """Refine a likelihood plateau with fresh L-BFGS curvature history.

    Near a plateau, a bounded Newton step can finish gradient convergence.
    All passes share the iteration budget. Report gradient convergence separately
    from SciPy's relative-function-change stopping criterion.
    """
    max_iterations = context.config("dmc_jax_max_iterations")
    tolerance = context.config("dmc_jax_gradient_tolerance")
    function_tolerance = context.config("dmc_jax_function_tolerance")
    refinements = context.config("dmc_jax_refinement_passes")
    newton_steps = context.config("dmc_jax_newton_steps")
    if (max_iterations < 1 or tolerance <= 0 or function_tolerance < 0
            or refinements < 0 or newton_steps < 0):
        raise ValueError("Invalid JAX optimizer iteration/tolerance/refinement settings")
    best = None
    total_iterations = total_evaluations = 0
    history = []
    for attempt in range(refinements + 1):
        remaining = max_iterations - total_iterations
        if remaining <= 0:
            break
        candidate = minimize(
            objective, x0 if best is None else best.x,
            method="L-BFGS-B", jac=True, bounds=bounds, callback=callback,
            options={"maxiter": remaining, "gtol": tolerance,
                     "ftol": function_tolerance if attempt == 0 else 0.0,
                     "maxls": 100, "maxcor": 100},
        )
        total_iterations += candidate.nit
        total_evaluations += candidate.nfev
        norm = projected_gradient_norm(candidate.x, candidate.jac, bounds)
        history.append({"pass": attempt + 1, "method": "L-BFGS-B",
                        "log_likelihood": float(-candidate.fun),
                        "projected_gradient_norm": norm,
                        "iterations": int(candidate.nit), "message": str(candidate.message)})
        logger.info("Optimizer pass %d: log likelihood %.9f; projected gradient %.3g",
                    attempt + 1, -candidate.fun, norm)
        if not np.isfinite(candidate.fun) or not np.isfinite(candidate.jac).all():
            if best is None:
                raise ValueError("Optimizer produced a non-finite likelihood or gradient")
            break
        improved = (best is None or candidate.fun < best.fun or
                    (abs(candidate.fun - best.fun) <= 1e-10 and
                     norm < projected_gradient_norm(best.x, best.jac, bounds)))
        if improved:
            best = candidate
        if norm <= tolerance or not improved:
            break
    remaining = max_iterations - total_iterations
    norm = projected_gradient_norm(best.x, best.jac, bounds)
    if tolerance < norm < 1.0 and remaining > 0 and newton_steps:
        best, steps, evaluations, polish_history = newton_polish(
            objective, best, bounds, tolerance, min(newton_steps, remaining))
        total_iterations += steps
        total_evaluations += evaluations
        history.extend(polish_history)
    best.nit = total_iterations
    best.nfev = total_evaluations
    best.projected_gradient_norm = projected_gradient_norm(best.x, best.jac, bounds)
    best.gradient_converged = best.projected_gradient_norm <= tolerance
    best.refinement_history = history
    if best.gradient_converged:
        best.success = True
        best.message = "CONVERGENCE: PROJECTED GRADIENT IS WITHIN TOLERANCE"
    return best


def newton_polish(objective, result, bounds, tolerance, max_steps):
    """Polish a near-stationary point using differences of exact JAX gradients.

    Only the interior/free part of the Hessian is solved. Skip singular or
    indefinite curvature; accept a step only when the projected gradient falls
    and likelihood does not worsen beyond floating-point rounding.
    """
    lower, upper = _bound_arrays(bounds)
    history = []
    evaluations = steps = 0
    for _ in range(max_steps):
        x, gradient = result.x, result.jac
        norm = projected_gradient_norm(x, gradient, bounds)
        if norm <= tolerance:
            break
        active = (((x <= lower + 1e-8) & (gradient > 0)) |
                  ((x >= upper - 1e-8) & (gradient < 0)) | (lower == upper))
        indices = np.flatnonzero(~active)
        if not len(indices):
            break
        increments = np.cbrt(np.finfo(float).eps) * np.maximum(1., np.abs(x))
        columns = []
        for i in indices:
            plus, minus = x.copy(), x.copy()
            plus[i] = min(x[i] + increments[i], upper[i])
            minus[i] = max(x[i] - increments[i], lower[i])
            columns.append((objective(plus)[1][indices] - objective(minus)[1][indices])
                           / (plus[i] - minus[i]))
            evaluations += 2
        hessian = np.column_stack(columns)
        hessian = (hessian + hessian.T) / 2
        if not np.isfinite(hessian).all():
            break
        try:
            if np.linalg.eigvalsh(hessian)[0] <= 1e-10:
                break
            direction = np.zeros_like(x)
            direction[indices] = np.linalg.solve(hessian, -gradient[indices])
        except np.linalg.LinAlgError:
            break
        rounding = 10 * np.finfo(float).eps * max(1., abs(result.fun))
        accepted = False
        for factor in (1., .5, .25, .125, .0625):
            candidate_x = np.clip(x + factor * direction, lower, upper)
            value, candidate_gradient = objective(candidate_x)
            evaluations += 1
            candidate_norm = projected_gradient_norm(candidate_x, candidate_gradient, bounds)
            if (np.isfinite(value) and np.isfinite(candidate_gradient).all()
                    and value <= result.fun + rounding and candidate_norm < norm):
                result = OptimizeResult(result)
                result.x, result.fun, result.jac = candidate_x, value, candidate_gradient
                steps += 1
                history.append({"method": "Newton", "log_likelihood": float(-value),
                                "projected_gradient_norm": candidate_norm, "iterations": 1})
                logger.info("Newton refinement: log likelihood %.9f; projected gradient %.3g",
                            -value, candidate_norm)
                accepted = True
                break
        if not accepted:
            break
    return result, steps, evaluations, history


def log_trip_stats(df, modes):
    for mode in modes:
        selected = df["mode"] == modes.index(mode)
        logger.info(
            "%s: number of trips is %d and average distance is %.2f km",
            mode, selected.sum(), df.loc[selected, "euclidean_distance_km"].mean(),
        )


def _write_estimation_outputs(context, result, free_parameters):
    csv_path = os.path.join(context.path(), "dmc_model_parameters.csv")
    pd.DataFrame(
        {"Value": [result.get_beta_values()[parameter.name] for parameter in free_parameters]},
        index=pd.Index([parameter.name for parameter in free_parameters], name="Name"),
    ).to_csv(csv_path)

    report_path = os.path.join(context.path(), "DMC_model_jax.yaml")
    report = {
        "estimator": "JAX/SciPy L-BFGS-B with bounded Newton refinement",
        "success": bool(result.optimization.success),
        "message": str(result.optimization.message),
        "log_likelihood": float(-result.optimization.fun),
        "iterations": int(result.optimization.nit),
        "function_evaluations": int(result.optimization.nfev),
        "elapsed_seconds": float(result.elapsed_seconds),
        "projected_gradient_norm": float(result.optimization.projected_gradient_norm),
        "gradient_tolerance": context.config("dmc_jax_gradient_tolerance"),
        "gradient_definition": getattr(result.optimization, "gradient_definition",
                                       "projected gradient of summed negative log likelihood"),
        "gradient_converged": bool(result.optimization.gradient_converged),
        "optimizer_passes": result.optimization.refinement_history,
        "estimated_parameters": result.get_beta_values(),
    }
    if hasattr(result.optimization, "vot_constraints"):
        report["estimator"] = "JAX/SciPy constrained maximum likelihood (SLSQP)"
        report["vot_constraints"] = result.optimization.vot_constraints
        result.optimization.parameter_comparison.to_csv(
            os.path.join(context.path(), "dmc_parameter_comparison.csv"), index=False)
    with open(report_path, "w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    return csv_path, report_path
