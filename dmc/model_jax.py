"""Estimate the DMC multinomial-logit model with JAX and SciPy.

This is a Biogeme-independent implementation of `dmc.model`.  It keeps
the same data preparation, utility specification, fixed parameters, bounds,
weights, MATSim YAML output and value-of-time calculations.  JAX compiles the
likelihood and its exact gradient; SciPy's L-BFGS-B handles the bounds.

Run this Synpp stage as ``dmc.model_jax``.
"""

from dataclasses import dataclass
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
from dmc.vot.functions import vot_utils
from dmc.writer import writer


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
    context.config("dmc_jax_max_iterations", 2000)
    context.config("dmc_jax_gradient_tolerance", 1e-5)

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
    df = context.stage("dmc.data.training_data")
    ignore_car_passenger = context.config("ignore_car_passenger")
    df, modes = preprocess_data(df, ignore_car_passenger)
    log_trip_stats(df, modes)

    parameters = define_parameters(
        ignore_car_passenger,
        context.config("use_exponents"),
        context.config("use_income_in_dmc"),
    )
    result = estimate(context, df, parameters, modes, ignore_car_passenger)
    logger.info(result.short_summary())
    if not result.optimization.success:
        logger.warning("Optimizer stopped without declaring convergence; outputs contain its best solution")

    mode_params_path, cost_params_path = writer(context, result, parameters).write()
    free_parameters = [parameter for parameter in parameters.values() if not parameter.fixed]
    csv_path, report_path = _write_estimation_outputs(context, result, free_parameters)
    logger.info("Estimated parameters saved to %s and %s", csv_path, report_path)

    vot_car, mean_vot_car = vot_utils.get_car_vot(context, df, result, MODES)
    vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer = vot_utils.get_pt_vot(
        context, df, result, MODES
    )
    logger.info("Mean marginal WTP for observed car trips: %.2f CHF/hour", mean_vot_car)
    logger.info("Mean marginal WTP for observed PT trips: %.2f CHF/hour", mean_vot_pt)

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
    return (
        result,
        df,
        (mode_params_path, cost_params_path),
        figure_path,
        (vot_car, vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer),
    )




####################### Functions ########################

@dataclass(frozen=True)
class Parameter:
    """Minimal, library-neutral equivalent of a Biogeme Beta declaration."""

    name: str
    init_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    fixed: bool = False


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
            f"elapsed={self.elapsed_seconds:.1f}s, "
            f"message={self.optimization.message}"
        )


def preprocess_data(df: pd.DataFrame, ignore_car_passenger: bool):
    """Apply exactly the coding and weighting used by ``dmc.model``."""
    modes = MODES.copy()
    if ignore_car_passenger:
        df = df[df["mode"] != "car_passenger"].reset_index(drop=True)
        df = df[[column for column in df.columns if "car_passenger" not in column]]
        modes.remove("car_passenger")

    df = df[df["mode"].isin(modes)].copy()
    for mode in modes:
        df[f"{mode}_availability"] = df[f"{mode}_availability"].astype(int)

    integer_columns = [
        "driving_license", "destination_work", "origin_home", "destination_home",
        "destination_education", "destination_shopping", "destination_leisure",
        "destination_other", "good_pt_service", "medium_pt_service",
        "destination_good_pt_service", "destination_medium_pt_service",
        "low_income", "high_income", "has_car", "destination_zurich",
        "destination_geneva", "destination_basel", "destination_lausanne",
        "destination_luzern", "destination_bern",
    ]
    for column in integer_columns:
        df[column] = df[column].astype(int)

    df["mode"] = df["mode"].map(modes.index)
    df["person_weight"] = len(df) * df["person_weight"] / df["person_weight"].sum()
    if not ignore_car_passenger:
        df["is_car_passenger"] = df["is_car_passenger"].astype(int)

    municipalities = ["rural", "suburban", "urban", "urbancore"]
    for column in ["home_municipality", "origin_municipality", "destination_municipality"]:
        df[column] = df[column].map(municipalities.index).astype(int)
    df["urban_destination"] = (df["destination_municipality"] == 2).astype(int)
    df["urbancore_destination"] = (df["destination_municipality"] == 3).astype(int)
    df["region_0"] = (df["ms_region"] == 0).astype(int)
    df["region_1"] = (df["ms_region"] == 1).astype(int)
    df["region_2"] = (df["ms_region"] == 2).astype(int)
    for column in ["short_distance", "long_distance", "very_long_distance", "car_ownership_ratio"]:
        df[column] = df[column].astype(float)
    
    return df.drop(columns=[
        "person_id", "trip_id", "home_municipality", "origin_municipality",
        "sp_region", "ms_region", "ovgk", "pt_egress_time_min", "income_class",
        "pt_access_time_min", "actual_parking_duration_min", "region_0", "purpose",
    ]), modes


def define_parameters(
    ignore_car_passenger: bool,
    use_exponents: bool,
    use_income: bool,
) -> dict[str, Parameter]:
    """Declare the same initial values, bounds and fixed coefficients as Biogeme."""
    exponent_fixed = not use_exponents
    max_disutility = -2e-3
    min_lambda, max_lambda = 1e-2, 2.0

    def p(name, initial, lower=None, upper=None, fixed=False):
        return Parameter(name, float(initial), lower, upper, bool(fixed))

    parameters = {
        item.name: item for item in [
            p("lambda_cost_distance", -0.08, upper=max_disutility),
            p("lambda_cost_income", 0.0, upper=0.0, fixed=not use_income),
            p("lambda_car_travel_time",0.7, min_lambda, max_lambda, fixed=False),
            p("lambda_pt_in_vehicle_time", 1.0, min_lambda, max_lambda, fixed=True),
            p("lambda_pt_access_egress_time", 0.593 if exponent_fixed else 1.0, min_lambda, max_lambda, exponent_fixed),
            p("lambda_pt_transfers", 1.187 if exponent_fixed else 1.0, min_lambda, max_lambda, exponent_fixed),
            p("lambda_pt_transfer_time", 1.0, min_lambda, max_lambda, True),
            p("lambda_pt_distance", 0.0, min_lambda, max_lambda, fixed=False),
            p("lambda_car_passenger_travel_time", 0.832 if exponent_fixed else 1.0, min_lambda, max_lambda, exponent_fixed),
            p("lambda_bike", 0.561 if exponent_fixed else 1.0, min_lambda, max_lambda, exponent_fixed),
            p("lambda_walk", 0.28 if exponent_fixed else 1.0, min_lambda, max_lambda, exponent_fixed),
            p("beta_cost_CHF", -0.12, upper=max_disutility),
            p("beta_destination_employee_density", 0.1),
            p("beta_destination_population_density", 0.2),
            p("beta_destination_companies_density", 0.1, fixed=True),
        ]
    }

    # (name, initial, lower, upper, fixed). Most linear coefficients are unbounded.
    rows = [
        ("beta_car_asc", 3.48), ("beta_car_travel_time_min", -0.964, None, max_disutility),
        ("beta_car_destination_work", .447), ("beta_car_destination_home", 0, None, None, True),
        ("beta_car_destination_education", -.596), ("beta_car_destination_shopping", .608),
        ("beta_car_destination_leisure", .251), ("beta_car_destination_other", .9),
        ("beta_car_origin_home", 0, None, None, True),
        ("beta_car_destination_urban", -.072, None, 0), ("beta_car_destination_urbancore", -1.039, None, 0),
        ("beta_car_sex", -.574), ("beta_car_age", .015), ("beta_car_retired", -.521),
        ("beta_car_junior", 0, None, None, True), ("beta_car_ownership_ratio", -2.25),
        ("beta_car_low_income", 0, None, None, not use_income),
        ("beta_car_high_income", 0, None, None, not use_income),
        ("beta_car_region_1", .223), ("beta_car_region_2", -.467),
        ("beta_car_short_distance", .321), ("beta_car_long_distance", .086),
        ("beta_car_densities", -.1),
        ("beta_car_destination_zurich", 0), ("beta_car_destination_geneva", 0),
        ("beta_car_destination_basel", 0), ("beta_car_destination_lausanne", 0),
        ("beta_car_destination_luzern", 0), ("beta_car_destination_bern", 0),
        ("beta_pt_asc", 0, None, None, True),
        ("beta_pt_access_egress_time_min", -.716, None, max_disutility),
        ("beta_pt_in_vehicle_time_min", -.044, None, max_disutility),
        ("beta_pt_transfers", -.477, None, max_disutility),
        ("beta_pt_transfer_time_min", -.0234, None, -.02),
        ("beta_pt_distance_km", 0.0, -0.5, 0.5),
        ("beta_pt_sex", 0, None, None, True), ("beta_pt_age", 0, None, None, True),
        ("beta_pt_retired", 0, None, None, True), ("beta_pt_junior", 0),
        ("beta_pt_low_income", 0, None, None, not use_income),
        ("beta_pt_high_income", 0, None, None, not use_income),
        ("beta_pt_destination_work", 0, None, None, True),
        ("beta_pt_destination_home", 0, None, None, True),
        ("beta_pt_destination_education", 0, None, None, True),
        ("beta_pt_destination_shopping", 0, None, None, True),
        ("beta_pt_destination_leisure", 0, None, None, True),
        ("beta_pt_destination_other", 0, None, None, True),
        ("beta_pt_origin_home", 0, None, None, True),
        ("beta_pt_destination_urban", 0, None, None, True),
        ("beta_pt_destination_urbancore", 0, None, None, True),
        ("beta_pt_region_1", 0, None, None, True), ("beta_pt_region_2", 0, None, None, True),
        ("beta_pt_short_distance", 0, None, None, True), ("beta_pt_long_distance", 0, None, None, True),
        ("beta_pt_good_service", .685), ("beta_pt_medium_service", .196),
        ("beta_pt_destination_good_service", 0), ("beta_pt_destination_medium_service", 0),
        ("beta_pt_densities", .08),
        ("beta_bike_asc", 3.667), ("beta_bike_travel_time_min", -2.873, None, max_disutility),
        ("beta_bike_age", .02), ("beta_bike_sex", -.441), ("beta_bike_retired", -.848),
        ("beta_bike_junior", 0, None, None, True),
        ("beta_bike_low_income", 0, None, None, not use_income),
        ("beta_bike_high_income", 0, None, None, not use_income),
        ("beta_bike_destination_work", 0, None, None, True),
        ("beta_bike_destination_home", 0, None, None, True),
        ("beta_bike_destination_education", -.45), ("beta_bike_destination_shopping", -.353),
        ("beta_bike_destination_leisure", -.093), ("beta_bike_destination_other", -.515),
        ("beta_bike_origin_home", .228), ("beta_bike_destination_urban", -.32),
        ("beta_bike_destination_urbancore", -.655), ("beta_bike_region_1", -.929),
        ("beta_bike_region_2", -.413), ("beta_bike_short_distance", .429),
        ("beta_bike_long_distance", 0, None, None, True), ("beta_bike_densities", 0),
        ("beta_walk_asc", 10.58), ("beta_walk_travel_time_min", -8.164, None, max_disutility),
        ("beta_walk_age", .007), ("beta_walk_sex", -.211), ("beta_walk_retired", -.285),
        ("beta_walk_junior", 0), ("beta_walk_low_income", 0, None, None, True),
        ("beta_walk_high_income", 0, None, None, True),
        ("beta_walk_destination_work", 0, None, None, True),
        ("beta_walk_destination_home", 0, None, None, True),
        ("beta_walk_destination_education", -.181),
        ("beta_walk_destination_shopping", .022, None, None, True),
        ("beta_walk_destination_leisure", .354), ("beta_walk_destination_other", .111),
        ("beta_walk_origin_home", .198), ("beta_walk_destination_urban", -.18),
        ("beta_walk_destination_urbancore", -.469), ("beta_walk_region_1", .241),
        ("beta_walk_region_2", -.155), ("beta_walk_short_distance", .606),
        ("beta_walk_long_distance", 0, None, None, True),
        ("beta_walk_densities", 0, None, None, True),
    ]
    for row in rows:
        parameter = p(*row)
        parameters[parameter.name] = parameter

    if not ignore_car_passenger:
        cp_rows = [
            ("beta_car_passenger_asc", .46),
            ("beta_car_passenger_travel_time_min", -1.27, None, max_disutility),
            ("beta_car_passenger_distance_km", -.1),
            ("beta_car_passenger_driving_permit", -.339),
            ("beta_car_passenger_age", -.003), ("beta_car_passenger_sex", .141),
            ("beta_car_passenger_retired", .249), ("beta_car_passenger_junior", 0),
            ("beta_car_passenger_low_income", 0, None, None, not use_income),
            ("beta_car_passenger_high_income", 0, None, None, not use_income),
            ("beta_car_passenger_destination_work", .135),
            ("beta_car_passenger_destination_home", 0, None, None, True),
            ("beta_car_passenger_destination_education", -.558),
            ("beta_car_passenger_destination_shopping", .999),
            ("beta_car_passenger_destination_leisure", 1.276),
            ("beta_car_passenger_destination_other", 1.01),
            ("beta_car_passenger_origin_home", 0, None, None, True),
            ("beta_car_passenger_destination_urban", -.146),
            ("beta_car_passenger_destination_urbancore", -1.097),
            ("beta_car_passenger_region_1", .291), ("beta_car_passenger_region_2", -.515),
            ("beta_car_passenger_short_distance", .284),
            ("beta_car_passenger_long_distance", .148),
            ("beta_car_passenger_very_long_distance", 0),
            ("beta_car_passenger_ownership_ratio", 0),
            ("beta_car_passenger_has_car", 0),
            ("beta_car_passenger_densities", -.15),
        ]
        for row in cp_rows:
            parameter = p(*row)
            parameters[parameter.name] = parameter
    else:
        # Retain the unused parameter for identical YAML defaults, but do not
        # send it to the optimizer when the passenger alternative is absent.
        cp_lambda = parameters["lambda_car_passenger_travel_time"]
        parameters[cp_lambda.name] = Parameter(
            cp_lambda.name,
            cp_lambda.init_value,
            cp_lambda.lower_bound,
            cp_lambda.upper_bound,
            True,
        )

    return parameters


LINEAR_TERMS = {
    "car": [
        ("destination_work", "destination_work"), ("destination_home", "destination_home"),
        ("destination_education", "destination_education"), ("destination_shopping", "destination_shopping"),
        ("destination_leisure", "destination_leisure"), ("destination_other", "destination_other"),
        ("destination_urban", "urban_destination"), ("destination_urbancore", "urbancore_destination"),
        ("sex", "sex"), ("retired", "is_retired"), ("junior", "is_junior"),
        ("ownership_ratio", "car_ownership_ratio"), ("low_income", "low_income"),
        ("high_income", "high_income"), ("region_1", "region_1"), ("region_2", "region_2"),
        ("origin_home", "origin_home"), ("short_distance", "short_distance"),
        ("long_distance", "long_distance"), ("destination_zurich", "destination_zurich"),
        ("destination_geneva", "destination_geneva"), ("destination_basel", "destination_basel"),
        ("destination_lausanne", "destination_lausanne"), ("destination_luzern", "destination_luzern"),
        ("destination_bern", "destination_bern"),
    ],
    "pt": [
        ("sex", "sex"), ("retired", "is_retired"), ("junior", "is_junior"),
        ("low_income", "low_income"), ("high_income", "high_income"),
        ("destination_work", "destination_work"), ("destination_home", "destination_home"),
        ("destination_education", "destination_education"), ("destination_shopping", "destination_shopping"),
        ("destination_leisure", "destination_leisure"), ("destination_other", "destination_other"),
        ("destination_urban", "urban_destination"), ("destination_urbancore", "urbancore_destination"),
        ("region_1", "region_1"), ("region_2", "region_2"), ("origin_home", "origin_home"),
        ("short_distance", "short_distance"), ("long_distance", "long_distance"),
        ("good_service", "good_pt_service"), ("medium_service", "medium_pt_service"),
        ("destination_good_service", "destination_good_pt_service"),
        ("destination_medium_service", "destination_medium_pt_service"),
    ],
    "bike": [
        ("sex", "sex"), ("retired", "is_retired"), ("junior", "is_junior"),
        ("low_income", "low_income"), ("high_income", "high_income"),
        ("destination_work", "destination_work"), ("destination_home", "destination_home"),
        ("destination_education", "destination_education"), ("destination_shopping", "destination_shopping"),
        ("destination_leisure", "destination_leisure"), ("destination_other", "destination_other"),
        ("destination_urban", "urban_destination"), ("destination_urbancore", "urbancore_destination"),
        ("region_1", "region_1"), ("region_2", "region_2"), ("origin_home", "origin_home"),
        ("short_distance", "short_distance"), ("long_distance", "long_distance"),
    ],
    "walk": [
        ("sex", "sex"), ("retired", "is_retired"), ("junior", "is_junior"),
        ("low_income", "low_income"), ("high_income", "high_income"),
        ("destination_work", "destination_work"), ("destination_home", "destination_home"),
        ("destination_education", "destination_education"), ("destination_shopping", "destination_shopping"),
        ("destination_leisure", "destination_leisure"), ("destination_other", "destination_other"),
        ("destination_urban", "urban_destination"), ("destination_urbancore", "urbancore_destination"),
        ("region_1", "region_1"), ("region_2", "region_2"), ("origin_home", "origin_home"),
        ("short_distance", "short_distance"), ("long_distance", "long_distance"),
    ],
    "car_passenger": [
        ("driving_permit", "driving_license"), ("sex", "sex"), ("retired", "is_retired"),
        ("junior", "is_junior"), ("low_income", "low_income"), ("high_income", "high_income"),
        ("destination_work", "destination_work"), ("destination_home", "destination_home"),
        ("destination_education", "destination_education"), ("destination_shopping", "destination_shopping"),
        ("destination_leisure", "destination_leisure"), ("destination_other", "destination_other"),
        ("destination_urban", "urban_destination"), ("destination_urbancore", "urbancore_destination"),
        ("region_1", "region_1"), ("region_2", "region_2"),
        ("short_distance", "short_distance"), ("long_distance", "long_distance"),
        ("very_long_distance", "very_long_distance"), ("origin_home", "origin_home"),
        ("ownership_ratio", "car_ownership_ratio"), ("has_car", "has_car"),
    ],
}


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


def build_utilities(
    context,
    arrays: dict[str, jax.Array],
    beta: dict[str, jax.Array],
    modes: list[str],
    ignore_car_passenger: bool,
):
    """Evaluate the utility equations from ``dmc.model`` with JAX."""
    if context.config("distance_cost_interaction"):
        cost_by_distance = _safe_power(
            arrays["euclidean_distance_km"] / context.config("reference_euclidean_distance_km"),
            beta["lambda_cost_distance"],
        )
    else:
        cost_by_distance = 1.0
    if context.config("income_cost_interaction") and context.config("use_income_in_dmc"):
        cost_by_income = _safe_power(
            arrays["income"] / context.config("reference_income_chf"), beta["lambda_cost_income"]
        )
    else:
        cost_by_income = 1.0
    cost_interaction = cost_by_distance * cost_by_income

    density = (
        beta["beta_destination_employee_density"]
        * _safe_power(arrays["destination_employee_density"], constants.EMPLOYEES_DENSITY_EXPONENT)
        / constants.EMPLOYEES_DENSITY_SCALE
        + beta["beta_destination_population_density"]
        * _safe_power(arrays["destination_population_density"], constants.POPULATION_DENSITY_EXPONENT)
        / constants.POPULATION_DENSITY_SCALE
        + beta["beta_destination_companies_density"]
        * _safe_power(arrays["destination_companies_density"], constants.COMPANIES_DENSITY_EXPONENT)
        / constants.COMPANIES_DENSITY_SCALE
    )
    adult_age = jnp.maximum(0.0, arrays["age"] - 17.0) / AGE_SCALE_YEAR

    def base_utility(mode):
        prefix = "beta_car_passenger" if mode == "car_passenger" else f"beta_{mode}"
        utility = beta[f"{prefix}_asc"] + beta[f"{prefix}_age"] * adult_age
        for suffix, column in LINEAR_TERMS[mode]:
            utility = utility + beta[f"{prefix}_{suffix}"] * arrays[column]
        return utility + beta[f"{prefix}_densities"] * density

    car_time = (arrays["car_travel_time_min"] + arrays["parking_searching_duration_min"]) / TIME_SCALE_MIN
    car = (
        base_utility("car")
        + beta["beta_car_travel_time_min"] * _safe_power(car_time, beta["lambda_car_travel_time"])
        + beta["beta_cost_CHF"]
        * (arrays["car_cost_CHF"] + arrays["parking_cost_CHF"])
        * cost_interaction
    )

    pt_distance = arrays["euclidean_distance_km"] / DISTANCE_SCALE_KM
    correction_base = jnp.maximum(PT_REGIONAL_RADIUS_KM / DISTANCE_SCALE_KM - pt_distance, 0.0)
    pt_cost = arrays["pt_cost_CHF"] + beta["beta_pt_distance_km"] * _safe_power(
        correction_base, beta["lambda_pt_distance"]
    )
    pt = (
        base_utility("pt")
        + beta["beta_pt_access_egress_time_min"]
        * _safe_power(arrays["pt_access_egress_time_min"] / TIME_SCALE_MIN, beta["lambda_pt_access_egress_time"])
        + beta["beta_pt_in_vehicle_time_min"]
        * _safe_power(arrays["pt_in_vehicle_time_min"] / TIME_SCALE_MIN, beta["lambda_pt_in_vehicle_time"])
        + beta["beta_pt_transfer_time_min"] * arrays["pt_transfer_time_min"] / TIME_SCALE_MIN
        + beta["beta_pt_transfers"] * _safe_power(arrays["pt_transfers"], beta["lambda_pt_transfers"])
        + beta["beta_cost_CHF"] * pt_cost * cost_interaction
    )

    bike = (
        base_utility("bike")
        + beta["beta_bike_travel_time_min"]
        * _safe_power(arrays["bike_travel_time_min"] / TIME_SCALE_MIN, beta["lambda_bike"])
    )
    walk = (
        base_utility("walk")
        + beta["beta_walk_travel_time_min"]
        * _safe_power(arrays["walk_travel_time_min"] / TIME_SCALE_MIN, beta["lambda_walk"])
    )

    by_mode = {"car": car, "pt": pt, "bike": bike, "walk": walk}
    if not ignore_car_passenger:
        cp_time = arrays["car_passenger_travel_time_min"] / TIME_SCALE_MIN
        by_mode["car_passenger"] = (
            base_utility("car_passenger")
            + beta["beta_car_passenger_travel_time_min"]
            * _safe_power(cp_time, beta["lambda_car_passenger_travel_time"])
            + beta["beta_car_passenger_distance_km"]
            * jnp.maximum(0.0, (arrays["car_passenger_distance_km"] - 50.0) / DISTANCE_SCALE_KM)
        )

    utilities = jnp.stack([by_mode[mode] for mode in modes], axis=1)
    availability = jnp.stack([arrays[f"{mode}_availability"] for mode in modes], axis=1)
    return utilities, availability


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
    optimization = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=callback,
        options={
            "maxiter": context.config("dmc_jax_max_iterations"),
            "gtol": context.config("dmc_jax_gradient_tolerance"),
            "ftol": 1e-11,
            "maxls": 50,
            # The model has 110 free coefficients. A deeper L-BFGS history
            # materially improves its ill-conditioned curvature approximation.
            "maxcor": 100,
        },
    )
    elapsed = time.perf_counter() - started
    beta_values = {
        parameter.name: float(value)
        for parameter, value in zip(free_parameters, optimization.x, strict=True)
    }
    return EstimationResult(beta_values, optimization, elapsed)


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
        "estimator": "JAX/SciPy L-BFGS-B",
        "success": bool(result.optimization.success),
        "message": str(result.optimization.message),
        "log_likelihood": float(-result.optimization.fun),
        "iterations": int(result.optimization.nit),
        "function_evaluations": int(result.optimization.nfev),
        "elapsed_seconds": float(result.elapsed_seconds),
        "estimated_parameters": result.get_beta_values(),
    }
    with open(report_path, "w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    return csv_path, report_path
