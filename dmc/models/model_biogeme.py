import os

from dmc.constants import constants
from dmc.parameters import writer
from dmc.vot.functions import vot_utils

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, BinaryMax, Variable, exp
from dmc.models.declarations import define_parameters
from dmc.models.data import preprocess_data
from dmc.models.utility import build as build_shared_utilities
from dmc.models.validation import validate
import numpy as np
from biogeme.results_processing import get_pandas_estimated_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODES = ['car', 'pt', 'bike', 'walk', 'car_passenger']
TIME_SCALE_MIN = constants.TIME_SCALE_MIN
DISTANCE_SCALE_KM = constants.DISTANCE_SCALE_KM
PT_REGIONAL_RADIUS_KM = constants.PT_REGIONAL_RADIUS_KM
AGE_SCALE_YEAR = constants.AGE_SCALE_YEAR

def configure(context):
    context.stage("dmc.data.training_data")
    context.stage("data.constants")
    if context.config("dmc_vot_constraints", True):
        raise ValueError("VOT-constrained estimation requires dmc_estimator: jax")
    context.config("ignore_car_passenger", False)
    context.config("distance_cost_interaction", True)
    context.config("income_cost_interaction", True)
    context.config("use_exponents", True)
    context.config("use_income_in_dmc", True)
    context.config("use_weights_for_vot", False)
    context.config("dmc_transfer_peak_km", 7.0)
    
    # these are used in the writer
    context.config("urbancore_parking_search_min", constants.URBANCORE_PARKING_SEARCH_MIN)
    context.config("urban_parking_search_min", constants.URBAN_PARKING_SEARCH_MIN) 
    context.config("suburban_parking_search_min", constants.SUBURBAN_PARKING_SEARCH_MIN)
    context.config("car_cost_per_km", constants.CAR_COST_PER_KM) #CHF per km
    context.config("parking_cost_per_hour_CHF_urban", constants.PARKING_COST_PER_HOUR_CHF_URBAN)
    context.config("parking_cost_per_hour_CHF_urbancore", constants.PARKING_COST_PER_HOUR_CHF_URBANCORE)
    context.config("parking_cost_per_hour_CHF_suburban", constants.PARKING_COST_PER_HOUR_CHF_SUBURBAN)
    context.config("parking_price_reduction_for_work", constants.PARKING_PRICE_REDUCTION_FOR_WORK)
    context.config("car_cost_model", constants.CAR_COST_MODEL)
    context.config("reference_euclidean_distance_km", constants.REF_EUCLIDEAN_DISTANCE_KM)
    context.config("reference_income_chf", constants.REF_INCOME_CHF)
    context.config("pt_regional_radius_km", constants.PT_REGIONAL_RADIUS_KM)


def execute(context):

    # Prepare conditional histories and common model covariates.

    df = context.stage("dmc.data.training_data")
    ignore_car_passenger = context.config("ignore_car_passenger")
    use_exponents = context.config("use_exponents")
    use_income = context.config("use_income_in_dmc")

    df, modes = preprocess_data(df, ignore_car_passenger)
    log_trip_stats(df, modes)

    # Build the same utility expressions and declarations as JAX.

    database = db.Database("data", df)
    vars = define_variables(database, ignore_car_passenger)
    betas = define_betas(ignore_car_passenger, use_exponents, use_income,
                         income_reference=float(np.average(df.income, weights=df.person_weight)),
                         transfer_peak_km=context.config('dmc_transfer_peak_km'),
                         distance_interaction=context.config('distance_cost_interaction'),
                         income_interaction=context.config('income_cost_interaction'))
    utilities, availability = build_utilities(context, vars, betas, modes, ignore_car_passenger)

    # Estimate and restore the caller's working directory on failure as well.

    logprob = models.loglogit(utilities, availability, vars["mode"])
    cwd = os.getcwd()
    try:
        os.chdir(context.path())
        biogeme = create_biogeme(database, logprob, vars["weight"])
        biogeme.calculate_null_loglikelihood(availability)
        result = biogeme.estimate()
    finally:
        os.chdir(cwd)
    
    # Print summary of the results
    logger.info(result.short_summary())

    # write the optimal parameters to a yaml file in MATSim input format    
    mode_params_path, cost_params_path = writer(context, result, betas).write()

    # write the optimal parameters to a csv file
    csv_params_path = os.path.join(context.path(), "dmc_model_parameters.csv")
    estimated_parameters = get_pandas_estimated_parameters(
        estimation_results=result
    )["Estimated parameters"].set_index("Name")
    estimated_parameters.to_csv(csv_params_path)
    logger.info("The estimated parameters are saved to %s", csv_params_path)

    # Compute the VOT for car users
    vot_car, mean_vot_car = vot_utils.get_car_vot(context, df, result, MODES, betas)
    vot_pt, mean_vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer = vot_utils.get_pt_vot(context, df, result, MODES, betas)

    logger.info("Mean marginal WTP for observed car trips: %.2f CHF/hour", mean_vot_car)
    logger.info("Mean marginal WTP for observed PT trips: %.2f CHF/hour", mean_vot_pt)
    logger.info("PT component WTP (CHF/hour): in-vehicle %.2f, access/egress %.2f, waiting %.2f",
                *[vot_utils._mean(values, df["person_weight"], context.config("use_weights_for_vot"))
                  for values in (vot_in_vehicle, vot_access_egress, vot_transfer)])

    path_to_figure = os.path.join(context.path(),"vot_distribution.png")
    use_vot_weights = context.config("use_weights_for_vot")
    logger.info(
        "WTP summaries and histogram use %s trip observations",
        "survey-weighted" if use_vot_weights else "unweighted",
    )
    vot_utils.plot_vot(
        vot_car,
        vot_pt,
        figure_path=path_to_figure,
        car_weights=df.loc[vot_car.index, "person_weight"] if use_vot_weights else None,
        pt_weights=df.loc[vot_pt.index, "person_weight"] if use_vot_weights else None,
    )
    logger.info("The VOT distribution figure is saved to %s", path_to_figure)
    
    return (result, 
            df, 
            (mode_params_path, cost_params_path), 
            path_to_figure,
            (vot_car, vot_pt, vot_in_vehicle, vot_access_egress, vot_transfer))


def define_variables(database, ignore_car_passenger):
    variables = {name: Variable(name) for name in database.dataframe.columns}
    variables['weight'] = Variable('person_weight')
    return variables

def define_betas(ignore_car_passenger, use_exponents, use_income, **kwargs):
    return {name: Beta(name, p.init_value, p.lower_bound, p.upper_bound, int(p.fixed))
            for name, p in define_parameters(ignore_car_passenger, use_exponents, use_income, **kwargs).items()}

def build_utilities(context, vars, betas, modes, ignore_car_passenger):
    def power(base, exponent):
        # Protect exponent derivatives at zero; retain exactly zero contribution.
        positive = base > 0
        safe_base = BinaryMax(base, 0.) + (base <= 0)
        return positive * safe_base**exponent
    by_mode = build_shared_utilities(context, vars, betas, modes,
                                     power=power, maximum=BinaryMax, exp=exp)
    return ({i: by_mode[mode] for i, mode in enumerate(modes)},
            {i: vars[f'{mode}_availability'] for i, mode in enumerate(modes)})

def log_trip_stats(df, modes):
    for m in modes:
        sel = df["mode"].isin([modes.index(m)])
        logger.info("%s : number of trips is %d and average distance is %.2f km", m, sel.sum(), df[sel].euclidean_distance_km.mean())
    pt_or_car = df["mode"].isin([modes.index("car"), modes.index("pt")])
    logger.info("The average euclidean distance is: %.2f km, for pt and car is %.2f km", df.euclidean_distance_km.mean(), df[pt_or_car].euclidean_distance_km.mean())
    logger.info("The average income is: %.2f CHF", df.income.mean())


def create_biogeme(database, log_probability, weight):
    estimator = bio.BIOGEME(
        database,
        {"log_like": log_probability, "weight": weight},
        number_of_threads=8,
        optimization_algorithm="simple_bounds_BFGS",
        numerically_safe=True,
        generate_html=True,
        generate_yaml=True,
    )
    estimator.model_name = "DMC_model"
    return estimator


