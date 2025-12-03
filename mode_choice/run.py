import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults
from mode_choice.dmc.run_dmc import DMC

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.prepare_data")
    context.stage("mode_choice.dmc.run_dmc")
    context.stage("mode_choice.dmc_defaults")
    
    context.config("data_path")
    context.config("dmc_simulation_data_path", default = os.path.join(context.config("data_path"), "simulation_data"))    

    context.config("random_seed")
    context.config("estimate_dmc_parameters", default = Defaults.ESTIMATE_DMC_PARAMETERS)
    context.config("calibrate_dmc_parameters", default = Defaults.CALIBRATE_DMC_PARAMETERS)
    context.config("mode_parameters_path", default = os.path.join(context.config("dmc_simulation_data_path"), "dmc_parameters.yml"))

    if context.config("estimate_dmc_parameters"):
        context.stage("mode_choice.estimate_model.run")

    if context.config("calibrate_dmc_parameters"):
        context.stage("mode_choice.calibration.calibrate")

    if not context.config("estimate_dmc_parameters") and not context.config("calibrate_dmc_parameters"):
        if context.config("mode_parameters_path")=="":
            raise ValueError("If not estimating or calibrating DMC parameters, 'mode_parameters_path' must be provided.")
        if not os.path.exists(context.config("mode_parameters_path")):
            raise FileNotFoundError(f"Provided mode parameters path does not exist: {context.config('mode_parameters_path')}")

def execute(context):
    prepared_data = context.stage("mode_choice.prepare_data")
    
    # get parameters path
    if context.config("calibrate_dmc_parameters"):
        parameters_file = context.stage("mode_choice.calibration.calibrate")
    elif context.config("estimate_dmc_parameters"):
        _, _, parameters_file = context.stage("mode_choice.estimate_model.run")
    else:
        parameters_file = context.config("mode_parameters_path")
    
    # Init DMC
    logger.info("\t Initializing DMC model...")
    dmc_class: DMC = context.stage("mode_choice.dmc.run_dmc")

    dmc = dmc_class(
        parameters_file=parameters_file,
        tours=prepared_data['tours'],
        persons=prepared_data['persons'],
        trips=prepared_data['trips'],
        variables=prepared_data['variables'],
        seed=context.config("random_seed")
    )

    # Run DMC
    logger.info("\t Running DMC model...")
    choices = dmc.run()

    # turn them back into trips
    choice = (choices
              .select(["person_id", "trip_id", "euclidean_distance_km", "mode_candidates"])
              .explode(["trip_id", "mode_candidates", "euclidean_distance_km"])
              .rename({"mode_candidates":"mode"})
              .with_columns([
                  pl.col("trip_id").str.split('_').list.get(1).cast(pl.Int32).alias("trip_index"),
              ]))

    return (parameters_file, 
            choice[["person_id","trip_index","trip_id","euclidean_distance_km","mode"]].to_pandas())