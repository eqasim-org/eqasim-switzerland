import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults

logger = logging.getLogger(__name__)

def configure(context):
    context.stage("mode_choice.prepare_data")
    context.stage("mode_choice.estimate_mode.run")
    context.stage("mode_choice.dmc.run_dmc")

    context.config("data_path")
    context.config("random_seed")
    context.config("calibrated_parameters", default=False)
    if context.config("calibrated_parameters"):
        context.stage("mode_choice.calibration.calibrate")

def execute(context):
    prepared_data = context.stage("mode_choice.prepare_data")
    
    # get parameters path
    if context.config("calibrated_parameters"):
        parameters_file = context.stage("mode_choice.calibration.calibrate")
    else:
        _, _, parameters_file = context.stage("mode_choice.estimate_mode.run")
    
    # Init DMC
    logger.info("\t Initializing DMC model...")
    DMC = context.stage("mode_choice.dmc.run_dmc")    

    dmc = DMC(
        parameters_file=parameters_file,
        tours=prepared_data['tours'],
        persons=prepared_data['persons'],
        trips = prepared_data['trips'],
        variables=prepared_data['variables'],
        seed = context.config("random_seed")
    )

    # Run DMC
    logger.info("\t Running DMC model...")
    choices = dmc.run()

    # turn them back into trips
    choice = (choices
              .select(["person_id","trip_id","mode_candidates"])
              .explode(["trip_id","mode_candidates"])
              .rename({"mode_candidates":"mode"})
              .with_columns([
                  pl.col("trip_id").str.split('_').list.get(1).cast(pl.Int32).alias("trip_index"),
              ]))

    return choice[["person_id","trip_index","trip_id","mode"]].to_pandas()