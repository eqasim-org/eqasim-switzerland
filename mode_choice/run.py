import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults

logger = logging.getLogger(__name__)
SHORT_DISTANCE_LIMIT_KM = Defaults.SHORT_DISTANCE_LIMIT_KM
LONG_DISTANCE_LIMIT_KM = Defaults.LONG_DISTANCE_LIMIT_KM

def configure(context):
    context.stage("mode_choice.prepare_data")
    context.stage("mode_choice.estimate_mode.run")
    context.stage("mode_choice.dmc.run_dmc")

    context.config("data_path")
    context.config("random_seed")

def execute(context):
    prepared_data = context.stage("mode_choice.prepare_data")
    
    # Init DMC
    logger.info("\t Initializing DMC model...")
    DMC = context.stage("mode_choice.dmc.run_dmc")
    _, _, parameters_file = context.stage("mode_choice.estimate_mode.run")
    
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