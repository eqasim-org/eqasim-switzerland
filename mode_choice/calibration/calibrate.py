import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults
from mode_choice.dmc.utilities.Parameters import Parameters

logger = logging.getLogger(__name__)
def configure(context):
    context.stage("mode_choice.prepare_data")
    context.config("estimate_dmc_parameters")      
    context.config("mode_parameters_path", default = "") 

    if context.config("estimate_dmc_parameters"):
        context.stage("mode_choice.estimate_model.run")

    context.stage("mode_choice.dmc.run_dmc")    
    context.config("data_path")
    context.config("random_seed")

def execute(context):
    prepared_data = context.stage("mode_choice.prepare_data")
    
    # Init DMC
    logger.info("\t Initializing DMC model...")
    DMC = context.stage("mode_choice.dmc.run_dmc")
    
    if context.config("estimate_dmc_parameters"):
        _, _, parameters_file = context.stage("mode_choice.estimate_model.run")
    else:
        parameters_file = context.config("mode_parameters_path")
    
    dmc = DMC(
        parameters_file=parameters_file,
        tours=prepared_data['tours'],
        persons=prepared_data['persons'],
        trips = prepared_data['trips'],
        variables=prepared_data['variables'],
        seed = context.config("random_seed")
    )

    # get the optimizer
    optimizer = context.stage("mode_choice.calibration.optimizer")
    optimizer.run()

    # save the parameters
    parameters_path = os.path.join(context.path(), "calibrated_parameters.yml")
    Parameters.to_yaml(parameters_path)

    return parameters_path