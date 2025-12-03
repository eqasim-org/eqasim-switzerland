import os
import polars as pl
import logging
from mode_choice.dmc_defaults import Defaults
from mode_choice.dmc.utilities.Parameters import Parameters
from mode_choice.dmc.run_dmc import DMC

logger = logging.getLogger(__name__)
def configure(context):
    context.stage("mode_choice.prepare_data")
    
    context.config("estimate_dmc_parameters", default = Defaults.ESTIMATE_DMC_PARAMETERS)  
    context.config("data_path")
    context.config("dmc_simulation_data_path", default = os.path.join(context.config("data_path"), "simulation_data"))        
    context.config("mode_parameters_path", default = os.path.join(context.config("dmc_simulation_data_path"), "dmc_parameters.yml"))

    if context.config("estimate_dmc_parameters"):
        context.stage("mode_choice.estimate_model.run")

    context.stage("mode_choice.dmc.run_dmc")
    context.stage("mode_choice.calibration.optimizer")
    context.stage("mode_choice.dmc_defaults")
        
    context.config("random_seed")

def execute(context):
    prepared_data = context.stage("mode_choice.prepare_data")
    
    # Init DMC
    logger.info("\t Initializing DMC model...")
    dmc_class: DMC = context.stage("mode_choice.dmc.run_dmc")
    
    if context.config("estimate_dmc_parameters"):
        _, _, parameters_file = context.stage("mode_choice.estimate_model.run")
    else:
        parameters_file = context.config("mode_parameters_path")
    
    dmc = dmc_class(
        parameters_file=parameters_file,
        tours=prepared_data['tours'],
        persons=prepared_data['persons'],
        trips = prepared_data['trips'],
        variables=prepared_data['variables'],
        seed = context.config("random_seed")
    )

    # get the optimizer
    optimizer = context.stage("mode_choice.calibration.optimizer")
    res = optimizer.run(dmc)

    # save the parameters
    new_parameters_path = os.path.join(context.path(), "calibrated_parameters.yml")
    Parameters.to_yaml(new_parameters_path)

    return new_parameters_path