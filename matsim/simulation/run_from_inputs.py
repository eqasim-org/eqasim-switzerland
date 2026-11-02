import os.path
import shutil
import matsim.runtime.eqasim as eqasim
from matsim.config.tools import (
    java_boolean, parse_boolean, copy_config, adapt_existing_scenario,
    resolve_output_file, scenario_input_parameters, change_params,
)
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.config("use_vdf", default=False)
    context.config("input_downsampling")
    context.config("use_freight", True)
    context.config("output_prefix", "switzerland_")

    context.config("useScheduleBasedTransport", default=False)
    context.config("preventwaitingtoentertraffic", default=False)
    context.config("writeexperiencedplans", default=False)

    context.config("simulation_inputs_path", default = "")
    context.config("experiment_name", default = "experiment")
    context.config("simulation_config_path", default = "switzerland_config.xml")


def execute(context):

    scheduleBasedPTconfig = java_boolean(
        context.config("useScheduleBasedTransport"), "useScheduleBasedTransport")
    if parse_boolean(context.config("useScheduleBasedTransport"), "useScheduleBasedTransport"):
        logger.info("Schedule-based PT: %s", scheduleBasedPTconfig)

    preventwaitingtoentertraffic = "n"
    if parse_boolean(context.config("preventwaitingtoentertraffic"), "preventwaitingtoentertraffic"):
        preventwaitingtoentertraffic = "y"
        logger.info("Prevent waiting to enter traffic: %s", preventwaitingtoentertraffic)

    writeExperiencedPlans = "false"
    if parse_boolean(context.config("writeexperiencedplans"), "writeexperiencedplans"):
        writeExperiencedPlans = "true"
        logger.info("Write experienced plans: %s", writeExperiencedPlans)

    input_path  = context.config("simulation_inputs_path")

    output_path = os.path.join(context.path(), context.config("experiment_name"))
    logger.info("Output path: %s", output_path)
    os.makedirs(output_path, exist_ok=True)

    config = context.config("simulation_config_path")
    origin      = os.path.join(input_path, config)
    destination = os.path.join(output_path, config)
    copy_config(origin, destination, input_paths={
        "eqasim": ("modeParametersPath", "costParametersPath"),
        "ptZones": ("ptZonesFilePath", "sbbDistancesPath", "pricingDescriptionPath"),
        "eqasim:alphaCalibration": ("filePath",),
        "eqasim:networkCalibration": ("countsFile",),
        "eqasim:networkCalibration.costCalibration": ("specialRegionPath",),
        "eqasim:networkCalibration.freespeedCalibration": ("specialRegionPath", "observedTripsFile"),
    })

    prefix = context.config("output_prefix")
    for file in ["facilities.xml.gz", "households.xml.gz", "network.xml.gz", "population.xml.gz",
                 "transit_schedule.xml.gz", "transit_vehicles.xml.gz", "vehicles.xml.gz"]:
        origin = resolve_output_file(os.path.join(input_path, prefix + file))
        destination = os.path.join(output_path, os.path.basename(origin))
        shutil.copy2(origin, destination)

    config_path = os.path.join(output_path, config)
    assert os.path.exists(config_path)

    adapt_existing_scenario(config_path, context.config("input_downsampling"),
                            context.config("output_prefix"), context.config("use_freight"))
    change_params(config_path, scenario_input_parameters(output_path, prefix))

    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.outputDirectory", os.path.join(output_path, "simulation_output"),
            "--config:controler.lastIteration", str(1),
            "--config:controler.writeEventsInterval", str(1),
            "--config:controler.writePlansInterval", str(1),
            #"--config:transitRouter.directWalkFactor", str(1.0),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
            "--preventwaitingtoentertraffic", preventwaitingtoentertraffic,
            "--config:scoring.writeExperiencedPlans", writeExperiencedPlans
        ])
    else:
        # Run simulation with vdf
        eqasim.run(context, "org.eqasim.switzerland.RunVDFSimulation", [
            "--config-path", config_path,
            "--config:controler.outputDirectory", os.path.join(output_path, "simulation_output"),
            "--config:controler.lastIteration", str(1),
            "--config:controler.writeEventsInterval", str(1),
            "--config:controler.writePlansInterval", str(1),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
        ])
    resolve_output_file(os.path.join(output_path, "simulation_output", "output_events.xml"))
    
    return context.path()
