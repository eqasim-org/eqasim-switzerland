import os.path
import shutil
import matsim.runtime.eqasim as eqasim
from matsim.simulation.config_utils import get_mode_shares_calibration_args, get_delays_args, get_network_calibration_args
import logging
logger = logging.getLogger("synpp")

def configure(context):
    context.stage("matsim.simulation.prepare")    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("data.microcensus.shares")
    
    context.config("use_vdf", default=False)
    context.config("threads")
    context.config("last_iteration", 60)
    context.config("data_path")  

    # dmc
    if context.config("estimate_dmc"):
        context.stage("dmc.model")

    # mode shares calibration
    context.config("estimate_dmc", default=False)
    context.config("calibrate_alphas_in_matsim", default=False)
    context.config("alphaCalibration.level", default="global")    
    context.config("calibrate_betas_in_matsim", default=False)

    # traffic light and intersection delays
    context.config("activate_traffic_light_delays", default=False)
    context.config("activate_unsignalized_intersections_delays", default=False)

    # network calibration
    context.config("network_calibration.activate", default=False)
    context.config("network_calibration.calibrate_disutilities", default=True)
    context.config("network_calibration.calibrate_freespeed", default=True)
    
    if context.config("network_calibration.activate") and context.config("network_calibration.calibrate_disutilities"):
        context.stage("analysis.counts.target")
        context.stage("calibration.road_regions.regions")
    
    if context.config("network_calibration.activate") and context.config("network_calibration.calibrate_freespeed"):
        context.stage("analysis.travel_times.APIs.target")

    context.config("correct_links_capacity", False)
    context.config("minimum_speed", 1.0)

    context.config("useScheduleBasedTransport", default=True)
    context.config("preventwaitingtoentertraffic", default = "no")
    context.config("writeexperiencedplans", default = "no")
    context.config("preventwaitingtoentertraffic", default = "no")
    context.config("writeexperiencedplans", default = "no")


def execute(context):
    config_path = "%s/%s" % (
        context.path("matsim.simulation.prepare"),
        context.stage("matsim.simulation.prepare")
    )
    
    if context.config("useScheduleBasedTransport"):
        scheduleBasedPTconfig = "true"
    else:
        scheduleBasedPTconfig = "false"

    preventwaitingtoentertraffic = "n"
    if context.config("preventwaitingtoentertraffic"):
        preventwaitingtoentertraffic = "y"
        logger.info("Prevent waiting to enter traffic: %s", preventwaitingtoentertraffic)

    writeExperiencedPlans = "false"
    if context.config("writeexperiencedplans"):
        writeExperiencedPlans = "true"
        logger.info("Write experienced plans: %s", writeExperiencedPlans)

    # dmc estimation and calibration
    additional_args = []
    if context.config("estimate_dmc"):
        _, _, (mode_params_path, cost_params_path), _ , _ = context.stage("dmc.model")
        mode_parameters_path = "%s/estimated_dmc_parameters.yml" % context.path("matsim.simulation.prepare")
        cost_parameters_path = "%s/dmc_cost_parameters.yml" % context.path("matsim.simulation.prepare")
        shutil.copy(mode_params_path, mode_parameters_path)
        shutil.copy(cost_params_path, cost_parameters_path)
        additional_args.extend(["--config:eqasim.costParametersPath", cost_parameters_path])
        additional_args.extend(["--config:eqasim.modeParametersPath", mode_parameters_path])

    additional_args.extend(get_mode_shares_calibration_args(context))
    
    # delays (signalized intersections delays using webster formula, and unsignalized intersection delays using BPR based approach)
    additional_args.extend(get_delays_args(context))

    # network calibration
    additional_args.extend(get_network_calibration_args(context))

    # Running the simulation
    last_iteration = context.config("last_iteration")
    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(int(last_iteration/2)),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(int(last_iteration/2)),
            "--config:linkStats.averageLinkStatsOverIterations", str(8),
            # if one wants to visualize outputs, trips file needs to be generated 
            # so one should set this to something other than 0, and preferebly to something 
            # that will output trips file at the end of the simulation
            "--config:controller.writeTripsInterval", str(int(last_iteration/2)),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
            "--preventwaitingtoentertraffic", preventwaitingtoentertraffic,
            "--config:scoring.writeExperiencedPlans", writeExperiencedPlans
        ] + additional_args)
    else:
        # Run simulation with vdf
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunVDFSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1),
            "--config:controller.writeTripsInterval", str(0),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
        ] + additional_args)
    assert os.path.exists("%s/simulation_output/output_events.xml.gz" % context.path())
    
    return context.path()
