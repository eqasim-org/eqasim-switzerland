import os.path
import shutil
import matsim.runtime.eqasim as eqasim
from matsim.simulation.config_utils import get_calibration_args, get_delays_args

def configure(context):
    context.stage("matsim.simulation.prepare")    
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    
    context.config("use_vdf", default=False)
    context.config("threads")
    context.config("last_iteration", 60)    

    context.config("estimate_dmc", default=False)
    context.config("calibrate_alphas_in_matsim", default=False)
    context.config("alphaCalibration.level", default="global")
    context.stage("data.microcensus.shares")

    context.config("calibrate_betas_in_matsim", default=False)
    context.config("activate_traffic_light_delays", default=False)
    context.config("activate_unsignalized_intersections_delays", default=False)

    if context.config("estimate_dmc"):
        context.stage("dmc.model")

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
        print("Prevent waiting to enter traffic: " + preventwaitingtoentertraffic)

    writeExperiencedPlans = "false"
    if context.config("writeexperiencedplans"):
        writeExperiencedPlans = "true"
        print("Write experienced plans: " + writeExperiencedPlans)

    # dmc estimation and calibration
    additional_args = []
    if context.config("estimate_dmc"):
        _, _, estimated_parameters_path, _ = context.stage("dmc.model")
        mode_parameters_path = "%s/estimated_dmc_parameters.yml" % context.path("matsim.simulation.prepare")
        shutil.copy(estimated_parameters_path, mode_parameters_path)
        additional_args.extend(["--config:eqasim.modeParametersPath", mode_parameters_path])

    additional_args.extend(get_calibration_args(context))
    # delays (signalized intersections delays using webster formula, and unsignalized intersection delays using BPR based approach)
    additional_args.extend(get_delays_args(context))

    # Running the simulation
    last_iteration = context.config("last_iteration")
    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.writeEventsInterval", str(last_iteration),
            "--config:controler.writePlansInterval", str(last_iteration),
            "--config:qsim.numberOfThreads", str(min(context.config("threads"),12)),
            "--config:linkStats.writeLinkStatsInterval", str(last_iteration),
            "--config:linkStats.averageLinkStatsOverIterations", str(1),
            # if one wants to visualize outputs, trips file needs to be generated 
            # so one should set this to something other than 0, and preferebly to something 
            # that will output trips file at the end of the simulation
            "--config:controller.writeTripsInterval", str(last_iteration),
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
