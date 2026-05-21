import os.path
import shutil
import glob
import matsim.runtime.eqasim as eqasim
import logging

logger = logging.getLogger("synpp")

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.simulation.prepare")
    context.stage("matsim.simulation.run")
    context.stage("data.pt_pricing.pt_pricing")
    context.stage("data.microcensus.shares")

    context.stage("calibration.road_regions.freespeed_calibration")
    context.stage("calibration.road_regions.penalty_calibration")

    context.config("calibrate_alphas_in_matsim", default=False)
    context.config("calibrate_betas_in_matsim", default=False)
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")



def execute(context):
    if context.config("extent_path") == "" or context.config("extent_prefix") == "":
        return ""
    
    
    # get the output config from the simulation run in matsim.simulation.run stage
    config_path = "%s/%s" % (context.path("matsim.simulation.run"), "simulation_output/output_config.xml" )
    assert os.path.exists(config_path)
    
    # the output path, where the scenario is created
    output_path = "%s/output" % context.path()

    # change the path of all input files to point to the output files
    # population, households, facitilies, network, transit_schedule, transit_vehicles, vehicles
    with open(config_path) as f_read:
        content = f_read.read()

        content = content.replace(
            'switzerland_population.xml.gz',
            '%s/%s/output_plans.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_households.xml.gz',
            '%s/%s/output_households.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_facilities.xml.gz',
            '%s/%s/output_facilities.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_network.xml.gz',
            '%s/%s/output_network.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_transit_schedule.xml.gz',
            '%s/%s/output_transitSchedule.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_transit_vehicles.xml.gz',
            '%s/%s/output_transitVehicles.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )

        content = content.replace(
            'switzerland_vehicles.xml.gz',
            '%s/%s/output_vehicles.xml.gz' % (context.path("matsim.simulation.run"), "simulation_output")
        )        

        with open("%s/config_cutter.xml" % context.path(), "w+") as f_write:
            f_write.write(content)

    # some args to avoid errors in the cutter
    args = [
    "--config:eqasim:calibration.activate", "false",
    "--config:eqasim:calibration.runCalibration", "false",
    "--config:eqasim:alphaCalibration.activate", "false",
    "--config:eqasim:alphaCalibration.filePath", "",
    "--config:eqasim:networkCalibration.activate", "false",
    "--config:eqasim:networkCalibration.calibrate", "false",
    "--config:eqasim:networkCalibration.countsFile", "",
    "--config:eqasim:networkCalibration.observedSpeedTripsFile", "",
    ]
    
    # use the new config to run the cutter
    config_path = "%s/config_cutter.xml" % context.path()
    events_path = "%s/%s/output_events.xml.gz" % (context.path("matsim.simulation.run"), "simulation_output")
    eqasim.run(context, "org.eqasim.core.scenario.cutter.RunScenarioCutter", [
        "--config-path", config_path,
        "--output-path", output_path,
        "--extent-path", context.config("extent_path"),
        "--threads", context.config("threads"),
        "--prefix", context.config("extent_prefix"),
        "--events-path", events_path,
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator"
    ] + args )

    # move some parameters files to the output path to be comprehensive for the scenario (self contained)
    # 1. mode parameters
    shutil.copyfile("%s/dmc_parameters.yml" % context.path("matsim.simulation.prepare"),
                    "%s/dmc_parameters.yml" % output_path)
    shutil.copyfile("%s/cost_parameters.yml" % context.path("matsim.simulation.prepare"),
                    "%s/cost_parameters.yml" % output_path)

    if context.config("calibrate_alphas_in_matsim") or context.config("calibrate_betas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/%s/*_parameters.yml" % (context.path("matsim.simulation.run"), "simulation_output"))
        if len(calibrated_parameters_path) > 0:            
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)
        shutil.copyfile(calibrated_parameters_path, 
                        "%s/calibrated_dmc_parameters.yml" % output_path)

    # 2.mode shares
    global_shares_output_path, cantonal_shares_output_path = context.stage("data.microcensus.shares")
    shutil.copy(global_shares_output_path, f"{output_path}/global_target_mode_shares.csv" )
    shutil.copy(cantonal_shares_output_path, f"{output_path}/cantonal_target_mode_shares.csv" )

    # 3. pricing parameters
    sbb_path, zones_path, pricing_path = context.stage("data.pt_pricing.pt_pricing")    
    shutil.copy(sbb_path, f"{output_path}/SBB_all_distances.csv" )    
    shutil.copy(zones_path, f"{output_path}/gtfs_zones.csv" )    
    shutil.copy(pricing_path, f"{output_path}/pricingDescription.xml" )

    # 4. special regions
    freespeed_calibration_path = context.stage("calibration.road_regions.freespeed_calibration")
    if freespeed_calibration_path!="" and os.path.exists(freespeed_calibration_path):
        shutil.copy(freespeed_calibration_path, f"{output_path}/freespeed_special_region.yml" )

    penalty_calibration_path = context.stage("calibration.road_regions.penalty_calibration")    
    if penalty_calibration_path!="" and os.path.exists(penalty_calibration_path):
        shutil.copy(penalty_calibration_path, f"{output_path}/penalties_special_region.yml" )

    return output_path