import glob
import os
import logging
import matsim.runtime.eqasim as eqasim

logger = logging.getLogger("synpp:    Regional Model\t")

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.cutter.scenario")
    
    context.config("use_vdf", default=False)
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")


def execute(context):
    # 1. get the regional scenario if exists
    regional_scenario = context.stage("matsim.cutter.scenario")
    if regional_scenario=="":
        return ("","")
    assert os.path.exists(regional_scenario), "Regional scenario does not exist: %s" % regional_scenario
    logger.info("Regional scenario found: %s" % regional_scenario)

    # 2. go to the regional scenario dir (this makes sure all paths are working)
    cwd = os.getcwd()
    os.chdir(regional_scenario)
    logger.info("Changed working directory to regional scenario: %s" % os.getcwd())

    # 3. config file
    config_file = "%s/%sconfig.xml" % (regional_scenario, context.config("extent_prefix") )
    assert os.path.exists(config_file), "Config file does not exist: %s" % config_file
    logger.info("Config file found: %s" % config_file)

    # 4. run the scenario in eqasim (we do not run any calibration, and all other files are already set in the config from the natioanl model)
    dmc_param_path = "calibrated_dmc_parameters.yml" if os.path.exists("%s/calibrated_dmc_parameters.yml" % regional_scenario) else "dmc_parameters.yml"
    freespeed_special_region = get_regions_path(regional_scenario, kind="freespeed")
    penalty_special_region = get_regions_path(regional_scenario, kind="penalty")
    args = [
        "--config:eqasim:calibration.activate", "false",
        "--config:eqasim:calibration.runCalibration", "false",
        "--config:eqasim:alphaCalibration.activate", "false",
        "--config:eqasim:alphaCalibration.filePath", "cantonal_target_mode_shares.csv",
        "--config:eqasim:networkCalibration.activate", "true", # this will not calibrate, but just activate the module to use penalties and speed factors
        "--config:eqasim:networkCalibration.calibrate", "false",
        "--config:eqasim:networkCalibration.countsFile", "",
        "--config:eqasim:networkCalibration.observedSpeedTripsFile", "",
        "--config:eqasim:networkCalibration.penaltiesSpecialRegionPath", penalty_special_region,
        "--config:eqasim:networkCalibration.freespeedSpecialRegionPath", freespeed_special_region,
        "--config:eqasim.costParametersPath", "cost_parameters.yml",
        "--config:eqasim.modeParametersPath", dmc_param_path,
        "--config:ptZones.ptZonesFilePath", "gtfs_zones.csv",
        "--config:ptZones.sbbDistancesPath", "SBB_all_distances.csv",
        "--config:ptZones.pricingDescriptionPath", "pricingDescription.xml",
    ]


    # run the simulation
    if not context.config("use_vdf"):
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_file,
        ] + args)
    else:
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunVDFSimulation", [
            "--config-path", config_file,
            "--generateNetworkEvents", "true",
        ] + args)        

    simulation_path = "%s/simulation_output" % context.path()
    assert os.path.exists(simulation_path), "Simulation output path does not exist: %s" % simulation_path
    assert os.path.exists("%s/output_events.xml.gz" % simulation_path), "Output events file does not exist: %s" % ("%s/output_events.xml.gz" % simulation_path)
    os.chdir(cwd)

    return simulation_path, regional_scenario



def get_regions_path(path,kind="freespeed"):
    regions = []
    if kind=="freespeed":
        region_dir = os.path.join(path, "calibration_regions")
        if os.path.exists(region_dir):
            regions = glob.glob(f"{region_dir}/freespeed_special_region_*.yml")
    elif kind=="penalty":
        region_dir = os.path.join(path, "calibration_regions")
        if os.path.exists(region_dir):
            regions = glob.glob(f"{region_dir}/penalties_special_region_*.yml")
    
    if len(regions)==0:
        return ""
    
    # only keep the region_dir/region.yml part, not the full path
    regions = [os.path.join("calibration_regions", os.path.basename(region)) for region in regions]
    return ";".join(regions)