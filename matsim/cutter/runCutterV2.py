import os.path
import shutil
import glob
import matsim.runtime.eqasim as eqasim
import logging
from .cutter_tools import change_params, cut_csv_to_region, cut_csv_to_network, get_regions_path

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
    context.config("shape_name")    
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")
    context.config("use_vdf", default=False)
    context.config("threads")

def execute(context):
    if context.config("extent_path") == "" or context.config("extent_prefix") == "":
        return ""

    assert context.config("use_vdf"), "use of vdf needs to be enabled to use cutterV2"
    # get the output config from the simulation run in matsim.simulation.run stage
    config_path = "%s/%s" % (context.path("matsim.simulation.run"), "simulation_output/output_config.xml")
    assert os.path.exists(config_path)
    
    # the output path, where the scenario is created
    output_path = "%s/output" % context.path()
    # change the path of all input files to point to the output files
    # population, households, facitilies, network, transit_schedule, transit_vehicles, vehicles
    p = context.path("matsim.simulation.run")
    cutter_config_path = "%s/config_cutter.xml" % context.path()
    change_params(config_path = config_path, 
                    output_path = cutter_config_path,
                    params = [
                            # First: use the simulation results to cut the scenario
                            ("plans.inputPlansFile", '%s/%s/output_plans.xml.gz' % (p, "simulation_output")),
                            ("households.inputFile", '%s/%s/output_households.xml.gz' % (p, "simulation_output")),
                            ("facilities.inputFacilitiesFile",'%s/%s/output_facilities.xml.gz' % (p, "simulation_output")),
                            ("network.inputNetworkFile",'%s/%s/output_network.xml.gz' % (p, "simulation_output")),
                            ("transit.transitScheduleFile", '%s/%s/output_transitSchedule.xml.gz' % (p, "simulation_output")),
                            ("transit.vehiclesFile", '%s/%s/output_transitVehicles.xml.gz' % (p, "simulation_output")),
                            ("vehicles.vehiclesFile", '%s/%s/output_vehicles.xml.gz' % (p, "simulation_output"))
                            # Second: turn off calibration config
                            ("eqasim:calibration.activate", "false"),
                            ("eqasim:calibration.runCalibration","false"),
                            ("eqasim:alphaCalibration.activate","false"),
                            ("eqasim:alphaCalibration.filePath","false"),
                            ("eqasim:networkCalibration.activate","false"),
                            ("eqasim:networkCalibration.calibrate","false"),
                            ("eqasim:networkCalibration.countsFile",""),
                            ("eqasim:networkCalibration.freespeedCalibration.observedTripsFile",""),
                            ] )

    # use the new config to run the cutter
    eqasim.run(context, "org.eqasim.core.scenario.cutter.RunScenarioCutterV2", [
        "--config-path", cutter_config_path,
        "--output-path", output_path,
        "--extent-path", "%s/%s" % (context.config("extent_path"), context.config("shape_name")),
        "--vdf-travel-times-path", "%s/%s" % (context.path("matsim.simulation.run"), "simulation_output/vdf.bin"),
        "--threads", context.config("threads"),
        "--prefix", context.config("extent_prefix")
        ] )

    # move some parameters files to the output path to be comprehensive for the scenario (self contained)
    # 1. mode parameters
    shutil.copyfile("%s/dmc_parameters.yml" % context.path("matsim.simulation.prepare"),
                    "%s/dmc_parameters.yml" % output_path)
    shutil.copyfile("%s/cost_parameters.yml" % context.path("matsim.simulation.prepare"),
                    "%s/cost_parameters.yml" % output_path)
    dmc_params = "dmc_parameters.yml"
    if context.config("calibrate_alphas_in_matsim") or context.config("calibrate_betas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/%s/*_parameters.yml" % (context.path("matsim.simulation.run"), "simulation_output"))
        if len(calibrated_parameters_path) > 0:            
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)
            shutil.copyfile(calibrated_parameters_path, 
                            "%s/calibrated_dmc_parameters.yml" % output_path)
            dmc_params = "calibrated_dmc_parameters.yml"

    # 2.mode shares
    global_shares_output_path, cantonal_shares_output_path = context.stage("data.microcensus.shares")
    shutil.copy(global_shares_output_path, f"{output_path}/global_target_mode_shares.csv" )
    shutil.copy(cantonal_shares_output_path, f"{output_path}/cantonal_target_mode_shares.csv" )

    # 3. pricing parameters
    sbb_path, zones_path, pricing_path = context.stage("data.pt_pricing.pt_pricing")    
    shutil.copy(sbb_path, f"{output_path}/SBB_all_distances.csv" )    
    shutil.copy(zones_path, f"{output_path}/gtfs_zones.csv" )    
    shutil.copy(pricing_path, f"{output_path}/pricingDescription.xml" )

    # 4. calibration files (special regions and target files)
    regionl_speeds_file = ""
    if context.config("network_calibration.activate") and context.config("network_calibration.calibrate_freespeed"):
        freespeed_calibration_path = context.stage("calibration.road_regions.freespeed_calibration")
        region_dir = os.path.join(output_path, "network_calibration_files")
        os.makedirs(region_dir, exist_ok=True)
        if freespeed_calibration_path!="":                        
            regions = freespeed_calibration_path.split(";")
            for i,region in enumerate(regions):
                if os.path.exists(region):
                    shutil.copy(region, f"{region_dir}/freespeed_special_region_{i}.yml" )

        target_traveltimes_file = context.stage("analysis.travel_times.APIs.target")
        if target_traveltimes_file!="":
            cut_csv_to_region(csv_path= target_traveltimes_file, 
                              region_path = context.config("extent_path"),
                              output_path = f"{region_dir}/target_travel_times.csv")
            if os.path.exists(f"{region_dir}/target_travel_times.csv"):
                    regionl_speeds_file = "network_calibration_files/target_travel_times.csv"

    regional_counts_file = ""
    if context.config("network_calibration.activate") and context.config("network_calibration.calibrate_disutilities"):
        penalty_calibration_path = context.stage("calibration.road_regions.penalty_calibration")    
        region_dir = os.path.join(output_path, "network_calibration_files")
        os.makedirs(region_dir, exist_ok=True)
        if penalty_calibration_path!="":            
            regions = penalty_calibration_path.split(";")
            for i,region in enumerate(regions):
                if os.path.exists(region):
                    shutil.copy(region, f"{region_dir}/penalties_special_region_{i}.yml" )
        
        target_counts_file = context.stage("analysis.counts.target")
        if target_counts_file!="":
            cut_csv_to_network(csv_path = target_counts_file, 
                               network_path = "%s/%snetwork.xml.gz" % (output_path, context.config("extent_prefix")),
                               output_path = f"{region_dir}/target_counts.csv")
            if os.path.exists(f"{region_dir}/target_counts.csv"):
                regional_counts_file = "network_calibration_files/target_counts.csv"

    # 5. modify the regional model config
    config_file = "%s/%sconfig.xml" % (output_path, context.config("extent_prefix") )
    assert os.path.exists(config_file), "Config file does not exist: %s" % config_file
    change_params(config_path = config_file, 
                  output_path = config_file,
                  params = [
                        ("eqasim:networkCalibration.activate","true"),  # this will not calibrate, but just activate the module to use penalties and speed factors
                        ("eqasim:networkCalibration.objective","freespeed,penalty,agent,subpopulations"),
                        ("eqasim:networkCalibration.calibrate","false"),
                        ("eqasim:networkCalibration.costCalibration.activate","false"),
                        ("eqasim:networkCalibration.freespeedCalibration.activate","false"),
                        ("eqasim:alphaCalibration.filePath","cantonal_target_mode_shares.csv"),
                        ("eqasim.modeParametersPath",dmc_params),
                        ("eqasim.costParametersPath","cost_parameters.yml"),
                        ("eqasim:networkCalibration.countsFile",regional_counts_file),
                        ("eqasim:networkCalibration.freespeedCalibration.observedTripsFile",regionl_speeds_file),
                        ("eqasim:networkCalibration.costCalibration.specialRegionPath",get_regions_path(output_path, kind="penalty")),
                        ("eqasim:networkCalibration.freespeedCalibration.specialRegionPath",get_regions_path(output_path, kind="freespeed")),
                        ("ptZones.ptZonesFilePath", "gtfs_zones.csv"),
                        ("ptZones.sbbDistancesPath", "SBB_all_distances.csv"),
                        ("ptZones.pricingDescriptionPath", "pricingDescription.xml")
                        ] )
    return output_path