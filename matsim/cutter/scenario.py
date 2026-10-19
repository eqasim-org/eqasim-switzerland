"""Assemble a runnable regional scenario from the cutter's raw output."""
import os
import shutil
import glob
import logging
from matsim.config.tools import change_params, get_regions_path
from .cutter_tools import cut_csv_to_region, cut_csv_to_network, cut_and_save_network_geometry
import matsim.config.tools as config_tools
from .file_paths import FilePaths, simulation_root

logger = logging.getLogger("synpp")


def configure(context):
    context.config("cutter.scenario_path", default="")
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")
    if not context.config("extent_path") or not context.config("extent_prefix"):
        return
    
    context.stage("matsim.cutter.runCutter")
    if not context.config("cutter.scenario_path"):
        context.stage("matsim.simulation.prepare")
        context.stage("matsim.simulation.run")
        context.stage("matsim.scenario.network.mapped")
        context.stage("data.pt_pricing.pt_pricing")
        context.stage("data.microcensus.shares")    

    context.config("network_calibration.activate", default=True)
    context.config("network_calibration.calibrate_disutilities", default=True)
    context.config("network_calibration.calibrate_freespeed", default=True)
    context.config("network_calibration.calibrate_agents_ascs", default=True)
    context.config("network_calibration.calibrate_subpopulations", default=True)

    need_counts = config_tools.need_counts_file(context)
    if not context.config("cutter.scenario_path") and context.config("network_calibration.activate") and need_counts:
        context.stage("analysis.counts.target")
        if context.config("network_calibration.calibrate_disutilities"):
            context.stage("calibration.road_regions.penalty_calibration")


    if not context.config("cutter.scenario_path") and context.config("network_calibration.activate") and context.config("network_calibration.calibrate_freespeed"):
        context.stage("analysis.travel_times.APIs.target")
        context.stage("calibration.road_regions.freespeed_calibration")

    context.config("calibrate_alphas_in_matsim", default=True)
    context.config("calibrate_betas_in_matsim", default=False)
    context.config("export_detailed_network", True)


def execute(context):
    if not context.config("extent_path") or not context.config("extent_prefix"):
        logger.warning("Extent path or extent prefix is not set. Cutter will not be executed.")
        return ""
    
    files = FilePaths(context)
    cut_output = context.stage("matsim.cutter.runCutter")
    output_path = os.path.join(context.path(), "output")
    # Keep the raw cut stage immutable when adding files and editing config.
    shutil.copytree(cut_output, output_path, dirs_exist_ok=True)

    # Copy supporting files to make the scenario self contained.
    # 1. mode parameters
    shutil.copyfile(files.get("dmc_parameters.yml"),
                    "%s/dmc_parameters.yml" % output_path)
    shutil.copyfile(files.get("cost_parameters.yml"),
                    "%s/cost_parameters.yml" % output_path)
    dmc_params = "dmc_parameters.yml"
    if context.config("calibrate_alphas_in_matsim") or context.config("calibrate_betas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/%s/*_parameters.yml" % (simulation_root(context), "simulation_output"))
        if len(calibrated_parameters_path) > 0:
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)
            shutil.copyfile(calibrated_parameters_path,
                            "%s/calibrated_dmc_parameters.yml" % output_path)
            dmc_params = "calibrated_dmc_parameters.yml"

    # 2.mode shares
    global_shares_output_path, cantonal_shares_output_path = (files.get("global_shares"), files.get("cantonal_shares"))
    if global_shares_output_path:
        shutil.copy(global_shares_output_path, f"{output_path}/global_target_mode_shares.csv" )
    shutil.copy(cantonal_shares_output_path, f"{output_path}/cantonal_target_mode_shares.csv" )

    # 3. pricing parameters
    sbb_path, zones_path, pricing_path = (files.get(name) for name in
                                          ("SBB_all_distances.csv", "gtfs_zones.csv", "pricingDescription.xml"))
    shutil.copy(sbb_path, f"{output_path}/SBB_all_distances.csv" )
    shutil.copy(zones_path, f"{output_path}/gtfs_zones.csv" )
    shutil.copy(pricing_path, f"{output_path}/pricingDescription.xml" )

    # 4. calibration files (special regions and target files)
    calibrate_network, calibrate_disutilities, calibrate_freespeed, calibrate_agent_acs, calibrate_subpopulations = config_tools.get_network_cal_config_as_bool(context)
    need_counts = config_tools.need_counts_file(context)

    regionl_speeds_file = ""
    if calibrate_network and calibrate_freespeed:
        region_dir = os.path.join(output_path, "network_calibration_files")
        os.makedirs(region_dir, exist_ok=True)

        freespeed_calibration_path = files.regions("freespeed")
        if freespeed_calibration_path!="":
            regions = freespeed_calibration_path.split(";")
            for i,region in enumerate(regions):
                if os.path.exists(region):
                    shutil.copy(region, f"{region_dir}/freespeed_special_region_{i}.yml" )

        target_traveltimes_file = files.get("travel_times")
        if target_traveltimes_file!="":
            cut_csv_to_region(csv_path= target_traveltimes_file,
                              region_path = context.config("extent_path"),
                              output_path = f"{region_dir}/target_travel_times.csv")
            if os.path.exists(f"{region_dir}/target_travel_times.csv"):
                    regionl_speeds_file = "network_calibration_files/target_travel_times.csv"

    regional_counts_file = ""
    if need_counts:
        region_dir = os.path.join(output_path, "network_calibration_files")
        os.makedirs(region_dir, exist_ok=True)

        target_counts_file = files.get("counts")
        if target_counts_file!="":
            cut_csv_to_network(csv_path = target_counts_file, context=context,
                               network_path = "%s/%snetwork.xml.gz" % (output_path, context.config("extent_prefix")),
                               output_path = f"{region_dir}/target_counts.csv")
            if os.path.exists(f"{region_dir}/target_counts.csv"):
                regional_counts_file = "network_calibration_files/target_counts.csv"

        if calibrate_network and calibrate_disutilities:
            penalty_calibration_path = files.regions("penalty")
            if penalty_calibration_path!="":
                regions = penalty_calibration_path.split(";")
                for i,region in enumerate(regions):
                    if os.path.exists(region):
                        shutil.copy(region, f"{region_dir}/penalties_special_region_{i}.yml" )

    # 5. copy the detailed geometry
    network_path = "%s/%snetwork.xml.gz" % (output_path, context.config("extent_prefix"))
    geometry_path = "%s/%snetwork_geometry.csv" % (output_path, context.config("extent_prefix"))
    if context.config("export_detailed_network"):
        geometry_file = files.get("geometry")
        cut_and_save_network_geometry(
            context=context,
            geometry = geometry_file,
            local_network = network_path,
            output_path = geometry_path
        )

    # 6. modify the regional model config
    config_file = "%s/%sconfig.xml" % (output_path, context.config("extent_prefix") )
    assert os.path.exists(config_file), "Config file does not exist: %s" % config_file
    change_params(config_path = config_file,
                  output_path = config_file,
                  params = [
                        ("eqasim:networkCalibration.activate","true"),  # this will not calibrate, but just activate the module to use penalties and speed factors
                        ("eqasim:networkCalibration.objective", config_tools.get_network_calibration_objectives(context)),
                        ("eqasim:networkCalibration.calibrate", config_tools.get_network_calibration_calibrate(context, ignore=("penalty","freespeed"))), # we do not calibrate penalty and freespeed here
                        ("eqasim:networkCalibration.costCalibration.activate","false"),
                        ("eqasim:networkCalibration.freespeedCalibration.activate","false"),
                        ("eqasim:alphaCalibration.filePath","cantonal_target_mode_shares.csv"),
                        ("eqasim.modeParametersPath",dmc_params),
                        ("eqasim.costParametersPath","cost_parameters.yml"),
                        ("eqasim:networkCalibration.countsFile",regional_counts_file),
                        ("eqasim:networkCalibration.freespeedCalibration.observedTripsFile",regionl_speeds_file),
                        ("eqasim:networkCalibration.costCalibration.specialRegionPath",get_regions_path(output_path, kind="penalty")),
                        ("eqasim:networkCalibration.freespeedCalibration.specialRegionPath",get_regions_path(output_path, kind="freespeed")),
                        ("eqasim:networkCalibration.agentAscsCalibration.gridRebuildMaxPopulations","1000,800,300"),
                        ("eqasim:networkCalibration.agentAscsCalibration.gridRebuildMinCellSizes","800,500,100"),
                        ("eqasim:networkCalibration.agentAscsCalibration.veryHighCountWeight","1.2"),
                        ("eqasim:networkCalibration.agentAscsCalibration.highCountWeight","1.1"),
                        ("eqasim:networkCalibration.agentAscsCalibration.mediumCountWeight","1.05"),
                        ("eqasim:networkCalibration.agentAscsCalibration.lowCountWeight","1.0"),
                        ("ptZones.ptZonesFilePath", "gtfs_zones.csv"),
                        ("ptZones.sbbDistancesPath", "SBB_all_distances.csv"),
                        ("ptZones.pricingDescriptionPath", "pricingDescription.xml")
                        ] )
    return output_path
