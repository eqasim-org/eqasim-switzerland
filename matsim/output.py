import os
import shutil
import glob
from matsim.config.tools import change_params, get_regions_path

def configure(context):
    context.stage("matsim.simulation.run")    
    context.stage("matsim.simulation.prepare")
    context.stage("contracts.contracts")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.cutter.run_scenario")

    context.config("output_path")
    context.config("output_id")    
    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", True)
    context.config("write_jar", True)
    context.config("calibrate_alphas_in_matsim", default=True)
    context.config("calibrate_betas_in_matsim", default=False)
    context.config("simulation_directory", 'simulation_output')
    context.config("extent_prefix", default="")



def execute(context):
    source_path = context.path("matsim.simulation.prepare")
    output_path = context.config("output_path")
    output_id = context.config("output_id")

    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    # create target path
    target_path = "%s/%s" % (output_path, output_id)

    if os.path.exists(target_path):
        if os.path.isdir(target_path):
            print("Cleaning target directory:", target_path)
            shutil.rmtree(target_path)
        else:
            raise RuntimeError("Cannot clean target path:", target_path)

    os.mkdir(target_path)
    
    # copy scenario files to from source path target path
    file_names = [
        "%shouseholds.xml.gz" % context.config("output_prefix"),
        "%spopulation.xml.gz" % context.config("output_prefix"),
        "%sfacilities.xml.gz" % context.config("output_prefix"),
        "%snetwork.xml.gz" % context.config("output_prefix"),
        "%stransit_schedule.xml.gz" % context.config("output_prefix"),
        "%stransit_vehicles.xml.gz" % context.config("output_prefix"),
        "%svehicles.xml.gz" % context.config("output_prefix"),
        "%sconfig.xml" % context.config("output_prefix"),
        "%sglobal_mode_shares.csv" % context.config("output_prefix"),
        "%scantonal_mode_shares.csv" % context.config("output_prefix"),
        "dmc_parameters.yml",
        "cost_parameters.yml",
        "SBB_all_distances.csv",
        "gtfs_zones.csv",
        "pricingDescription.xml"
    ]
    
    if context.config("export_detailed_network"):
        file_names.append("%sdetailed_network.csv" % context.config("output_prefix"))
    file_names.extend(glob.glob("network_calibration_files/*", root_dir=source_path))

    for file in file_names:
        destination = os.path.join(target_path, file)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(os.path.join(source_path, file), destination)

    # copy the jar file
    if context.config("write_jar"):
        shutil.copy(
            "%s/%s" % (context.path("matsim.runtime.eqasim"), context.stage("matsim.runtime.eqasim")),
            "%s/%srun.jar" % (target_path, context.config("output_prefix"))
        )
    
    # Copy results while preserving the simulation stage's cached files.
    path_to_results =  "%s/simulation_output" % context.path("matsim.simulation.run")
    new_path_to_results = "%s/%s" % (target_path, context.config("simulation_directory"))
    shutil.copytree(path_to_results, new_path_to_results)
    
    # if calibration is activated, copy the calibrated parameters
    dmc_params = "dmc_parameters.yml"
    if context.config("calibrate_alphas_in_matsim") or context.config("calibrate_betas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/*_parameters.yml" % new_path_to_results)
        if len(calibrated_parameters_path) > 0:            
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)
            shutil.copyfile(calibrated_parameters_path, "%s/calibrated_dmc_parameters.yml" % target_path)
            dmc_params = "calibrated_dmc_parameters.yml"

    # replace the config file with the real output config file
    output_config = "%s/output_config.xml" % new_path_to_results
    config_file = "%s/%sconfig.xml" % (target_path, context.config("output_prefix"))
    shutil.copyfile(output_config, config_file)

    # modify the config to remove absolute paths (to have a self contained directory of the scenario)
    change_params(config_path = config_file, 
                  output_path = config_file,
                  params = [("eqasim:alphaCalibration.filePath","%scantonal_mode_shares.csv" % context.config("output_prefix")),
                            ("eqasim.modeParametersPath",dmc_params),
                            ("eqasim.costParametersPath","cost_parameters.yml"),
                            ("eqasim:networkCalibration.countsFile", "network_calibration_files/target_counts.csv"
                             if os.path.isfile(os.path.join(target_path, "network_calibration_files/target_counts.csv")) else ""),
                            ("eqasim:networkCalibration.freespeedCalibration.observedTripsFile", "network_calibration_files/target_travel_times.csv"
                             if os.path.isfile(os.path.join(target_path, "network_calibration_files/target_travel_times.csv")) else ""),
                            ("eqasim:networkCalibration.costCalibration.specialRegionPath",get_regions_path(target_path, kind="penalty")),
                            ("eqasim:networkCalibration.freespeedCalibration.specialRegionPath",get_regions_path(target_path, kind="freespeed")),
                            ("ptZones.ptZonesFilePath", "gtfs_zones.csv"),
                            ("ptZones.sbbDistancesPath", "SBB_all_distances.csv"),
                            ("ptZones.pricingDescriptionPath", "pricingDescription.xml")
                            ] )

    # copy contract information
    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    # copy the regional model results too
    regional_model_scenario_path, regional_model_results_path = context.stage("matsim.cutter.run_scenario")

    if len(regional_model_scenario_path) and context.config("extent_prefix") != "":
        target_regional_model_path = "%s/%s%s" % (target_path, context.config("extent_prefix"), context.config("simulation_directory"))
        shutil.copytree(regional_model_scenario_path, target_regional_model_path)

        # Copy separately stored regional results into the exported scenario.
        target_regional_results_path = "%s/simulation_output" % target_regional_model_path
        if (len(regional_model_results_path) and 
            not os.path.exists(target_regional_results_path) and 
            os.path.exists(regional_model_results_path)):
            shutil.copytree(regional_model_results_path, target_regional_results_path)

        # replace the config file (output_config contains all the parameters used in the simulation)
        config_file = "%s/%sconfig.xml" % (target_regional_model_path, context.config("extent_prefix") )
        output_config_file = "%s/output_config.xml" %target_regional_results_path
        assert os.path.exists(output_config_file), "Output config file does not exist: %s" % output_config_file
        if os.path.exists(config_file): os.remove(config_file)
        shutil.copyfile(output_config_file, config_file)
        # make sure all calibrations are deactivated
        change_params(config_path = config_file, 
                      output_path = config_file,
                      params = [
                            ("eqasim:calibration.activate", "false"),
                            ("eqasim:calibration.runCalibration","false"),
                            ("eqasim:alphaCalibration.activate","false"),
                            ("eqasim:networkCalibration.activate","true"),  # this will not calibrate, but just activate the module to use penalties and speed factors
                            ("eqasim:networkCalibration.calibrate","false"),
                            ("eqasim:networkCalibration.objective",""),
                            ("eqasim:networkCalibration.costCalibration.activate","false"),
                            ("eqasim:networkCalibration.freespeedCalibration.activate","false"),
                            ("eqasim:networkCalibration.agentAscsCalibration.activate","false"),
                            ("eqasim:networkCalibration.subpopulationsCalibration.activate","false"),
                            ("eqasim:networkCalibration.subpopulationsCalibration.calibrateCrossBorder","false"),
                            ] )

    return {}
