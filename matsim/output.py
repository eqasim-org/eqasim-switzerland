from multiprocessing import context
import os.path
import shutil
import glob

def configure(context):
    context.stage("matsim.simulation.run")    
    context.stage("matsim.simulation.prepare")
    context.stage("contracts.contracts")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.scenario.network.convert_osm")
    context.stage("matsim.cutter.run_scenario")

    context.config("output_path")
    context.config("output_id")    
    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", False)
    context.config("write_jar", True)
    context.config("calibrate_alphas_in_matsim", default=False)
    context.config("calibrate_betas_in_matsim", default=False)
    context.config("simulation_directory", "simulation_output")
    context.config("extent_prefix", default="")

def execute(context):
    source_path = context.path("matsim.simulation.prepare")
    output_path = context.config("output_path")
    output_id = context.config("output_id")

    if not os.path.isdir(output_path):
        raise RuntimeError("Output path does not exist:", output_path)

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
        "cost_parameters.yml"
    ]
    
    for file in file_names:
        shutil.copyfile("%s/%s" % (source_path, file), 
                        "%s/%s" % (target_path, file))

    # copy the detailed geometry
    if context.config("export_detailed_network"):
        shutil.copy(
            "%s/%s" % (context.path("matsim.scenario.network.convert_osm"), "detailed_network.csv"),
            "%s/%s" % (target_path, "%sdetailed_network.csv" % context.config("output_prefix"))
        )
    
    # copy the jar file
    if context.config("write_jar"):
        shutil.copy(
            "%s/%s" % (context.path("matsim.runtime.eqasim"), context.stage("matsim.runtime.eqasim")),
            "%s/%srun.jar" % (target_path, context.config("output_prefix"))
        )
    
    # move the results to the output
    path_to_results =  "%s/simulation_output" % context.path("matsim.simulation.run")
    new_path_to_results = "%s/%s" % (target_path, context.config("simulation_directory"))
    shutil.move(path_to_results, new_path_to_results)
    
    # replce the config file with the real output config file
    output_config = "%s/output_config.xml" % new_path_to_results
    shutil.copyfile(output_config, "%s/%sconfig.xml" % (target_path, context.config("output_prefix")))

    # if calibration is activated, copy the calibrated parameters
    if context.config("calibrate_alphas_in_matsim") or context.config("calibrate_betas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/*_parameters.yml" % new_path_to_results)
        if len(calibrated_parameters_path) > 0:            
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)
            shutil.copyfile(calibrated_parameters_path, 
                            "%s/calibrated_dmc_parameters.yml" % target_path)
        
    # copy contract information
    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    # copy the regional model results too
    regional_model_results_path, regional_model_scenario_path = context.stage("matsim.cutter.run_scenario")
    if len(regional_model_scenario_path) > 0 and context.config("extent_prefix") != "":
        target_regional_model_path = "%s/%s%s" % (target_path, context.config("extent_prefix"), context.config("simulation_directory"))
        shutil.move(regional_model_scenario_path, target_regional_model_path)

        target_regional_results_path = "%s/%s%s/simulation_output" % (target_path, context.config("extent_prefix"), context.config("simulation_directory"))
        shutil.move(regional_model_results_path, target_regional_results_path)

        # replace the config file (output_config contains all the parameters used in the simulation)
        config_file = "%s/%sconfig.xml" % (target_regional_model_path, context.config("extent_prefix") )
        output_config_file = "%s/output_config.xml" %target_regional_results_path
        assert os.path.exists(output_config_file), "Output config file does not exist: %s" % output_config_file
        os.replace(output_config_file, config_file)

    return {}
