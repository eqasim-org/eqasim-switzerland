import os.path
import shutil
import glob

def configure(context):
    context.stage("matsim.simulation.run")
    context.stage("matsim.simulation.prepare")
    context.stage("contracts.contracts")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.scenario.network.convert_osm")

    context.config("output_path")
    context.config("output_id")    
    context.config("output_prefix", "switzerland_")
    context.config("export_detailed_network", False)
    context.config("write_jar", True)
    context.config("estimate_dmc", default=False)
    context.config("calibrate_alphas_in_matsim", default=False)
    context.config("simulation_directory", "simulation_output")

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
        "%scantonal_mode_shares.csv" % context.config("output_prefix")
    ]
    if context.config("estimate_dmc"):
        file_names.append("estimated_dmc_parameters.yml")
        file_names.append("dmc_cost_parameters.yml")
    
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
    
    # if calibration is activated, copy the calibrated parameters
    if context.config("calibrate_alphas_in_matsim"):
        calibrated_parameters_path = glob.glob("%s/*_parameters.yml" % new_path_to_results)
        if len(calibrated_parameters_path) > 0:            
            calibrated_parameters_path = max(calibrated_parameters_path, key=os.path.getctime)

        shutil.copyfile(calibrated_parameters_path, 
                        "%s/calibrated_mode_parameters.yml" % target_path)
        
    # copy contract information
    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    return {}
