import os.path
import shutil


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
        "%sconfig.xml" % context.config("output_prefix")
    ]
    if context.config("estimate_dmc"):
        file_names.append("estimated_dmc_parameters.yml")
    
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
    shutil.move(path_to_results, 
                "%s/simulation_output" % target_path)
    # TODO: if calibration is activated, copy the calibrated parameters
    
    # copy contract information
    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    return {}
