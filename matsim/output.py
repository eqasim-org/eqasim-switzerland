import os.path
import shutil


def configure(context):
    context.stage("matsim.simulation.run")
    context.config("output_path")
    context.config("output_id")
    context.stage("contracts.contracts")
    
    context.config("input_prefix", "output_")
    context.config("output_prefix", "switzerland_")

def execute(context):
    results_path = "%s/simulation_output" % context.stage("matsim.simulation.run")

    output_path = context.config("output_path")
    output_id = context.config("output_id")

    if not os.path.isdir(output_path):
        raise RuntimeError("Output path does not exist:", output_path)

    target_path = "%s/%s" % (output_path, output_id)

    if os.path.exists(target_path):
        if os.path.isdir(target_path):
            print("Cleaning target directory:", target_path)
            shutil.rmtree(target_path)
        else:
            raise RuntimeError("Cannot clean target path:", target_path)

    os.mkdir(target_path)

    for file in [
        "network.xml.gz",
        "transitSchedule.xml.gz",
        "transitVehicles.xml.gz",
        "facilities.xml.gz",
        "households.xml.gz",
        "plans.xml.gz",
        "config.xml"
    ]:
        shutil.copyfile("%s/%s%s" % (results_path, context.config("input_prefix"), file), 
                        "%s/%s%s" % (target_path, context.config("output_prefix"), file))

    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    return {}
