import os.path
import shutil


def configure(context):
    context.stage("matsim.run")
    context.config("output_path")
    context.config("output_id")
    context.stage("contracts.contracts")

def execute(context):
    results_path = context.stage("matsim.run")

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
        "switzerland_network.xml.gz",
        "switzerland_transit_schedule.xml.gz",
        "switzerland_transit_vehicles.xml.gz",
        "switzerland_facilities.xml.gz",
        "switzerland_households.xml.gz",
        "switzerland_population.xml.gz",
        "switzerland_config.xml"
    ]:
        shutil.copyfile("%s/%s" % (results_path, file), "%s/%s" % (target_path, file))

    contracts_path = context.stage("contracts.contracts")
    shutil.copyfile(contracts_path, "%s/CONTRACTS.html" % target_path)

    return {}
