import os.path
import shutil
import matsim.runtime.eqasim as eqasim

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.config("use_vdf", default=False)
    context.config("input_downsampling")
    context.config("use_freight")
    context.config("output_prefix", "switzerland_")

    context.config("useScheduleBasedTransport", default=True)
    context.config("preventwaitingtoentertraffic", default = "no")
    context.config("writeexperiencedplans", default = "no")

    context.config("simulation_inputs_path", default = "")
    context.config("experiment_name", default = "experiment")
    context.config("simulation_config_path", default = "switzerland_config.xml")


def execute(context):

    scheduleBasedPTconfig = "false"
    if context.config("useScheduleBasedTransport"):
        scheduleBasedPTconfig = "true"
        print("Schedule-based PT: " + scheduleBasedPTconfig)

    preventwaitingtoentertraffic = "n"
    if context.config("preventwaitingtoentertraffic"):
        preventwaitingtoentertraffic = "y"
        print("Prevent waiting to enter traffic: " + preventwaitingtoentertraffic)

    writeExperiencedPlans = "false"
    if context.config("writeexperiencedplans"):
        writeExperiencedPlans = "true"
        print("Write experienced plans: " + writeExperiencedPlans)

    input_path  = context.config("simulation_inputs_path")

    output_path = os.path.join(context.path(), context.config("experiment_name"))
    print(output_path)
    os.makedirs(output_path, exist_ok=True)

    config = context.config("simulation_config_path")
    origin      = os.path.join(input_path, config)
    destination = os.path.join(output_path, config)
    shutil.copy2(origin, destination)

    prefix = "switzerland"
    for file in ["facilities.xml.gz", "households.xml.gz", "network.xml.gz", "population.xml.gz",
                 "transit_schedule.xml.gz", "transit_vehicles.xml.gz", "vehicles.xml.gz"]:
        origin      = os.path.join(input_path, prefix + "_" + file)
        destination = os.path.join(output_path, prefix + "_" + file)
        shutil.copy2(origin, destination)

    config_path = os.path.join(output_path, config)
    assert os.path.exists(config_path)

    eqasim.run(context, "org.eqasim.switzerland.ch.scenario.RunAdaptConfig", [
        "--input-path", config_path,
        "--output-path", config_path,
        "--downsamplingRate", context.config("input_downsampling"),
        "--replanningRate", "0.05",
        "--hasFreight", context.config("use_freight"),
        "--prefix", context.config("output_prefix")
    ])

    if (not context.config("use_vdf")):
        # Run simulation
        eqasim.run(context, "org.eqasim.switzerland.ch.RunSimulation", [
            "--config-path", config_path,
            "--config:controler.outputDirectory", os.path.join(output_path, "simulation_output"),
            "--config:controler.lastIteration", str(1),
            "--config:controler.writeEventsInterval", str(1),
            "--config:controler.writePlansInterval", str(1),
            #"--config:transitRouter.directWalkFactor", str(1.0),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
            "--preventwaitingtoentertraffic", preventwaitingtoentertraffic,
            "--config:scoring.writeExperiencedPlans", writeExperiencedPlans
        ])
    else:
        # Run simulation with vdf
        eqasim.run(context, "org.eqasim.switzerland.RunVDFSimulation", [
            "--config-path", config_path,
            "--config:controler.outputDirectory", os.path.join(output_path, "simulation_output"),
            "--config:controler.lastIteration", str(1),
            "--config:controler.writeEventsInterval", str(1),
            "--config:controler.writePlansInterval", str(1),
            "--config:eqasim.useScheduleBasedTransport", scheduleBasedPTconfig,
        ])
    assert os.path.exists("%s/simulation_output/output_events.xml.gz" % output_path)
    
    return context.path()