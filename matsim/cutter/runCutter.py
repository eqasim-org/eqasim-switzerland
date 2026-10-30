"""Run the selected Java cutter; scenario assembly lives in scenario.py."""
import os.path
import matsim.runtime.eqasim as eqasim
from matsim.config.tools import change_params, resolve_output_file
from .file_paths import simulation_root


def configure(context):
    context.config("cutter.scenario_path", default="")
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")
    context.config("use_vdf", default=False)
    context.config("cutterVersion", default="v2" if context.config("use_vdf") else "v1")
    context.config("threads", volatile=True)
    if not context.config("extent_path") or not context.config("extent_prefix"):
        return
    
    cutter_version(context)
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.config("java_memory", "50G", volatile=True)
    
    # If this path is provided, we just use it to get the parent model and cut it
    if not context.config("cutter.scenario_path"):
        context.stage("matsim.simulation.run")    


def execute(context):
    if not context.config("extent_path") or not context.config("extent_prefix"):
        return ""
    version = cutter_version(context)
    
    # Read the supplied simulation or the pipeline simulation stage.
    p = simulation_root(context)
    simulation_path = os.path.join(p, "simulation_output")
    config_path = os.path.join(simulation_path, "output_config.xml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Simulation output config does not exist: {config_path}")

    # the output path, where the scenario is created
    output_path = "%s/output" % context.path()

    # change the path of all input files to point to the output files
    # population, households, facitilies, network, transit_schedule, transit_vehicles, vehicles
    cutter_config_path = "%s/config_cutter.xml" % context.path()
    simulation_inputs = {
        parameter: resolve_output_file(os.path.join(simulation_path, filename))
        for parameter, filename in {
            "plans.inputPlansFile": "output_plans.xml",
            "households.inputFile": "output_households.xml",
            "facilities.inputFacilitiesFile": "output_facilities.xml",
            "network.inputNetworkFile": "output_network.xml",
            "transit.transitScheduleFile": "output_transitSchedule.xml",
            "transit.vehiclesFile": "output_transitVehicles.xml",
            "vehicles.vehiclesFile": "output_vehicles.xml",
        }.items()
    }
    change_params(config_path = config_path,
                  output_path = cutter_config_path,
                  params = [*simulation_inputs.items(),
                            # Second: turn off calibration config
                            ("eqasim:calibration.activate", "false"),
                            ("eqasim:calibration.runCalibration","false"),
                            ("eqasim:alphaCalibration.activate","false"),
                            ("eqasim:alphaCalibration.filePath",""),
                            ("eqasim:networkCalibration.activate","false"),
                            ("eqasim:networkCalibration.calibrate",""),
                            ("eqasim:networkCalibration.countsFile",""),
                            ("eqasim:networkCalibration.costCalibration.specialRegionPath",""),
                            ("eqasim:networkCalibration.freespeedCalibration.observedTripsFile",""),
                            ("eqasim:networkCalibration.freespeedCalibration.specialRegionPath",""),
                            ] )


    arguments = [
        "--config-path", cutter_config_path,
        "--output-path", output_path,
        "--extent-path", context.config("extent_path"),
        "--threads", context.config("threads"),
        "--prefix", context.config("extent_prefix"),
    ]
    if version == "v1":
        main_class = "org.eqasim.core.scenario.cutter.RunScenarioCutter"
        arguments.extend([
            "--events-path", resolve_output_file(os.path.join(simulation_path, "output_events.xml")),
            "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator",
        ])
    else:
        main_class = "org.eqasim.core.scenario.cutter.RunScenarioCutterV2"
        arguments.extend([
            "--vdf-travel-times-path", os.path.join(p, "simulation_output", "vdf.bin"),
        ])
    eqasim.run(context, main_class, arguments)
    return output_path


################### Functions ####################

def cutter_version(context):
    version = str(context.config("cutterVersion")).lower().replace(".", "")
    if version not in {"v1", "v2"}:
        raise ValueError(f"Invalid cutter version: {context.config('cutterVersion')}")
    if version == "v2" and not context.config("use_vdf"):
        raise ValueError("VDF must be enabled for cutter version 2")
    return version
