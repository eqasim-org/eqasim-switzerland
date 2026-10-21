import os
import logging
import shutil
from pathlib import Path
import matsim.runtime.eqasim as eqasim

logger = logging.getLogger("synpp:    Regional Model\t")

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.cutter.scenario")
    
    context.config("use_vdf", default=False)
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")
    context.config("last_iteration", default = 140)
    context.config("regional_model_last_iteration", default = 2 * context.config("last_iteration"))
    context.config("run_regional_model", default=True)
    context.config("cutter.output_path", default="")
    context.config("java_memory", "50G", volatile=True)


def execute(context):
    # 1. get the regional scenario if exists
    regional_scenario = context.stage("matsim.cutter.scenario")
    if regional_scenario=="":
        return "", ""
    
    assert os.path.exists(regional_scenario), "Regional scenario does not exist: %s" % regional_scenario
    logger.info("Regional scenario found: %s" % regional_scenario)

    # Copy the assembled scenario before running in the requested directory.
    regional_scenario = str(Path(regional_scenario).resolve())
    destination = context.config("cutter.output_path")
    if destination:
        destination = Path(destination).expanduser().resolve()
        source = Path(regional_scenario)
        if destination != source:
            if source in destination.parents:
                raise ValueError("cutter.output_path must not be inside the source scenario directory")
            shutil.copytree(source, destination, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("simulation_output"))
        regional_scenario = str(destination)
    logger.info("Regional scenario directory: %s", regional_scenario)

    # 3. config file
    config_file = "%s/%sconfig.xml" % (regional_scenario, context.config("extent_prefix") )
    assert os.path.exists(config_file), "Config file does not exist: %s" % config_file
    logger.info("Config file found: %s" % config_file)

    
    # 4. run the simulation or return the path
    if not context.config("run_regional_model"):
        return regional_scenario, ""
    
    simulation_path = os.path.join(regional_scenario, "simulation_output")
    last_iteration = int(context.config("regional_model_last_iteration"))
    if not context.config("use_vdf"):
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_file,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.outputDirectory", simulation_path,
            ], cwd = regional_scenario)
    else:
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunVDFSimulation", [
            "--config-path", config_file,
            "--config:controler.lastIteration", str(last_iteration),
            "--config:controler.outputDirectory", simulation_path,
            "--generateNetworkEvents", "true",
            ], cwd = regional_scenario)

    assert os.path.exists(simulation_path), "Simulation output path does not exist: %s" % simulation_path
    assert os.path.exists("%s/output_events.xml.gz" % simulation_path), "Output events file does not exist: %s" % ("%s/output_events.xml.gz" % simulation_path)

    return regional_scenario, simulation_path
