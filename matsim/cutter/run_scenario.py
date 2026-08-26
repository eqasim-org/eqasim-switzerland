import os
import logging
import matsim.runtime.eqasim as eqasim

logger = logging.getLogger("synpp:    Regional Model\t")

def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.cutter.scenario")
    
    context.config("use_vdf", default=False)
    context.config("extent_path", default="")
    context.config("extent_prefix", default="")
    context.config("last_iteration", default = 60)
    context.config("regional_model_last_iteration", default = context.config("last_iteration"))
    context.config("run_regional_model", default=True)


def execute(context):
    # 1. get the regional scenario if exists
    regional_scenario = context.stage("matsim.cutter.scenario")
    if regional_scenario=="":
        return "", ""
    
    assert os.path.exists(regional_scenario), "Regional scenario does not exist: %s" % regional_scenario
    logger.info("Regional scenario found: %s" % regional_scenario)

    # 2. go to the regional scenario dir (this makes sure all paths are working)
    cwd = os.getcwd()
    os.chdir(regional_scenario)
    logger.info("Changed working directory to regional scenario: %s" % os.getcwd())

    # 3. config file
    config_file = "%s/%sconfig.xml" % (regional_scenario, context.config("extent_prefix") )
    assert os.path.exists(config_file), "Config file does not exist: %s" % config_file
    logger.info("Config file found: %s" % config_file)

    
    # 4. run the simulation or return the path
    if not context.config("run_regional_model"):
        return regional_scenario, ""
    
    last_iteration = int(context.config("regional_model_last_iteration"))
    if not context.config("use_vdf"):
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunSimulation", [
            "--config-path", config_file,
            "--config:controler.lastIteration", str(last_iteration),
            ], cwd = regional_scenario)
    else:
        eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.RunVDFSimulation", [
            "--config-path", config_file,
            "--config:controler.lastIteration", str(last_iteration),
            "--generateNetworkEvents", "true",
            ], cwd = regional_scenario)

    simulation_path = "%s/simulation_output" % regional_scenario
    assert os.path.exists(simulation_path), "Simulation output path does not exist: %s" % simulation_path
    assert os.path.exists("%s/output_events.xml.gz" % simulation_path), "Output events file does not exist: %s" % ("%s/output_events.xml.gz" % simulation_path)
    os.chdir(cwd)

    return regional_scenario, simulation_path
