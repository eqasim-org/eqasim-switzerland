"""Generate the shared MATSim config and adapt local routing/simulation copies."""
from pathlib import Path

import matsim.runtime.eqasim as eqasim
from dmc.constants import constants as dmc_constants
from matsim.config import tools


def configure(context):
    context.stage("matsim.runtime.java")
    context.stage("matsim.runtime.eqasim")
    context.stage("calibration.car_routing_vot.optimal_value")
    context.config("input_downsampling")
    context.config("output_prefix", "switzerland_")
    context.config("random_seed")
    context.config("threads", volatile=True)
    context.config("use_freight", True)
    context.config("car_cost_model", dmc_constants.CAR_COST_MODEL)
    context.config("route_bike", True)
    context.config("useScheduleBasedTransport", False)


def execute(context):
    path = str(Path(context.path()) / f"{context.config('output_prefix')}config.xml")
    eqasim.run(context, "org.eqasim.core.scenario.config.RunGenerateConfig", [
        "--output-path", path,
        "--prefix", context.config("output_prefix"),
        "--sample-size", context.config("input_downsampling"),
        "--random-seed", context.config("random_seed"),
        "--threads", context.config("threads"),
        "--eqasim-configurator", "org.eqasim.switzerland.ch_cmdp.SwitzerlandConfigurator",
    ])
    eqasim.run(context, "org.eqasim.switzerland.ch_cmdp.scenario.RunAdaptConfig", [
        "--input-path", path, "--output-path", path,
        "--downsamplingRate", context.config("input_downsampling"),
        "--replanningRate", "0.05",
        "--hasFreight", tools.java_boolean(context.config("use_freight")),
        "--prefix", context.config("output_prefix"),
        "--carCostModel", context.config("car_cost_model").lower(),
        "--routingDistanceUtility", context.stage("calibration.car_routing_vot.optimal_value"),
        "--routeBikeInNetwork", tools.java_boolean(context.config("route_bike")),
    ])
    if not tools.parse_boolean(context.config("useScheduleBasedTransport")):
        tools.add_SBBPT_module(path)
    tools.add_ptZones_module(path)
    tools.change_params(path, {"eqasim.useScheduleBasedTransport": tools.parse_boolean(
        context.config("useScheduleBasedTransport"))}, create=True)
    return path


def adapt_for_routing(context, config_path):
    """Set local pricing files and PT parameters before routing any trips."""
    tools.add_ptZones_module(config_path, context.path())
    tools.adjust_pt_routing_parameters(
        config_path, context.stage("calibration.pt_routing.pt_routing_parameters"))


def adapt_for_simulation(context, config_path):
    """Finish a prepared scenario's config; run only needs to launch Java."""
    adapt_for_routing(context, config_path)
    tools.change_params(config_path, tools.simulation_parameters(context), create=True)

    return str(Path(config_path).resolve())
