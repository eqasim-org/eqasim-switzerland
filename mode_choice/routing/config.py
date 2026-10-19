"""A private router config using the existing fitted DMC cost parameters."""
from pathlib import Path
from matsim.config.tools import copy_config

from mode_choice.defaults import ROUTER_CONFIG_INPUTS


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("matsim.config.pt")
    context.stage("dmc.parameter_files")


def execute(context):
    source = Path(context.stage("matsim.config.pt")).resolve()
    _, cost_parameters = context.stage("dmc.parameter_files")
    target = Path(context.path()) / "router_config.xml"
    return copy_config(source, target,
                       params={"eqasim.costParametersPath": str(Path(cost_parameters).resolve())},
                       input_paths=ROUTER_CONFIG_INPUTS)
