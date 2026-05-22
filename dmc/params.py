import os
import yaml

def configure(context):
    if context.config("estimate_dmc"):
        context.stage("dmc.model")
    else:
        context.stage("calibration.dmc.default")

def execute(context):
    if context.config("estimate_dmc"):
        _,_,(mode_params_path, cost_params_path), _, _ = context.stage("dmc.model")
    else:
        mode_params_path, cost_params_path = context.stage("calibration.dmc.default")

    return mode_params_path, cost_params_path
