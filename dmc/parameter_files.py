"""Resolve fitted artifacts for the current DMC baseline."""
import os

from dmc.parameters import Parameters


def configure(context):
    context.config('estimate_dmc', True)
    context.config('dmc_estimator', 'jax')
    context.config('dmc_mode_parameters_path', '')
    context.config('dmc_cost_parameters_path', '')
    mode_path = context.config('dmc_mode_parameters_path')
    cost_path = context.config('dmc_cost_parameters_path')
    if bool(mode_path) != bool(cost_path):
        raise ValueError('Provide both dmc_mode_parameters_path and dmc_cost_parameters_path')
    if mode_path:
        return
    if not context.config('estimate_dmc'):
        raise ValueError('Provide current fitted DMC parameter paths when estimate_dmc is false')
    estimator = str(context.config('dmc_estimator')).strip().lower()
    if estimator not in ('biogeme', 'jax'):
        raise ValueError("dmc_estimator must be either 'biogeme' or 'jax'")
    context.stage(f'dmc.models.model_{estimator}', alias='dmc_model')


def execute(context):

    # Resolve supplied artifacts or the selected estimator's fitted exports.

    if context.config('dmc_mode_parameters_path'):
        paths = (context.config('dmc_mode_parameters_path'), context.config('dmc_cost_parameters_path'))
    else:
        _, _, paths, _, _ = context.stage('dmc_model')

    # Validate the baseline format without changing the upstream artifacts.

    parameters = Parameters.from_yaml(*paths)
    if parameters.get('utilityModelVersion') != 2:
        raise ValueError('DMC parameter files must use the current baseline; re-estimate or provide current fitted artifacts')
    return paths


def validate(context):
    """Track changes to explicitly supplied fitted artifacts without estimating."""
    paths = [context.config('dmc_mode_parameters_path'), context.config('dmc_cost_parameters_path')]
    for path in filter(None, paths):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Fitted DMC parameter file does not exist: {path}')
    return [os.stat(path).st_mtime_ns for path in paths if path]
