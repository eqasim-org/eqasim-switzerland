"""Report a fitted production DMC model on its Microcensus estimation cohort.

Cached-only CLI: uv run -m analysis.dmc.run --config config_dib_10.yml
Synpp stage: analysis.dmc.run
"""
import argparse
import logging
import os
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


def configure(context):
    context.config('dmc_estimator', 'jax')
    context.config('analysis_path')
    context.config('random_seed')
    context.config('analysis.dmc.repetitions', 200)
    context.config('analysis.dmc.elasticity_step', .01)
    estimator = str(context.config('dmc_estimator')).strip().lower()
    if estimator not in ('jax', 'biogeme'):
        raise ValueError('dmc_estimator must be jax or biogeme')
    context.stage(f'dmc.models.model_{estimator}', alias='dmc_model')
    context.stage('dmc.data.training_data')
    context.stage('data.microcensus.trips')
    # Declare the same fit-defining defaults used by both estimators.
    context.config('ignore_car_passenger', False)
    context.config('use_exponents', True)
    context.config('use_income_in_dmc', True)
    context.config('distance_cost_interaction', True)
    context.config('income_cost_interaction', True)
    context.config('dmc_transfer_peak_km', 7.)
    context.config('use_weights_for_vot', False)
    context.config('dmc_vot_constraints', True)
    context.config('dmc_vot_car_target', 32.)
    context.config('dmc_vot_pt_target', 18.)
    from dmc.constants import constants
    context.config('reference_income_chf', constants.REF_INCOME_CHF)
    context.config('reference_euclidean_distance_km', constants.REF_EUCLIDEAN_DISTANCE_KM)


def execute(context):

    # Load the estimator's exact cohort and fitted/exported parameters.

    from . import data, report
    dataset = data.load(context)

    # Generate cached analysis artifacts without estimating any coefficients.

    source = Path(context.path())
    report.generate(context, dataset, source)

    # Export copies, preserving both the model cache and the analysis cache.

    destination = Path(context.config('analysis_path')) / 'dmc'
    if source.resolve() != destination.resolve():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    return str(destination / 'index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config_dib_10.yml'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('threads must be positive')
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.threads])
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    import yaml
    from . import data, report
    settings = yaml.safe_load(args.config.read_text())
    try:
        context = data.cached_context(settings)
        configure(context)
        dataset = data.load(context)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'DMC analysis: {error}\n')
    destination = args.output or Path(context.config('analysis_path')) / 'dmc'
    report.generate(context, dataset, destination)
    logger.info('Report: %s', destination / 'index.html')


if __name__ == '__main__':
    main()
