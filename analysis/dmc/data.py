"""Load fitted artifacts and align their exact estimation cohort by trip key."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from dmc.models import model_jax
from dmc.models.data import prepare_sequences
from dmc.parameters import NAMES_CONVERSION, Parameters

KEYS = ['person_id', 'trip_id']
_MISSING = object()


class CachedContext:
    """Small read-only stage adapter for the cached-only command-line entry point."""
    def __init__(self, values, stages=None, paths=None):
        self.values = dict(values)
        self.stages = stages or {}
        self.paths = paths or {}

    def config(self, name, default=_MISSING, **kwargs):
        if name not in self.values:
            if default is _MISSING:
                raise ValueError(f'Missing configuration: {name}')
            self.values[name] = default
        return self.values[name]

    def stage(self, name, alias=None, **kwargs):
        return self.stages[name]

    def path(self, name):
        return str(self.paths[name])


@dataclass
class Dataset:
    frame: pd.DataFrame
    keys: pd.DataFrame
    survey_weights: np.ndarray
    modes: list[str]
    coefficients: dict
    declarations: dict
    export_references: dict
    audit: dict


def load(context):
    result, fitted_frame, paths, _, _ = context.stage('dmc_model')
    required = {'transfer_count_power', 'beta_pt_origin_quality'}
    if not context.config('ignore_car_passenger'):
        required.update({'beta_passenger_history', 'beta_passenger_long_distance'})
    missing_required = required.difference(result.get_beta_values())
    if missing_required:
        raise ValueError(f'The cached fit is missing current DMC coefficients {sorted(missing_required)}. Estimate the current model before generating its report.')
    training = context.stage('dmc.data.training_data')
    diary = context.stage('data.microcensus.trips')[0]
    ignore = context.config('ignore_car_passenger')
    frame, modes = model_jax.preprocess_data(training, ignore)
    try:
        pd.testing.assert_frame_equal(frame, fitted_frame, check_exact=True)
    except AssertionError as error:
        raise ValueError('The fitted model and current training/preprocessing data differ; re-estimate the current DMC model') from error
    selected = prepare_sequences(training, ignore)
    keys = selected[KEYS].copy()
    declarations = model_jax.define_parameters(
        ignore, context.config('use_exponents'), context.config('use_income_in_dmc'),
        income_reference=float(np.average(frame.income, weights=frame.person_weight)),
        transfer_peak_km=context.config('dmc_transfer_peak_km'),
        distance_interaction=context.config('distance_cost_interaction'),
        income_interaction=context.config('income_cost_interaction'))
    values = result.get_beta_values()
    missing = {name for name, p in declarations.items() if not p.fixed}.difference(values)
    unknown = set(values).difference(declarations)
    if missing or unknown:
        raise ValueError(f'Cached fit does not match the current DMC model: missing {sorted(missing)}, unknown {sorted(unknown)}. Estimate the current model first.')
    estimated = {name: float(values.get(name, p.init_value)) for name, p in declarations.items()}
    for name, p in declarations.items():
        value = estimated[name]
        if (not np.isfinite(value) or (p.lower_bound is not None and value < p.lower_bound - 1e-8)
                or (p.upper_bound is not None and value > p.upper_bound + 1e-8)):
            raise ValueError(f'Invalid fitted coefficient {name}: {value}')
    optimization = getattr(result, 'optimization', None)
    if optimization is not None and not getattr(optimization, 'gradient_converged', False):
        raise ValueError('The cached JAX fit did not pass its convergence checks')
    if hasattr(result, 'algorithm_has_converged') and not result.algorithm_has_converged:
        raise ValueError('The cached Biogeme fit did not converge')
    parameters = Parameters.from_yaml(*paths)
    if parameters.get('utilityModelVersion') != 2:
        raise ValueError('Fitted exports must use the current DMC model format')
    applied = {name: float(parameters[NAMES_CONVERSION[name]]) for name in declarations}

    # Original survey weights serve the MZ comparisons; fit weights stay separate.
    original = keys.merge(diary[KEYS + ['weight_person']], on=KEYS, how='left', validate='1:1', sort=False)
    weights = original.weight_person.to_numpy(float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Original MZ person weights are missing or nonpositive in the estimation cohort')
    day_counts = diary.groupby('person_id').size()
    selected_counts = keys.groupby('person_id').size()
    last = diary.groupby('person_id').trip_id.max()
    audit = {
        'training_trips_before_sequence_filter': len(training), 'reported_trips': len(frame),
        'people': int(keys.person_id.nunique()),
        'complete_days': int(selected_counts.eq(day_counts.reindex(selected_counts.index)).sum()),
        'retained_final_trips': int(keys.trip_id.eq(keys.person_id.map(last)).sum()),
        'excluded_training_trips': len(training) - len(frame),
        'mode_parameters': str(Path(paths[0]).resolve()), 'cost_parameters': str(Path(paths[1]).resolve()),
        'artifact_sha256': {Path(path).name: fingerprint(path) for path in paths},
        'fit_weight_sum': float(frame.person_weight.sum()), 'survey_weight_sum': float(weights.sum()),
        'in_sample': True, 'new_estimation_performed': False,
        'fitted_log_likelihood': float(-optimization.fun) if optimization is not None else
            float(result.raw_estimation_results.final_log_likelihood),
        'estimator': context.config('dmc_estimator'),
        'fitting_frame_sha256': hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest(),
        'trip_keys_sha256': hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest(),
    }
    return Dataset(frame, keys, weights, modes, {'estimated': estimated, 'exported': applied}, declarations,
                   {'reference_income_chf': parameters['referenceIncome'],
                    'reference_euclidean_distance_km': parameters['referenceEuclideanDistance_km']}, audit)


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def cached_context(settings):
    """Resolve completed stage results only; never invoke Synpp or an estimator."""
    root = Path(settings['working_directory'])
    metadata = json.loads((root / 'pipeline.json').read_text())
    configuration = dict(settings['config'])
    estimator = str(configuration.get('dmc_estimator', 'jax')).lower().strip()
    if estimator not in ('jax', 'biogeme'):
        raise ValueError('dmc_estimator must be jax or biogeme')
    model_name = f'dmc.models.model_{estimator}'
    candidates = [key for key in metadata if key.startswith(model_name + '__') and (root / (key + '.p')).is_file()]
    relevant = {'ignore_car_passenger', 'use_exponents', 'use_income_in_dmc', 'distance_cost_interaction',
                'income_cost_interaction', 'dmc_transfer_peak_km', 'dmc_vot_constraints',
                'dmc_vot_car_target', 'dmc_vot_pt_target', 'use_weights_for_vot'}
    candidates = [key for key in candidates if all(
        configuration[name] == metadata[key]['config'][name]
        for name in relevant & configuration.keys() & metadata[key]['config'].keys())]
    if not candidates:
        raise ValueError(f'No completed matching {model_name} cache. Estimate the model before running cached-only analysis.')
    model_key = max(candidates, key=lambda key: metadata[key]['updated'])

    def dependency(parent, stage):
        matches = [key for key in metadata[parent]['dependencies'] if key.split('__')[0] == stage]
        if len(matches) != 1:
            raise ValueError(f'Expected one {stage} dependency of {parent}')
        return matches[0]

    training_key = dependency(model_key, 'dmc.data.training_data')
    diary_key = dependency(training_key, 'data.microcensus.trips')
    stages = {}; paths = {}
    for name, key in [('dmc_model', model_key), ('dmc.data.training_data', training_key), ('data.microcensus.trips', diary_key)]:
        with (root / (key + '.p')).open('rb') as stream:
            stages[name] = pickle.load(stream)
        paths[name] = root / (key + '.cache')
    stages[model_name] = stages['dmc_model']
    # Fit-defining defaults come from the cached estimation, not today's defaults.
    values = {**metadata[model_key]['config'], **configuration}
    return CachedContext(values, stages, paths)
