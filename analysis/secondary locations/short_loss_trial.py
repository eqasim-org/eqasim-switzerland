#!/usr/bin/env python3
"""Retrain only the distance-enabled short expert and reuse the completed weight sweep."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import json
import logging
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from synpp.pipeline import ConfigurationContext, get_stage_hash

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from cache_io import RunCache
from cohorts import population
from loss_sweep_tools import (
    ASSIGNMENT, MODELS, SECONDARY, FrozenCache, execute_stages,
    source_fingerprint, stamp, verify_completed, verify_stamp, write_json,
)
from loss_sweep_report import replay_errors, write_comparison_report
from replay import prepare_inputs, replay
from run import replay_input_fingerprint

LOG = logging.getLogger('secondary_short_loss_trial')
SHORT_MODEL = MODELS[-1]
REFERENCES = (.1, .2, .3, .5)


class ShortTrialCache(FrozenCache):
    execution_stages = (SHORT_MODEL, ASSIGNMENT)

    def __init__(self, plan):
        super().__init__(plan)
        self.weight = plan['short_distribution_weight']
        self.options.update(
            secondary_nn_distribution_loss_weight=.1,
            secondary_nn_short_distribution_loss_weight=plan['short_distribution_weight'],
            secondary_nn_short_relative_distance_loss_weight=plan['short_relative_weight'],
            secondary_nn_use_target_distance=True,
            overwrite_short_range_model_if_exists=True)
        kind = plan.get('short_loss_kind', 'crps_energy')
        self.options.update(
            secondary_nn_short_distance_loss_kind=kind,
            secondary_nn_short_relative_underestimation_factor=plan.get('short_underestimation_factor', 1.),
            short_range_model_validation_strategy=plan.get('validation_strategy', 'fixed'),
            short_range_model_min_epochs=plan.get('min_epochs', 0),
            short_range_model_validation_folds=plan.get('validation_folds', 5))
        if kind == 'relative':
            self.options['secondary_nn_distance_loss_weight'] = plan['short_relative_weight']
            self.options['secondary_nn_short_relative_distance_loss_weight'] = 0.
        self.variant = Path(plan['output']) / 'trial'

    def stage(self, name, retain=False):
        actual = ASSIGNMENT if name == SECONDARY else name
        if actual in self.execution_stages:
            directory = self.stage_directory(actual)
            verify_completed(directory, self.plan, self.weight)
            with (directory / 'result.p').open('rb') as stream:
                result = pickle.load(stream)
            record = json.loads((directory / 'completed.json').read_text())
            self.used[name] = dict(result_pickle=stamp(directory / 'result.p'),
                                   focused_stage_outputs=record['outputs'])
            return result
        return RunCache.stage(self, name, retain=retain)

    def provenance(self):
        result = super().provenance()
        result.update(scope='Short-range expert only; baseline hierarchy remains at 0.1',
                      hierarchy_distribution_weight=.1,
                      short_distribution_weight=self.plan['short_distribution_weight'],
                      short_relative_weight=self.plan['short_relative_weight'])
        result.update(short_loss_kind=self.plan.get('short_loss_kind', 'crps_energy'),
                      validation_strategy=self.plan.get('validation_strategy', 'fixed'),
                      minimum_training_epochs=self.plan.get('min_epochs', 0))
        return result


def preflight(plan):
    cache = ShortTrialCache(plan)
    cache.verify()
    for name in cache.execution_stages:
        configured = ConfigurationContext(cache.options)
        importlib.import_module(name).configure(configured)
        for entry in configured.required_stages:
            if entry['descriptor'] not in cache.execution_stages:
                cache.resolve(entry['descriptor'])


def create_plan(sweep, output, distribution, relative, threads, *, loss_kind='crps_energy',
                validation_strategy='fixed', min_epochs=0, folds=5, reference_focused=()):
    if not np.isfinite([distribution, relative]).all() or min(distribution, relative) < 0:
        raise ValueError('Loss coefficients must be finite and nonnegative')
    if threads < 1:
        raise ValueError('Threads must be positive')
    if loss_kind not in ('relative', 'crps_energy') or validation_strategy not in ('fixed', 'cross_validation'):
        raise ValueError('Invalid short loss or validation strategy')
    if min_epochs < 0 or folds < 2:
        raise ValueError('Minimum epochs must be nonnegative and CV needs at least two folds')
    if loss_kind == 'relative' and distribution != 0:
        raise ValueError('Legacy-only trials require distribution weight zero')
    original = json.loads((sweep / 'plan.json').read_text())
    manifest = json.loads((sweep / 'comparison_manifest.json').read_text())
    if set(manifest['runs']) != {f'{w:g}' for w in REFERENCES}:
        raise ValueError('The reference report must contain 0.1, 0.2, 0.3 and 0.5')
    if manifest['baseline_root'] != original['baseline_root']:
        raise ValueError('Reference report and frozen inputs have different baseline roots')
    for record in original['frozen_files']:
        verify_stamp(record)

    # Only the short expert is updated; the three saved hierarchy stages stay fixed.

    baseline = FrozenCache(original)
    for name in MODELS[:-1]:
        recorded = baseline.meta[baseline.resolve(name)]
        if recorded.get('module_hash') != get_stage_hash(importlib.import_module(name)):
            raise RuntimeError('Unchanged hierarchy stage differs from the reference: ' + name)
    plan = deepcopy(original)
    label = f'short ({distribution:g}, {relative:g})'
    if loss_kind == 'relative':
        label = f'short legacy ({distribution:g}, {relative:g})'
    if validation_strategy == 'cross_validation':
        label += f', CV{folds}'
    elif min_epochs:
        label += f', min{min_epochs}'
    plan.update(
        kind='short_range_legacy' if loss_kind == 'relative' else 'short_range_hybrid',
        output=str(output), parent_sweep=str(sweep),
        source_fingerprint=source_fingerprint(), created_utc=time.time(),
        short_distribution_weight=distribution, short_relative_weight=relative,
        hierarchy_distribution_weight=.1, threads=threads,
        label=label, short_loss_kind=loss_kind, validation_strategy=validation_strategy,
        min_epochs=min_epochs, validation_folds=folds,
        reference_focused=[str(path.resolve()) for path in reference_focused])
    plan['configuration']['config'].update(
        threads=threads, secondary_nn_distribution_loss_weight=.1,
        secondary_nn_short_distribution_loss_weight=distribution,
        secondary_nn_short_relative_distance_loss_weight=relative if loss_kind == 'crps_energy' else 0.,
        secondary_nn_short_distance_loss_kind=loss_kind,
        secondary_nn_short_relative_underestimation_factor=1.,
        short_range_model_validation_strategy=validation_strategy,
        short_range_model_min_epochs=min_epochs,
        short_range_model_validation_folds=folds)
    if loss_kind == 'relative':
        plan['configuration']['config']['secondary_nn_distance_loss_weight'] = relative
    references = [sweep / name for name in (
        'plan.json', 'comparison_manifest.json', 'survey.parquet',
        'replay_observed.parquet', 'replay_errors.parquet')]
    for weight in REFERENCES:
        references.extend(sweep / f'weight_{weight:g}' / name
                          for name in ['population.parquet', 'replay.parquet'])
    labels = {f'{w:g}' for w in REFERENCES} | {label}
    for path in reference_focused:
        recorded_plan = json.loads((path / 'plan.json').read_text())
        recorded = json.loads((path / 'comparison_manifest.json').read_text())
        if recorded['common_input_sha256'] != manifest['common_input_sha256']:
            raise ValueError('Focused reference used a different common survey input realization')
        if recorded_plan['label'] in labels:
            raise ValueError('Focused reference has a duplicate trial label')
        labels.add(recorded_plan['label'])
        for name in ['plan.json', 'comparison_manifest.json', 'replay_errors.parquet',
                     'trial/population.parquet', 'trial/replay.parquet']:
            references.append(path / name)
    plan['frozen_files'].extend(stamp(path) for path in references)
    plan['reference_manifest_sha256'] = hashlib.sha256(
        (sweep / 'comparison_manifest.json').read_bytes()).hexdigest()
    preflight(plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    return plan


def compare(plan):
    sweep = Path(plan['parent_sweep'])
    output = Path(plan['output'])
    original = json.loads((sweep / 'plan.json').read_text())
    previous = json.loads((sweep / 'comparison_manifest.json').read_text())
    cache = ShortTrialCache(plan)
    cache.verify()
    cohort = pd.read_parquet(sweep / 'replay_observed.parquet')
    inputs = prepare_inputs(FrozenCache(original), cohort)
    fingerprint = replay_input_fingerprint(inputs)
    if fingerprint != previous['common_input_sha256']:
        raise RuntimeError('Replay inputs differ from the completed reference comparison')

    # Reuse the completed populations and replay errors for all four reference weights.

    datasets = {'survey': pd.read_parquet(sweep / 'survey.parquet')}
    for weight in REFERENCES:
        datasets[f'{weight:g}'] = pd.read_parquet(sweep / f'weight_{weight:g}' / 'population.parquet')
    old_errors = pd.read_parquet(sweep / 'replay_errors.parquet')
    expected = len(cohort) * plan['draws']
    if set(old_errors.trial.unique()) != {f'{w:g}' for w in REFERENCES}:
        raise ValueError('Reference replay error table has unexpected trials')
    if not old_errors.groupby('trial').size().eq(expected).all():
        raise ValueError('Reference trials do not contain the expected number of draws')
    additional_settings = []
    for path in plan.get('reference_focused', []):
        path = Path(path)
        recorded_plan = json.loads((path / 'plan.json').read_text())
        recorded = json.loads((path / 'comparison_manifest.json').read_text())
        if recorded['common_input_sha256'] != fingerprint:
            raise ValueError('Focused reference inputs differ from the common cohort')
        label = recorded_plan['label']
        reference_errors = pd.read_parquet(path / 'replay_errors.parquet')
        reference_errors = reference_errors.loc[reference_errors.trial.eq(label)]
        if len(reference_errors) != expected:
            raise ValueError('Focused reference has an incomplete replay')
        datasets[label] = pd.read_parquet(path / 'trial/population.parquet')
        old_errors = pd.concat([old_errors, reference_errors], ignore_index=True)
        previous['runs'][label] = recorded['runs'][label]
        additional_settings.append(dict(trial=label, hierarchy_distribution=.1,
            short_distribution=recorded_plan['short_distribution_weight'],
            short_relative=recorded_plan['short_relative_weight'],
            short_loss_kind=recorded_plan.get('short_loss_kind', 'crps_energy'),
            relative_scope='origin + home' if recorded_plan.get('short_loss_kind') == 'relative' else 'origin only',
            short_validation=recorded_plan.get('validation_strategy', 'fixed'),
            minimum_training_epochs=recorded_plan.get('min_epochs', 0)))
    frame, audit = population(cache)
    cache.variant.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache.variant / 'population.parquet', index=False)
    datasets[plan['label']] = frame
    predictions, detour = replay(cache, cohort, inputs, plan['draws'],
                                 cache.config('random_seed'), min(plan['threads'], 4))
    keys = ['person_id', 'trip_id', 'draw']
    if len(predictions) != expected or predictions.duplicated(keys).any():
        raise ValueError('New replay must contain one prediction for every common trip/draw')
    if set(map(tuple, predictions[keys].to_numpy())) != set(map(tuple, old_errors.loc[old_errors.trial.eq('0.1'), keys].to_numpy())):
        raise ValueError('New and reference replay keys differ')
    predictions.to_parquet(cache.variant / 'replay.parquet', index=False)
    predictions['trial'] = plan['label']
    new_errors = replay_errors(predictions, cohort, inputs)
    del detour
    manifest = dict(
        baseline_root=cache.root, source_fingerprint=plan['source_fingerprint'],
        common_input_sha256=fingerprint, people=previous['people'], trips=len(cohort),
        draws=plan['draws'], reused_report=str(sweep / 'report.html'),
        runs={**previous['runs'], plan['label']: dict(audit=audit, provenance=cache.provenance())})
    settings = pd.DataFrame([
        dict(trial=f'{w:g}', hierarchy_distribution=w, short_distribution=w, short_relative=0.)
        for w in REFERENCES] + additional_settings + [dict(
            trial=plan['label'], hierarchy_distribution=.1,
            short_distribution=plan['short_distribution_weight'], short_relative=plan['short_relative_weight'],
            short_loss_kind=plan.get('short_loss_kind', 'crps_energy'),
            relative_scope='origin + home' if plan.get('short_loss_kind') == 'relative' else 'origin only',
            short_validation=plan.get('validation_strategy', 'fixed'),
            minimum_training_epochs=plan.get('min_epochs', 0))])
    settings['short_validation'] = settings.short_validation.fillna('fixed')
    settings['short_loss_kind'] = settings.short_loss_kind.fillna('crps_energy')
    settings['relative_scope'] = settings.relative_scope.fillna('none')
    settings['minimum_training_epochs'] = settings.minimum_training_epochs.fillna(0)
    note = (
        'The short-range pairs show (distribution weight, relative Huber weight). Legacy-only scoring uses the original origin and home distance terms; hybrid scoring adds an origin-only relative term. '
        'Only the distance-enabled short expert is retrained; its three hierarchy experts remain the saved 0.1 models. '
        'Reference labels 0.1/0.2/0.3/0.5 denote the earlier global distribution weights for all four experts, with no auxiliary relative term. '
        'Their saved population predictions and replay errors are reused. All trials use identical frozen inputs, survey people and draw seeds. '
        'The expert switch is still 1.5 km. Replay uses observed origins and is not independent held-out validation. '
        'Positive paired improvement means lower absolute distance error than the original 0.1 baseline. '
        'The new trial may also change validation and refitting; its differences cannot be attributed to loss alone. '
        'Independent person folds select duration with fixed upstream feature transforms; this is not unbiased full-pipeline validation.')
    training = None
    directory = cache.stage_directory(SHORT_MODEL)
    if (directory / 'loss_history.json').exists():
        history = json.loads((directory / 'loss_history.json').read_text())
        training = {key: history.get(key) for key in ('validation_mode', 'validation_folds',
            'early_stopping_min_epochs', 'epochs_completed', 'best_epoch', 'refit_epochs',
            'distance_loss_kind', 'distance_loss_weight', 'auxiliary_relative_distance_loss_weight')}
        training['effective_distribution_weight'] = (history['distance_loss_weight']
            if history['distance_loss_kind'] == 'crps_energy' else 0.)
        training['effective_relative_weight'] = (history['distance_loss_weight']
            if history['distance_loss_kind'] == 'relative' else history.get('auxiliary_relative_distance_loss_weight', 0.))
        manifest['new_short_training'] = training
        if all((directory / ('lossevolution.' + extension)).exists() for extension in ['png', 'pdf', 'svg']):
            (output / 'figures').mkdir(exist_ok=True)
            for extension in ['png', 'pdf', 'svg']:
                shutil.copy2(directory / ('lossevolution.' + extension),
                             output / 'figures' / ('05_training_validation.' + extension))
    cache.verify()
    write_comparison_report(datasets, pd.concat([old_errors, new_errors], ignore_index=True),
                            manifest, output, note, trial_settings=settings, training_summary=training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, default=Path(__file__).with_name('results') / 'loss_weight_sweep')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--distribution-weight', type=float, required=True)
    parser.add_argument('--relative-weight', type=float, required=True)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--loss-kind', choices=['crps_energy', 'relative'], default='crps_energy')
    parser.add_argument('--validation-strategy', choices=['fixed', 'cross_validation'], default='fixed')
    parser.add_argument('--min-epochs', type=int, default=0)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--reference-focused', type=Path, action='append', default=[])
    parser.add_argument('--phase', choices=['plan', 'all', 'fit', 'compare'], default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    output, sweep = args.output.resolve(), args.sweep.resolve()
    path = output / 'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        actual = (plan['short_distribution_weight'], plan['short_relative_weight'], plan['threads'], plan['parent_sweep'],
            plan.get('short_loss_kind', 'crps_energy'), plan.get('validation_strategy', 'fixed'),
            plan.get('min_epochs', 0), plan.get('validation_folds', 5), plan.get('reference_focused', []))
        requested = (args.distribution_weight, args.relative_weight, args.threads, str(sweep), args.loss_kind,
            args.validation_strategy, args.min_epochs, args.folds, [str(path.resolve()) for path in args.reference_focused])
        if actual != requested:
            raise ValueError('Arguments differ from the saved plan; choose another output directory')
        preflight(plan)
    else:
        if args.phase in ['fit', 'compare']:
            raise FileNotFoundError('Prepare the focused experiment first')
        plan = create_plan(sweep, output, args.distribution_weight, args.relative_weight, args.threads,
            loss_kind=args.loss_kind, validation_strategy=args.validation_strategy,
            min_epochs=args.min_epochs, folds=args.folds, reference_focused=args.reference_focused)
    LOG.info('Focused trial: %s; hierarchy frozen at 0.1', plan['label'])
    if args.phase == 'plan':
        return
    if args.phase == 'fit':
        cache = ShortTrialCache(plan)
        execute_stages(cache, cache.execution_stages)
        return
    if args.phase == 'compare':
        compare(plan)
        return

    # Release fitting/assignment memory before loading the comparison data.

    common = [sys.executable, str(Path(__file__).resolve()), '--sweep', str(sweep),
              '--output', str(output), '--distribution-weight', str(args.distribution_weight),
              '--relative-weight', str(args.relative_weight), '--threads', str(args.threads),
              '--loss-kind', args.loss_kind, '--validation-strategy', args.validation_strategy,
              '--min-epochs', str(args.min_epochs), '--folds', str(args.folds)]
    for path in args.reference_focused:
        common.extend(['--reference-focused', str(path.resolve())])
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase in ['fit', 'compare']:
            subprocess.run([*common, '--phase', phase], check=True, cwd=REPOSITORY)
    LOG.info('Focused comparison complete: %s', output / 'report.html')


if __name__ == '__main__':
    main()
