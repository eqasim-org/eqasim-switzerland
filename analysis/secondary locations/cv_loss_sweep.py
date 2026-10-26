#!/usr/bin/env python3
"""Compare all short-expert losses with identical independent person-fold training."""
import argparse
import fcntl
import html
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from loss_sweep_report import write_comparison_report
from loss_sweep_tools import stamp, verify_stamp, write_json
from metrics import mean_ci
from short_loss_trial import create_plan, preflight, REPOSITORY

DIRECTORY = Path(__file__).resolve().parent
CASES = [('proper_0.1', .1, 0.), ('proper_0.2', .2, 0.),
         ('proper_0.3', .3, 0.), ('proper_0.5', .5, 0.), ('hybrid', .05, .05)]
LOG = logging.getLogger(__name__)


def trial_command(sweep, output, distribution, relative, threads):
    return [sys.executable, str(DIRECTORY / 'short_loss_trial.py'), '--sweep', str(sweep),
            '--output', str(output), '--distribution-weight', str(distribution),
            '--relative-weight', str(relative), '--threads', str(threads),
            '--validation-strategy', 'cross_validation', '--min-epochs', '20', '--folds', '5']


def prepare(sweep, legacy, output, threads):

    # Pin the existing legacy result and create independently resumable new trials.

    output.mkdir(parents=True, exist_ok=True)
    identity = dict(sweep=str(sweep), legacy=str(legacy), threads=threads)
    path = output / 'sweep_plan.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['arguments'] != identity:
            raise ValueError('Arguments differ from the saved CV sweep')
        for record in saved['references']:
            verify_stamp(record)
    else:
        original = json.loads((sweep / 'comparison_manifest.json').read_text())
        old = json.loads((legacy / 'comparison_manifest.json').read_text())
        legacy_plan = json.loads((legacy / 'plan.json').read_text())
        history = json.loads((legacy / 'trial/stages/short_range_model/loss_history.json').read_text())
        if old['common_input_sha256'] != original['common_input_sha256']:
            raise ValueError('Legacy trial has different replay inputs')
        if (legacy_plan['short_loss_kind'], legacy_plan['short_distribution_weight'],
            legacy_plan['short_relative_weight']) != ('relative', 0., .07):
            raise ValueError('Expected the completed old-loss-only trial')
        validate_training(history)
        names = ['plan.json', 'comparison_manifest.json', 'replay_errors.parquet',
                 'trial/population.parquet', 'trial/stages/short_range_model/loss_history.json']
        saved = dict(arguments=identity, references=[stamp(legacy / name) for name in names])
        hybrid = legacy.parent / 'short_loss_0.05_0.05'
        saved['references'].extend(stamp(hybrid / name) for name in
            ['plan.json', 'comparison_manifest.json', 'replay_errors.parquet'])
        write_json(path, saved)
    for name, distribution, relative in CASES:
        directory = output / name
        if (directory / 'plan.json').exists():
            plan = json.loads((directory / 'plan.json').read_text())
            if (plan['short_distribution_weight'], plan['short_relative_weight'], plan['threads'],
                plan['validation_strategy'], plan['min_epochs'], plan['validation_folds'],
                plan['parent_sweep']) != (distribution, relative, threads, 'cross_validation', 20, 5, str(sweep)):
                raise ValueError('Existing trial settings differ: ' + name)
            preflight(plan)
        else:
            create_plan(sweep, directory, distribution, relative, threads,
                        validation_strategy='cross_validation', min_epochs=20, folds=5)


def validate_training(history):
    if (history['validation_mode'], history['validation_folds'], history['early_stopping_min_epochs']) != ('independent_kfold', 5, 20):
        raise ValueError('Every compared model must use five independent person folds and minimum 20 epochs')
    if history['epochs_completed'] < 20 or not history['refit_full'] or history['refit_epochs'] != history['best_epoch']:
        raise ValueError('Training did not complete exploration and full-data refitting')


def before_after(old, new):

    # Pair draws within the same short trips; hierarchy differences cannot affect this scope.

    keys = ['person_id', 'trip_id', 'draw']
    columns = keys + ['absolute_error_km', 'spatial_error_km', 'weight', 'observed_km']
    old = old.loc[old.observed_km.between(.01, 1.5), columns]
    new = new.loc[new.observed_km.between(.01, 1.5), columns]
    pair = old.merge(new, on=keys, suffixes=('_fixed', '_cv'), validate='one_to_one')
    if len(pair) != len(old) or len(pair) != len(new):
        raise ValueError('Before/after short-trip keys differ')
    pair['weight'] = pair.weight_cv
    pair['improvement'] = pair.absolute_error_km_fixed - pair.absolute_error_km_cv
    mean, _, low, high = mean_ci(pair, 'improvement')
    return dict(fixed_mae_m=np.average(pair.absolute_error_km_fixed, weights=pair.weight)*1000,
                cv_mae_m=np.average(pair.absolute_error_km_cv, weights=pair.weight)*1000,
                improvement_m=mean*1000, low_m=low*1000, high_m=high*1000,
                fixed_endpoint_error_m=np.average(pair.spatial_error_km_fixed, weights=pair.weight)*1000,
                cv_endpoint_error_m=np.average(pair.spatial_error_km_cv, weights=pair.weight)*1000)


def compare(sweep, legacy, output):

    # Read saved predictions only; all fitting and replay run in separate processes.

    baseline_manifest = json.loads((sweep / 'comparison_manifest.json').read_text())
    fixed_errors = pd.read_parquet(sweep / 'replay_errors.parquet')
    datasets = {'survey': pd.read_parquet(sweep / 'survey.parquet'),
                '0.1': pd.read_parquet(sweep / 'weight_0.1/population.parquet')}
    errors = [fixed_errors.loc[fixed_errors.trial.eq('0.1')]]
    manifest = dict(baseline_manifest, runs={'0.1': baseline_manifest['runs']['0.1']},
                    scope='Short expert only; all hierarchy models frozen at 0.1', training_trials={})
    settings, training, paired = [], [], []
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    directories = [output / name for name, _, _ in CASES] + [legacy]
    expected = baseline_manifest['trips'] * baseline_manifest['draws']
    baseline_keys = pd.MultiIndex.from_frame(errors[0][['person_id', 'trip_id', 'draw']])
    for index, directory in enumerate(directories):
        plan = json.loads((directory / 'plan.json').read_text())
        record = json.loads((directory / 'comparison_manifest.json').read_text())
        label = plan['label']
        if record['common_input_sha256'] != baseline_manifest['common_input_sha256']:
            raise ValueError('CV trial used different replay inputs')
        frame = pd.read_parquet(directory / 'replay_errors.parquet')
        frame = frame.loc[frame.trial.eq(label)]
        keys = pd.MultiIndex.from_frame(frame[['person_id', 'trip_id', 'draw']])
        if len(frame) != expected or not keys.is_unique or len(keys.difference(baseline_keys)):
            raise ValueError('Incomplete or mismatched CV replay')
        history_path = directory / 'trial/stages/short_range_model/loss_history.json'
        history = json.loads(history_path.read_text())
        validate_training(history)
        datasets[label] = pd.read_parquet(directory / 'trial/population.parquet')
        errors.append(frame)
        manifest['runs'][label] = record['runs'][label]
        manifest['training_trials'][label] = dict(report=str(directory / 'report.html'), history=stamp(history_path))
        settings.append(dict(trial=label, hierarchy_distribution=.1,
            short_distribution=plan['short_distribution_weight'], short_relative=plan['short_relative_weight'],
            short_loss_kind=plan['short_loss_kind'],
            relative_scope=('origin + home' if plan['short_loss_kind']=='relative' else
                            'origin only' if plan['short_relative_weight'] else 'none'),
            short_validation='cross_validation', minimum_training_epochs=20))
        training.append(dict(trial=label, **{k: history[k] for k in ['epochs_completed', 'best_epoch',
            'refit_epochs', 'best_val_loss', 'distance_loss_kind', 'distance_loss_weight']}))
        if index < len(CASES):
            fixed_label = f'{plan["short_distribution_weight"]:g}'
            if plan['short_relative_weight']:
                previous_dir = legacy.parent / 'short_loss_0.05_0.05'
                previous_plan = json.loads((previous_dir / 'plan.json').read_text())
                previous_manifest = json.loads((previous_dir / 'comparison_manifest.json').read_text())
                if previous_manifest['common_input_sha256'] != baseline_manifest['common_input_sha256']:
                    raise ValueError('Hybrid reference inputs differ')
                previous_errors = pd.read_parquet(previous_dir / 'replay_errors.parquet')
                previous_errors = previous_errors.loc[previous_errors.trial.eq(previous_plan['label'])]
            else:
                previous_errors = fixed_errors.loc[fixed_errors.trial.eq(fixed_label)]
            paired.append(dict(trial=label, **before_after(previous_errors, frame)))
        for extension in ['png', 'pdf', 'svg']:
            shutil.copy2(history_path.parent / ('lossevolution.' + extension),
                         figures / f'{5+index:02d}_training_{index+1}.{extension}')
    note = ('All six CV cases use independent five-fold person validation, minimum 20 exploration epochs, '
            'and full-data refitting for the selected epoch count. Only the distance-enabled short expert changes; '
            'every hierarchy remains at weight 0.1. The old-loss CV result is reused. The original fixed-validation '
            '0.1 case is shown as a reference. Other historical global weights had different hierarchy models, '
            'so their before/after comparisons are restricted to observed short trips (10 m–1.5 km). '
            'Inputs and replay draw seeds are identical. Replay uses observed origins on fitting records, '
            'not an independent held-out test. Person-cluster intervals exclude fitting uncertainty. '
            'Loss values across loss functions are not directly comparable.')
    settings.insert(0, dict(trial='0.1', hierarchy_distribution=.1, short_distribution=.1,
                           short_relative=0., short_loss_kind='crps_energy', relative_scope='none',
                           short_validation='fixed', minimum_training_epochs=0))
    write_comparison_report(datasets, pd.concat(errors, ignore_index=True), manifest, output, note,
                            trial_settings=pd.DataFrame(settings), training_summary=pd.DataFrame(training))
    paired = pd.DataFrame(paired)
    paired.to_csv(output / 'tables/before_after_short_errors.csv', index=False)
    path = output / 'report.html'
    table = '<h2>Effect of CV training on short trips</h2><p>Positive improvement means lower distance error; intervals cluster by person.</p>'
    table += paired.to_html(index=False, float_format=lambda x: f'{x:.1f}')
    path.write_text(path.read_text().replace('<h2>Population distances</h2>', table + '<h2>Population distances</h2>'))
    LOG.info('Completed controlled CV comparison: %s', path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, default=DIRECTORY/'results/loss_weight_sweep')
    parser.add_argument('--legacy', type=Path, default=DIRECTORY/'results/short_loss_legacy_cv5')
    parser.add_argument('--output', type=Path, default=DIRECTORY/'results/short_loss_cv_sweep')
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--phase', choices=['plan', 'all', 'compare'], default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sweep, legacy, output = (p.resolve() for p in [args.sweep, args.legacy, args.output])
    prepare(sweep, legacy, output, args.threads)
    if args.phase == 'plan':
        return
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.phase == 'all':
            for name, distribution, relative in CASES:
                LOG.info('Running CV loss trial: %s', name)
                subprocess.run(trial_command(sweep, output/name, distribution, relative, args.threads),
                               check=True, cwd=REPOSITORY)
        compare(sweep, legacy, output)


if __name__ == '__main__':
    main()
