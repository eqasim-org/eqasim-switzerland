"""Analysis-only, household-held-out search for the seven behavioural loss weights."""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.model_selection import GroupShuffleSplit

from synthesis.population.matching.V2 import model as model_stage
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.matching.V2.wrappers import DiaryMatcher
from synthesis.population.matching import matched_v1, matched_v3, opening_hours
from .metrics import diary_outcomes, outcome_scale, score_ensembles
from .validate_algorithms import FoldContext

KEYS = ('activity', 'trip_count', 'mode', 'departure', 'distance', 'duration', 'activity_duration')


def configure(context):
    model_stage.configure(context)
    matched_v3.configure(context)
    context.config('analysis_path')
    context.config('matching_hpo_trials', 20)
    context.config('matching_hpo_draws', 50)


def execute(context):

    # Prepare inputs and independent household groups for fitting, selection and testing.

    source, population, trips = prepare_inputs(context, context.stage('data.constants'))
    source, _ = opening_hours.add_regions(context, source, population)
    seed = int(context.config('random_seed'))
    options = replace(MatchingOptions.from_context(context), overwrite=False)
    root = Path(context.config('analysis_path')) / 'matching' / 'loss_weight_search'
    root.mkdir(parents=True, exist_ok=True)
    count, draws = int(context.config('matching_hpo_trials')), int(context.config('matching_hpo_draws'))
    if count < 2 or not 2 <= draws <= 50:
        raise ValueError('Require at least two trials and 2..50 draws')
    groups = source.get('household_id', source.mz_id).fillna(source.mz_id)
    development, test = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(source, groups=groups))
    fit_local, tune_local = next(GroupShuffleSplit(1, test_size=.25, random_state=seed + 1).split(
        source.iloc[development], groups=groups.iloc[development]))
    fit, tune = development[fit_local], development[tune_local]
    membership = source[['mz_id']].copy()
    membership['split'] = 'fit'
    membership.loc[tune, 'split'] = 'selection'
    membership.loc[test, 'split'] = 'test'
    membership.to_csv(root / 'membership.csv', index=False)
    observed, labels = diary_outcomes(source, trips)
    _, scaled, _ = outcome_scale(observed, fit, source.person_weight.to_numpy())
    base = np.array([options.training['loss_weights'][k] for k in KEYS])
    candidates = np.vstack([base, base * np.exp(qmc.LatinHypercube(len(KEYS), seed=seed).random(count - 1)
                                               * np.log(16) - np.log(4))])
    donors = source.iloc[fit].reset_index(drop=True)
    donor_trips = trips.loc[trips.person_id.isin(donors.mz_id)]
    target = source.iloc[tune].reset_index(drop=True).assign(person_id=source.iloc[tune].mz_id.to_numpy())
    records = []

    # Compare candidates with identical seeds, donor availability and unweighted score definitions.

    for trial, values in enumerate(candidates):
        directory = root / f'trial_{trial:03d}'
        directory.mkdir(exist_ok=True)
        weights = dict(zip(KEYS, values.tolist()))
        trial_options = replace(options, training={**options.training,
            'loss_weights': {**options.training['loss_weights'], **weights}})
        started = time.perf_counter()
        matcher = DiaryMatcher(trial_options, cache_directory=directory).fit(donors, donor_trips)
        selected = matcher.match(target, n_draws=draws, random_seed=seed + 2).donor_rows
        global_rows = np.where(selected >= 0, fit[np.maximum(selected, 0)], -1)
        scores = score_ensembles(observed, labels, tune, global_rows, scaled)
        weights_survey = source.iloc[tune].person_weight.to_numpy()
        coverage = float(np.average(scores.covered, weights=weights_survey))
        objective = (float(np.average(scores.energy, weights=weights_survey))
                     if coverage == 1 and np.isfinite(scores.energy).all() else float('inf'))
        record = dict(trial=trial, energy=objective, coverage=coverage,
                      seconds=time.perf_counter() - started, **weights)
        records.append(record)
        pd.DataFrame(records).to_csv(root / 'trials.csv', index=False)
        print('Completed loss-weight trial', record, flush=True)
    best = min(records, key=lambda r: r['energy'])
    if not np.isfinite(best['energy']):
        raise ValueError('No trial attained complete coverage and finite scores')
    selected_weights = {key: best[key] for key in KEYS}
    tuned = replace(options, training={**options.training, 'loss_weights': {
        **options.training['loss_weights'], **selected_weights}})

    # Test once on untouched households, using the same development donors for every method.

    donors = source.iloc[development].reset_index(drop=True)
    donor_trips = trips.loc[trips.person_id.isin(donors.mz_id)]
    target = source.iloc[test].reset_index(drop=True).assign(person_id=source.iloc[test].mz_id.to_numpy())
    _, scaled, _ = outcome_scale(observed, development, source.person_weight.to_numpy())
    all_scores = []
    raw = context.stage('data.microcensus.persons')
    raw = raw.loc[raw.person_id.isin(donors.mz_id)]
    chains = context.stage('data.microcensus.activity_chains')
    lookup = pd.Series(np.arange(len(donors)), index=donors.mz_id)
    for name, settings, module in [('BSLM current', options, None), ('BSLM optimized', tuned, None),
                                    ('RHDM', None, matched_v1), ('OHDM', None, matched_v3)]:
        directory = root / 'test' / name.replace(' ', '_')
        directory.mkdir(parents=True, exist_ok=True)
        if module is None:
            matcher = DiaryMatcher(settings, cache_directory=directory).fit(donors, donor_trips)
            rows = matcher.match(target, n_draws=draws, random_seed=seed + 3).donor_rows
        else:
            rows = np.full((len(target), draws), -1, dtype=int)
            for draw in range(draws):
                proxy = FoldContext(context, raw, target, donor_trips, chains, seed + 3 + draw, directory / str(draw))
                mapping, removed = module.execute(proxy)
                ids = target.person_id.map(mapping.set_index('person_id').mz_person_id)
                rows[:, draw] = ids.mask(target.person_id.isin(removed)).map(lookup).fillna(-1).to_numpy(dtype=int)
        np.savez_compressed(directory / 'donor_rows.npz', rows=rows)
        global_rows = np.where(rows >= 0, development[np.maximum(rows, 0)], -1)
        scores = score_ensembles(observed, labels, test, global_rows, scaled)
        scores['method'], scores['mz_id'] = name, target.person_id.to_numpy()
        scores['survey_weight'] = source.iloc[test].person_weight.to_numpy()
        all_scores.append(scores)
    pd.concat(all_scores, ignore_index=True).to_csv(root / 'test_scores.csv', index=False)
    (root / 'selection.json').write_text(json.dumps(dict(best_trial=best['trial'], weights=selected_weights,
        trials=count, draws=draws, seed=seed, fit=len(fit), selection=len(tune), test=len(test),
        objective='Survey-weighted multivariate energy score; lower is better',
        search='Baseline plus seeded log-scale Latin hypercube, 0.25x..4x each baseline weight',
        interpretation='Finite search, not a guarantee of global optimality'), indent=2))

    # Refit on the complete survey for the separate donor-reconstruction/population application.

    started = time.perf_counter()
    final = DiaryMatcher(tuned, cache_directory=root / 'full_model').fit(source, trips)
    final.training_wall_seconds_ = time.perf_counter() - started
    return final
