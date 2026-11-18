"""Four analysis-only architectural variants, with selection and household testing."""
import json
import hashlib
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from synthesis.population.matching.V2 import model as model_stage
from synthesis.population.matching.V2.configuration import MatchingOptions
from synthesis.population.matching.V2.features import FeatureProcessor
from synthesis.population.matching.V2.targets import SequenceTargetBuilder
from synthesis.population.matching.V2.training import split_training_rows
from synthesis.population.matching.V2.wrappers import DiaryMatcher
from synthesis.population.matching.V2.inputs import prepare_inputs
from synthesis.population.matching import opening_hours
from .ablation_model import count_targets
from .ablation_training import EmbeddingTrainer
from .metrics import diary_outcomes, outcome_scale, score_ensembles

VARIANTS = {'Current': (False, False), 'Counts': (True, False),
            'Features': (False, True), 'Counts + features': (True, True)}


def configure(context):
    model_stage.configure(context)
    context.config('analysis_path')
    context.config('matching_ablation_seeds', 3)
    context.config('matching_ablation_draws', 50)


def execute(context, *, variants=None, count_multiplier=1.0, output_name="architecture_ablation"):

    # Prepare a new household split and save membership for reproducible comparisons.

    variants = VARIANTS if variants is None else variants
    source, population, trips = prepare_inputs(context, context.stage('data.constants'))
    source, _ = opening_hours.add_regions(context, source, population)
    seed = int(context.config('random_seed')) + 202
    root = Path(context.config('analysis_path')) / 'matching' / output_name
    root.mkdir(parents=True, exist_ok=True)
    groups = source.get('household_id', source.mz_id).fillna(source.mz_id)
    development, test = next(GroupShuffleSplit(1, test_size=.2, random_state=seed).split(source, groups=groups))
    fit_local, selection_local = next(GroupShuffleSplit(1, test_size=.25, random_state=seed + 1).split(
        source.iloc[development], groups=groups.iloc[development]))
    fit, selection = development[fit_local], development[selection_local]
    membership = source[['mz_id']].copy()
    membership['household'] = groups.to_numpy()
    membership['split'] = 'fit'
    membership.loc[selection, 'split'] = 'selection'
    membership.loc[test, 'split'] = 'test'
    membership.to_csv(root / 'membership.csv', index=False)
    draws, repeats = int(context.config('matching_ablation_draws')), int(context.config('matching_ablation_seeds'))
    if not 2 <= draws <= 50 or not 1 <= repeats <= 5:
        raise ValueError('Require 2..50 draws and 1..5 fitting seeds')
    options = replace(MatchingOptions.from_context(context), overwrite=False)
    options = replace(options, training={**options.training, 'loss_weights': {
        **options.training['loss_weights'], 'trip_count': options.training['loss_weights']['trip_count'] * count_multiplier}})
    observed, labels = diary_outcomes(source, trips)
    manifest = dict(seed=seed, draws=draws, seeds=repeats, weights=options.training['loss_weights'],
        variants=variants, count_weight_multiplier=count_multiplier, fit=len(fit), selection=len(selection), test=len(test),
        count_order=['total trips', 'shopping', 'leisure', 'education', 'work'],
        count_loss='Total-trip loss plus mean of four purpose-count losses, times existing trip-count weight',
        reconstruction='Direct masked numerical and categorical reconstruction replaces projected-input reconstruction',
        interpretation='New household split; previous experiments informed this follow-up, not external validation')
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # Fit all variants with shared seeds, testing on recipients excluded from fitting.

    scores, timing = [], []
    for phase, donor_rows, recipient_rows in [('selection', fit, selection), ('test', development, test)]:
        donors = source.iloc[donor_rows].reset_index(drop=True)
        donor_trips = trips.loc[trips.person_id.isin(donors.mz_id)]
        target = source.iloc[recipient_rows].reset_index(drop=True).assign(person_id=source.iloc[recipient_rows].mz_id.to_numpy())
        _, scaled, _ = outcome_scale(observed, donor_rows, source.person_weight.to_numpy())
        for repetition in range(repeats):
            run_options = replace(options, random_seed=seed + repetition,
                training={**options.training, 'random_seed': seed + repetition})
            for name, (counts, features) in variants.items():
                directory = root / phase / name.replace(' ', '_') / str(repetition)
                directory.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                matcher = DiaryMatcher(run_options)
                matcher.store = ExperimentalStore(directory, counts, features)
                matcher.fit(donors, donor_trips)
                rows = matcher.match(target, n_draws=draws, random_seed=seed + 10000 + repetition).donor_rows
                global_rows = np.where(rows >= 0, donor_rows[np.maximum(rows, 0)], -1)
                np.savez_compressed(directory / 'donor_rows.npz', rows=global_rows,
                                    recipient_ids=target.person_id.to_numpy())
                scored = score_ensembles(observed, labels, recipient_rows, global_rows, scaled)
                scored['mz_id'], scored['method'] = target.person_id.to_numpy(), name
                scored['phase'], scored['seed'] = phase, repetition
                scored['survey_weight'] = source.iloc[recipient_rows].person_weight.to_numpy()
                scores.append(scored)
                timing.append(dict(phase=phase, method=name, seed=repetition, seconds=time.perf_counter()-started))
                pd.concat(scores, ignore_index=True).to_csv(root / 'scores.csv', index=False)
                pd.DataFrame(timing).to_csv(root / 'runtime.csv', index=False)
                (directory / 'training.json').write_text(json.dumps(matcher.training_report_, indent=2))
                print('Finished', phase, name, repetition, timing[-1], flush=True)
    return str(root)


class ExperimentalStore:
    def __init__(self, directory, counts, features):
        self.directory, self.counts, self.features = Path(directory), counts, features

    def fit_or_load(self, donors, trips, options):
        path = self.directory / 'model.pt'
        digest = hashlib.sha256((validate(None) + repr(options) + repr((self.counts, self.features))).encode())
        for frame in (donors, trips):
            digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        fingerprint = digest.hexdigest()
        if path.exists():
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload.get('fingerprint') == fingerprint:
                return payload['model'], payload['processor'], payload['report']
        train, validation = split_training_rows(donors, options.validation_fraction, options.random_seed)
        processor = FeatureProcessor().fit(donors.iloc[train])
        inputs = processor.transform(donors)
        targets = SequenceTargetBuilder(**options.targets).build(donors, trips)
        targets['trip_count_tgt'] = count_targets(donors, trips, options.targets['trip_count_scale'])
        targets['missing_categories'] = {k: mapping.get('<na>', -1) for k, mapping in processor.mappings_.items()}
        trainer = EmbeddingTrainer(**options.training, count_head=self.counts,
                                  explicit_reconstruction=self.features).fit(inputs, targets, train, validation)
        trainer.model_.cpu()
        payload = dict(model=trainer.model_, processor=processor, report=trainer.report_, fingerprint=fingerprint)
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        return payload['model'], processor, payload['report']


def validate(context):
    from synthesis.population.matching.V2.features import source_token
    from synthesis.population.matching.V2.artifacts import implementation_token
    from . import ablation_model, ablation_training, metrics
    return source_token(ablation_model, ablation_training, metrics) + implementation_token()
