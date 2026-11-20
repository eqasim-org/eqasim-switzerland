"""Audit saved secondary assignments without loading or running neural models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(frame):
    weights = frame.weight.to_numpy()
    return dict(draws=len(frame), trips=len(frame[['person_id', 'trip_id']].drop_duplicates()),
                weight=weights.sum(), observed_km=np.average(frame.observed, weights=weights),
                predicted_km=np.average(frame.distance_km, weights=weights),
                mae_km=np.average(abs(frame.distance_km-frame.observed), weights=weights),
                bias_km=np.average(frame.distance_km-frame.observed, weights=weights),
                short_probability=np.average(frame.distance_km <= 1.5, weights=weights))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    root = args.results
    output = root/'investigation'
    output.mkdir(exist_ok=True)

    # Join repeated predictions to their common observed survey trips.

    observed = pd.read_parquet(root/'replay_observed.parquet')
    predictions = pd.read_parquet(root/'replay_predictions.parquet')
    data = predictions.merge(observed[['person_id', 'trip_id', 'distance_km', 'destination_x', 'destination_y']].rename(
        columns={'distance_km': 'observed', 'destination_x': 'observed_x', 'destination_y': 'observed_y'}),
        on=['person_id', 'trip_id'], validate='many_to_one')
    bins = [0, .01, .1, .25, .7, 1.5, 1.6, 1.8, 3, 5, 20, np.inf]
    data['observed_band'] = pd.cut(data.observed, bins, right=False)
    rows = []
    for keys, group in data.groupby(['dataset', 'observed_band'], observed=True):
        rows.append(dict(dataset=keys[0], observed_band=str(keys[1]), **summarize(group)))
    pd.DataFrame(rows).to_csv(output/'replay_distance_bands.csv', index=False)

    # Distance errors alone can hide choosing the opposite side of the origin.

    rows = []
    for keys, group in data.groupby(['dataset', 'observed_band'], observed=True):
        endpoint = np.hypot(group.destination_x-group.observed_x, group.destination_y-group.observed_y)/1000
        rows.append(dict(dataset=keys[0], observed_band=str(keys[1]),
                         mean_endpoint_error_km=np.average(endpoint, weights=group.weight)))
    pd.DataFrame(rows).to_csv(output/'replay_endpoint_errors.csv', index=False)

    # Separate hierarchy routing decisions from within-expert distance errors.

    rows = []
    routed = data.loc[data.dataset == 'nn_no_distance'].copy()
    routed['observed_short'] = routed.observed <= 1.5
    for keys, group in routed.groupby(['observed_short', 'selected_short'], observed=True):
        rows.append(dict(observed_short=keys[0], selected_short=keys[1], **summarize(group)))
    pd.DataFrame(rows).to_csv(output/'routing_errors.csv', index=False)
    rows = []
    for keys, group in routed.groupby(['purpose', 'home_pt', 'car_group'], observed=True):
        rows.append(dict(purpose=keys[0], home_pt=keys[1], car_group=keys[2],
                         selected_short=np.average(group.selected_short, weights=group.weight),
                         **summarize(group)))
    pd.DataFrame(rows).to_csv(output/'replay_cohorts.csv', index=False)

    # Inspect fixed-holdout learning curves using the report's exact checkpoints.

    replay = json.loads((root/'replay_manifest.json').read_text())
    training = []
    holdout_rows = []
    for name in ['nn_distance', 'nn_no_distance']:
        artifacts = replay['runs'][name]['artifacts']
        heldout = []
        for stage, artifact in artifacts.items():
            if not stage.endswith(('regional_model', 'subregional_model', 'local_model', 'short_range_model')):
                continue
            directory = Path(artifact['path']).with_suffix('.cache')
            history = json.loads((directory/'loss_history.json').read_text())
            folds = json.loads((directory/'cross_validation_folds.json').read_text())
            heldout.append(set(folds['reserved_people']))
            epoch = history['best_epoch']
            validation = dict(zip(history['val_epochs'], history['val_losses']))
            training.append(dict(dataset=name, expert=stage.rsplit('.', 1)[-1],
                epochs=history['epochs_completed'], best_epoch=epoch,
                train_at_best=history['train_losses'][epoch-1], validation_at_best=validation[epoch],
                validation_first_fixed=validation.get(history['fixed_holdout_start_epoch']),
                validation_last=history['val_losses'][-1]))
        # This intersection was withheld from gradient training in all four experts,
        # but was still used for early stopping, so it is not an untouched test set.
        common = set.intersection(*heldout)
        sample = data.loc[data.dataset == name].copy()
        sample['all_experts_reserved'] = sample.person_id.isin(common)
        for flag, group in sample.groupby('all_experts_reserved'):
            holdout_rows.append(dict(dataset=name, all_experts_reserved=flag, **summarize(group)))
    pd.DataFrame(training).to_csv(output/'training_diagnostics.csv', index=False)
    pd.DataFrame(holdout_rows).to_csv(output/'replay_reserved_people.csv', index=False)

    # Compare assigned population distances against their actual supplied targets.

    prepared = json.loads((root/'prepared_manifest.json').read_text())
    stationary = []
    stationary_cohorts = []
    target_bands = []
    for name in ['nn_distance', 'nn_no_distance']:
        population = pd.read_parquet(root/f'population_{name}.parquet')
        run = prepared['runs'][name]
        cache = Path(run['cache'])
        metadata = json.loads((cache/'pipeline.json').read_text())
        key = Path(run['artifacts']['synthesis.population.spatial.secondary.locations']['path']).stem
        # Follow recorded dependencies rather than choosing the newest cached stage.
        pending, visited, chains = [key], set(), []
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            if current.startswith('synthesis.population.spatial.secondary_nn.mz_chains__'):
                chains.append(current)
            else:
                pending.extend(metadata[current]['dependencies'])
        if len(set(chains)) != 1:
            raise ValueError(f'{name}: expected one recorded survey-chain dependency, got {chains}')
        features = pd.read_pickle(cache/(chains[0]+'.p'))
        targets = features[['person_id', 'trip_id', 'target_distance']].rename(
            columns={'person_id': 'mz_person_id', 'target_distance': 'target_m'})
        population['mz_person_id'] = pd.to_numeric(population.mz_person_id)
        population = population.merge(targets, on=['mz_person_id', 'trip_id'], how='left', validate='many_to_one')
        if not np.isfinite(population.target_m).all():
            raise ValueError(f'{name}: missing supplied target distances')
        for target_short, group in population.groupby(population.target_m < 10):
            stationary.append(dict(dataset=name, target_below_10m=target_short, trips=len(group),
                stationary_trips=int((group.distance_km < .01).sum()),
                total_weight=group.weight.sum(), stationary_weight=group.loc[group.distance_km < .01, 'weight'].sum()))
        positive = population.loc[population.target_m >= 10].copy()
        positive['stationary'] = positive.distance_km < .01
        for keys, group in positive.groupby(['origin_purpose', 'purpose'], observed=True):
            stationary_cohorts.append(dict(dataset=name, origin_purpose=keys[0], purpose=keys[1],
                trips=len(group), stationary_trips=int(group.stationary.sum()),
                stationary_fraction=np.average(group.stationary, weights=group.weight)))
        population['band'] = pd.cut(population.target_m/1000, bins, right=False)
        population['observed'] = population.target_m/1000
        for band, group in population.groupby('band', observed=True):
            target_bands.append(dict(dataset=name, target_band=str(band), **summarize(group)))
    pd.DataFrame(stationary).to_csv(output/'population_stationary_sources.csv', index=False)
    pd.DataFrame(stationary_cohorts).to_csv(output/'population_positive_target_stationary.csv', index=False)
    pd.DataFrame(target_bands).to_csv(output/'population_target_bands.csv', index=False)
    print(f'Investigation tables: {output}')


if __name__ == '__main__':
    main()
