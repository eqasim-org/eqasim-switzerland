#!/usr/bin/env python3
"""Compare neural short-trip predictions with distance-only sampling on the same observed origins."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))
from cache_io import RunCache
from metrics import mean_ci
from synthesis.population.spatial.secondary_nn.location_helpers import _prepare_destination_level2_index
from synthesis.population.spatial.secondary_nn.tools import (
    DistanceDestinationSampler, base_activity_purpose, destination_support_masks, origin_candidate_indices)
from synthesis.population.spatial.secondary_nn.hierarchical_utils import SECONDARY_ACTIVITIES


def summarize(frame, observed, label):
    data = frame.merge(observed[['person_id', 'trip_id', 'distance_km']],
        on=['person_id', 'trip_id'], suffixes=('', '_observed'), validate='many_to_one')
    data['signed_error_m'] = 1000 * (data.distance_km - data.distance_km_observed)
    data['absolute_error_m'] = data.signed_error_m.abs()
    data['band'] = pd.cut(data.distance_km_observed, [.01, .1, .25, .5, .7, 1., 1.5, 2., np.inf])
    rows = []
    for band, group in data.groupby('band', observed=True):
        rows.append(dict(model=label, observed_band=str(band), trips=group[['person_id', 'trip_id']].drop_duplicates().shape[0],
            observed_m=np.average(1000*group.distance_km_observed, weights=group.weight),
            predicted_m=np.average(1000*group.distance_km, weights=group.weight),
            bias_m=np.average(group.signed_error_m, weights=group.weight),
            mae_m=np.average(group.absolute_error_m, weights=group.weight)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DIRECTORY.parents[1]/'config_dib_10_nn_distance.yml')
    parser.add_argument('--output', type=Path, default=DIRECTORY/'results/distance_short_investigation')
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = json.loads((output/'before_replay_manifest.json').read_text())
    cache = RunCache(args.config, root=manifest['runs']['nn_distance']['root'])

    # Read the actual completed checkpoint and frozen comparison data.

    checkpoint = Path(cache.stage('synthesis.population.spatial.secondary_nn.short_range_model')[0])
    history = json.loads((checkpoint.parent/'loss_history.json').read_text())
    (output/'training_history.json').write_text(json.dumps(history, indent=2))
    observed = pd.read_parquet(output/'before_replay_observed.parquet')
    frame = pd.read_parquet(output/'before_replay_predictions.parquet')
    columns = ['person_id', 'trip_id', 'draw', 'origin_x', 'origin_y', 'destination_x', 'destination_y',
               'distance_km', 'weight', 'purpose']
    baseline = frame.loc[frame.dataset.eq('nn_distance'), columns].copy()
    predicted = baseline.copy()
    targets = observed.set_index(['person_id', 'trip_id']).distance_km

    # Use the same H3 origin neighbourhood and real purpose-eligible facility catalogue.

    h3_data, levels, _ = cache.stage('synthesis.population.spatial.secondary_nn.h3')
    pools, _, _, _, xy, _ = _prepare_destination_level2_index(cache, h3_data, levels)
    leaves = levels['level_2']
    ids = leaves.h3_index.astype(str).tolist()
    tree = KDTree(np.column_stack([leaves['centroid'].x, leaves['centroid'].y]))
    masks = destination_support_masks(ids, pools, SECONDARY_ACTIVITIES)
    samplers = {}
    coverage = []
    for (person_id, trip_id), group in predicted.groupby(['person_id', 'trip_id'], sort=False):
        target = float(targets.loc[(person_id, trip_id)] * 1000)
        if not 10 <= target <= 1500:
            continue
        row = group.iloc[0]
        base = base_activity_purpose(row.purpose)
        if base not in samplers:
            samplers[base] = DistanceDestinationSampler(ids, pools[base], xy[base])
        sampler = samplers[base]
        cells = origin_candidate_indices(tree, row.origin_x, row.origin_y, masks[base],
            radius_m=1800., min_candidates=6, expand_step_m=200.)
        indices, probability = sampler.distribution(cells, row.origin_x, row.origin_y, target)
        radius = np.hypot(sampler.xy[indices, 0]-row.origin_x, sampler.xy[indices, 1]-row.origin_y)
        coverage.append(dict(person_id=person_id, trip_id=trip_id, cells=len(cells), facilities=len(indices),
            target_m=target, closest_radius_error_m=float(np.abs(radius-target).min()),
            expected_radius_m=float(np.dot(radius, probability))))
        for index, draw in group.draw.items():
            rng = np.random.RandomState(np.random.SeedSequence(
                [manifest['seed'], int(person_id), int(trip_id), int(draw)]).generate_state(1)[0])
            chosen = int(rng.choice(indices, p=probability))
            x, y = sampler.xy[chosen]
            predicted.loc[index, ['destination_x', 'destination_y', 'distance_km']] = [
                x, y, np.hypot(x-row.origin_x, y-row.origin_y)/1000]

    # Export paired diagnostics, keeping hierarchy predictions identical.

    errors = pd.concat([summarize(baseline, observed, 'neural'),
                        summarize(predicted, observed, 'distance_only')], ignore_index=True)
    errors.to_csv(output/'short_distance_errors.csv', index=False)
    pd.DataFrame(coverage).to_parquet(output/'candidate_coverage.parquet', index=False)
    predicted.to_parquet(output/'counterfactual_replay.parquet', index=False)
    target = observed[['person_id', 'trip_id', 'distance_km']].rename(columns={'distance_km':'observed_km'})
    paired = baseline.merge(predicted[['person_id', 'trip_id', 'draw', 'distance_km']],
        on=['person_id', 'trip_id', 'draw'], suffixes=('_before', '_after'), validate='one_to_one')
    paired = paired.merge(target, on=['person_id', 'trip_id'], validate='many_to_one')
    paired = paired.loc[paired.observed_km.between(.01, 1.5)].copy()
    paired['before_error_m'] = 1000 * (paired.distance_km_before-paired.observed_km).abs()
    paired['after_error_m'] = 1000 * (paired.distance_km_after-paired.observed_km).abs()
    paired['improvement_m'] = paired.before_error_m-paired.after_error_m
    improvement, _, low, high = mean_ci(paired, 'improvement_m')
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), baseline_root=cache.root,
        checkpoint=str(checkpoint), best_epoch=history['best_epoch'], explored_epochs=history['epochs_completed'],
        validation_mode=history['validation_mode'], first_validation_loss=history['val_losses'][0],
        selected_validation_loss=history['best_val_loss'], sigma_fraction=.1, sigma_min_m=25.,
        short_mae_before_m=float(np.average(paired.before_error_m, weights=paired.weight)),
        short_mae_after_m=float(np.average(paired.after_error_m, weights=paired.weight)),
        paired_improvement_m=improvement, improvement_low_m=low, improvement_high_m=high,
        interpretation='Observed-origin counterfactual on existing survey records; not independent validation')
    (output/'diagnosis.json').write_text(json.dumps(summary, indent=2))
    (output/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Short-distance diagnosis</title>'
        '<h1>Short-distance sampling diagnosis</h1><p>Same trips, origins, purposes, facilities and draw seeds. '
        'Only distance-enabled trips from 10 m to 1.5 km change. This is a fitting-record diagnostic.</p>'
        + errors.to_html(index=False, float_format=lambda value:f'{value:.1f}'))
    print(errors.to_string(index=False))


if __name__ == '__main__':
    main()
