#!/usr/bin/env python3
"""Inspect short-range training/inference choice sets without changing cached data."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from cache_io import RunCache
from synthesis.population.spatial.secondary_nn.h3 import H3_LEVEL_NAMES


def diagnose(cache, frame):

    # Load the occupied coordinate leaves and the survey's training branch labels.

    data, geometry, _ = cache.stage('synthesis.population.spatial.secondary_nn.h3')
    leaf = geometry[H3_LEVEL_NAMES[-1]]
    occupied = leaf.loc[leaf.num_statent.gt(0)]
    branches = np.array(['/'.join(value.split('/')[:2]) for value in occupied.h3_index.astype(str)])
    branch_counts = pd.Series(branches).value_counts()
    centers = np.column_stack([occupied['centroid'].x, occupied['centroid'].y])
    tree = KDTree(centers)
    threshold = float(cache.config('secondary_nn_short_trip_threshold_m', 1500.))
    radius = threshold + 1.5 * float(cache.config('secondary_nn_short_query_radius_m', 200.))
    frame = frame.loc[frame.crowfly_distance.between(10, threshold)].copy()
    input_count = len(frame)
    column = 'destination_' + H3_LEVEL_NAMES[1]
    labels = data['microcensus_trips'][['person_id', 'trip_id', column]]
    frame = frame.merge(labels, on=['person_id', 'trip_id'], validate='one_to_one').dropna(subset=[column])

    # Compare geography before purpose filtering; actual eligible pools may be smaller.

    fractions, query_sizes, training_sizes = [], [], []
    neighborhoods = tree.query_radius(frame[['origin_x', 'origin_y']].to_numpy(), r=radius)
    for branch, nearby in zip(frame[column], neighborhoods):
        if len(nearby):
            fractions.append(float(np.mean(branches[nearby] != branch)))
            query_sizes.append(len(nearby))
            training_sizes.append(int(branch_counts.get(branch, 0)))
    if not fractions:
        raise ValueError('No short survey trips with occupied H3 neighborhoods')
    return dict(
        reference_root=cache.root,
        input_short_survey_trips=input_count,
        trips_with_training_branch=len(frame),
        trips_with_occupied_origin_neighborhood=len(fractions),
        choice_set_scope='Occupied H3 cells before purpose filtering',
        initial_inference_radius_m=radius,
        median_fraction_origin_candidates_outside_training_branch=float(np.median(fractions)),
        share_trips_with_origin_candidates_outside_training_branch=float(np.mean(np.array(fractions) > 0)),
        median_origin_candidate_cells=float(np.median(query_sizes)),
        median_observed_branch_candidate_cells=float(np.median(training_sizes)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPOSITORY / 'config_dib_10_nn_distance.yml')
    results = Path(__file__).with_name('results') / 'loss_weight_sweep'
    parser.add_argument('--observed', type=Path, default=results / 'replay_observed.parquet')
    parser.add_argument('--output', type=Path, default=results / 'short_range_diagnosis.json')
    args = parser.parse_args()
    result = diagnose(RunCache(args.config), pd.read_parquet(args.observed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
