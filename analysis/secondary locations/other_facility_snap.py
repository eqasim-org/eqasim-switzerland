"""Paired test of subtype preferences among the Swiss solver's 20 nearest sites."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree

from cache_io import RunCache
from force_density_trial import CENTER, radius, tours, trial
from other_facility_trial import GROUPS, facilities
from synthesis.population.spatial.secondary.locations_rda import NUMBER_CANDIDATES, ALPHA_PROBABILITIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path, help='Directory from other_facility_trial.py')
    parser.add_argument('--draws', type=int, default=12)
    args = parser.parse_args()
    cache = RunCache(args.config)
    folder = Path(cache.yaml['config']['data_path']) / 'microcensus'
    sample = tours(folder)
    result = trial(cache, sample, args.draws, int(cache.config('random_seed')), betas=[.2])
    result = result.merge(sample[['year', 'HHNR', 'WEGNR', 'wzweck1', 'Zx', 'Zy', 'Wx', 'Wy', 'length']],
                          left_on=['year', 'person', 'trip'],
                          right_on=['year', 'HHNR', 'WEGNR'],
                          validate='many_to_one')
    # Recover identical local sites, queries and random draws for every alternative policy.
    cat = cache.stage('synthesis.population.destinations')
    xy = np.column_stack([cat.geometry.x, cat.geometry.y])
    mask = cat.offers_other.fillna(False).astype(bool) & (np.abs(xy - CENTER).max(axis=1) <= 30000)
    xy = xy[mask]
    tree = KDTree(xy)
    idx = tree.query(result[['continuous_x', 'continuous_y']].to_numpy(),
                     k=NUMBER_CANDIDATES,
                     return_distance=False)
    employee_weights = cat.loc[mask, 'number_employees'].fillna(1.).to_numpy()**ALPHA_PROBABILITIES
    weights = employee_weights[idx]
    u = np.array([np.random.RandomState(seed).random_sample() for seed in result.snap_seed])

    def choose(w, idx):
        total = w.sum(axis=1, keepdims=True)
        prob = np.divide(w, total, out=np.full_like(w, 1. / w.shape[1]), where=total > 0)
        rank = np.minimum((u[:, None] > np.cumsum(prob, axis=1)).sum(axis=1), idx.shape[1] - 1)
        return xy[idx[np.arange(len(idx)), rank]]

    original = choose(weights, idx)
    assert np.array_equal(original, result[['x', 'y']].to_numpy())
    catalogs = facilities(cache)
    trees = {name: cKDTree(g[['x', 'y']]) for name, g in catalogs.groupby('sector', sort=True)}
    coef = pd.read_csv(args.output / 'facility_coefficients.csv').pivot(index=['method', 'code'],
                                                                        columns='sector',
                                                                        values='coefficient')
    features = np.stack([np.exp(-t.query(xy)[0] / 50.) for t in trees.values()], axis=1)
    coef = coef[list(trees)]
    # Apply only the estimated increment over the pooled 'other' preference.
    multipliers = {}
    for code in GROUPS:
        delta = coef.loc[('three_types', code)] - coef.loc[('pooled', 0)]
        multipliers[code] = np.exp(features @ delta.to_numpy())

    def rerank(indices):
        adjusted = employee_weights[indices].copy()
        for code, multiplier in multipliers.items():
            rows = result.wzweck1.eq(code)
            adjusted[rows] *= multiplier[indices[rows]]
        return choose(adjusted, indices)

    field_trial = trial(cache,
                        sample,
                        args.draws,
                        int(cache.config('random_seed')),
                        betas=[.2],
                        subtype_multipliers=multipliers)
    assert np.array_equal(result[['year', 'person', 'trip', 'draw', 'snap_seed']],
                          field_trial[['year', 'person', 'trip', 'draw', 'snap_seed']])
    field_idx = tree.query(field_trial[['continuous_x', 'continuous_y']].to_numpy(),
                           k=NUMBER_CANDIDATES,
                           return_distance=False)

    # Compare mutually exclusive 25 m facility profiles (one type, several, or none).
    def profile(points):
        near = np.stack([t.query(points)[0] <= 25 for t in trees.values()], axis=1)
        return np.select([near.sum(axis=1) == 0, near.sum(axis=1) > 1], [len(trees), len(trees) + 1],
                         default=near.argmax(axis=1))

    obs = result[['Zx', 'Zy']].to_numpy()
    truth = profile(obs)
    summary = []
    for name, points in [('density_only', original), ('subtype_snap', rerank(idx)),
                         ('subtype_field_and_snap', rerank(field_idx))]:
        classified = profile(points)
        for (yr, code), g in result.groupby(['year', 'wzweck1']):
            i = g.index.to_numpy()

            def avg(values):
                return np.average(values, weights=g.weight)

            def shares(values):
                return np.bincount(values[i], weights=g.weight, minlength=len(trees) + 2) / g.weight.sum()

            summary.append(
                dict(year=yr,
                     subtype=GROUPS[code],
                     method=name,
                     tours=len(i) // args.draws,
                     profile_error_pp=50 * abs(shares(truth) - shares(classified)).sum(),
                     core_observed_pct=100 * avg(radius(*obs[i].T) < 1000),
                     core_model_pct=100 * avg(radius(*points[i].T) < 1000),
                     leg_error_m=avg(abs(np.hypot(points[i, 0] - g.Wx, points[i, 1] - g.Wy) - g.length)),
                     facility_changed_pct=100 * avg((points[i] != original[i]).any(axis=1))))
    d = pd.DataFrame(summary)
    d.to_csv(args.output / 'heldout_force_facilities.csv', index=False)
    print(d.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
