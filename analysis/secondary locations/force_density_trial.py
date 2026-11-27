"""Paired Vaud pilot for the destination of short observed home-other-home tours.

Run from the repo root with PYTHONPATH=. and pass the pipeline YAML and output
directory. Fields use sites within a 30 km square half-width; the evaluation
grid extends 18 km from Bel-Air. The 2015 survey uses official person weights.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from pyproj import Transformer

from cache_io import RunCache
from synthesis.population.spatial.secondary import locations_rda
from synthesis.population.spatial.secondary.components import CandidateIndex
from synthesis.population.spatial.secondary.force_model import force_field
from synthesis.population.spatial.secondary.force_model.attractors import employment_density, uniform

CENTER = np.array([2537879.798, 1152652.731])  # Lausanne Bel-Air, LV95
BINS = [1000., 2000., 4000., 8000.]
BETAS = (0., 0.1, 0.2)


def radius(x, y):
    # Nested hexagons with the same orientation as the Vaud 2 km map.
    dx, dy = abs(x - CENTER[0]), abs(y - CENTER[1])
    return np.maximum(dx, (dx + np.sqrt(3) * dy) / 2)


def tours(folder):
    options = dict(encoding='latin1')
    day = 'tag'
    people = pd.read_csv(folder / 'zielpersonen.csv', usecols=['HHNR', 'WP', 'alter', day], **options)
    people = people.loc[people[day].between(1, 5) & people.alter.ge(6)]
    homes = pd.read_csv(folder / 'haushalte.csv', usecols=['HHNR', 'W_X', 'W_Y'], **options)
    t = pd.read_csv(folder / 'wege.csv',
                    usecols=['HHNR', 'WEGNR', 'pseudo', 'wzweck1', 'wzweck2', 'S_X', 'S_Y', 'Z_X', 'Z_Y'],
                    **options)
    t = t[t.pseudo == 1].merge(people, on='HHNR', validate='many_to_one').merge(homes,
                                                                                on='HHNR',
                                                                                validate='many_to_one')
    t = t.sort_values(['HHNR', 'WEGNR'])
    tr = Transformer.from_crs(4326, 2056, always_xy=True)
    for prefix in ['W', 'S', 'Z']:
        t[prefix + 'x'], t[prefix + 'y'] = tr.transform(t[prefix + '_X'], t[prefix + '_Y'])
    following = t.groupby('HHNR').shift(-1)

    def close(x, y, xx, yy):
        return np.hypot(x - xx, y - yy) <= 25

    mask = (t.wzweck1.isin([5, 9, 10]) & t.wzweck2.eq(1) & following.WEGNR.eq(t.WEGNR + 1)
            & (following.wzweck2.gt(1) | following.wzweck1.eq(11))
            & close(t.Sx, t.Sy, t.Wx, t.Wy) & close(t.Zx, t.Zy, following.Sx, following.Sy)
            & close(following.Zx, following.Zy, t.Wx, t.Wy))
    t = t[mask].copy()
    t['length'] = np.hypot(t.Zx - t.Wx, t.Zy - t.Wy)
    near = np.hypot(t.Wx - CENTER[0], t.Wy - CENTER[1]) <= 8000
    return t[near & t.length.between(1, 8000)].assign(year=2015)


def trial(cache, sample, draws, seed, betas=BETAS, subtype_multipliers=None):
    # Use the production attractor stages, overriding only the pilot's density settings.
    catalog = cache.stage('synthesis.population.destinations', retain=True)
    base = uniform.execute(SimpleNamespace(stage=lambda name: catalog))
    data = {
        'synthesis.population.spatial.secondary.force_model.attractors.uniform': base,
        'data.statent.statent': cache.stage('data.statent.statent'),
        'data.spatial.cantons': cache.stage('data.spatial.cantons')
    }
    other = catalog[catalog.offers_other.fillna(False).astype(bool)]
    xy = np.column_stack((other.geometry.x, other.geometry.y))
    local = np.abs(xy - CENTER).max(axis=1) <= 30000
    employees = other.loc[local, 'number_employees'].fillna(1.).to_numpy(dtype=float)
    catalog_data = {
        'other': dict(locations=xy[local], identifiers=np.arange(local.sum()), number_employees=employees)
    }
    index = CandidateIndex(catalog_data, locations_rda.NUMBER_CANDIDATES, locations_rda.ALPHA_PROBABILITIES)
    grid = dict(xmin=CENTER[0] - 18000, ymin=CENTER[1] - 18000, cell_size=250., nx=145, ny=145)
    records = []
    for beta in betas:
        config = {employment_density.PREFIX + 'canton_ids': [22]}
        for p in employment_density.PURPOSES:
            config[employment_density.PREFIX + 'density_beta.' + p] = beta if p == 'other' else 0.
        weighted = employment_density.execute(SimpleNamespace(config=config.__getitem__, stage=data.__getitem__))
        # Same catalogue order as the national uniform stage, one row per eligible site.
        w = weighted.loc[weighted.activity_type.eq('other'), 'weight'].to_numpy()
        points = np.column_stack((xy[local], w[local]))
        fields = {}
        multipliers = {'other': 1.} if subtype_multipliers is None else subtype_multipliers
        for purpose, multiplier in multipliers.items():
            weighted_points = points.copy()
            weighted_points[:, 2] *= multiplier
            fields[purpose] = force_field.calculate_field(grid, weighted_points)
        solver = locations_rda.make_chain_solver(np.random.RandomState(seed), (grid, fields))
        for t in sample.itertuples():
            anchor = np.array([[t.Wx, t.Wy]])
            purpose = 'other' if subtype_multipliers is None else t.wzweck1
            problem = dict(origin=anchor, destination=anchor, size=1, purposes=[purpose])
            for draw in range(draws):
                seeds = np.random.SeedSequence([seed, t.year, t.HHNR, t.WEGNR, draw]).generate_state(2)
                solver.random = np.random.RandomState(seeds[0])
                index.random = np.random.RandomState(seeds[1])
                loc = solver.solve(problem, np.array([t.length, t.length]))['locations']
                _, pos = index.query_batch('other', loc)
                x, y = pos[0]
                records.append(
                    dict(year=t.year,
                         beta=beta,
                         person=t.HHNR,
                         trip=t.WEGNR,
                         draw=draw,
                         weight=t.WP,
                         observed=radius(t.Zx, t.Zy),
                         predicted=radius(x, y),
                         x=x,
                         y=y,
                         continuous_x=loc[0, 0],
                         continuous_y=loc[0, 1],
                         snap_seed=seeds[1],
                         distance_error=abs(np.hypot(x - t.Wx, y - t.Wy) - t.length)))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', help='Specific completed spatial assignment cache key')
    parser.add_argument('--draws', type=int, default=8)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error('--draws must be positive')
    cache = RunCache(args.config, root=args.root)
    path = Path(cache.yaml['config']['data_path']) / 'microcensus'
    sample = tours(path)
    if sample.empty:
        raise ValueError('No eligible 2015 return tours found near Lausanne Bel-Air')
    seed = int(cache.config('random_seed'))
    print('Assignment cache',
          cache.root,
          'seed',
          seed,
          'tours',
          sample.groupby('year').size().to_dict(),
          flush=True)
    df = trial(cache, sample, args.draws, seed)
    rows = []
    # Bootstrap paired household differences, keeping repeated tours and draws together.
    df['core_weight'] = df.weight * (df.predicted < BINS[0])
    paired = df.groupby(['year', 'person', 'beta']).core_weight.sum().unstack('beta')
    mass = df[df.beta == 0].groupby(['year', 'person']).weight.sum()
    for (year, beta), g in df.groupby(['year', 'beta']):
        shares = {}
        for col in ['observed', 'predicted']:
            counts = np.bincount(np.digitize(g[col], BINS), weights=g.weight, minlength=len(BINS) + 1)
            shares[col] = 100 * counts / counts.sum()
        by_person = paired.loc[year]
        diff = (by_person[beta] - by_person[0.]).to_numpy()
        weights = mass.loc[year].reindex(by_person.index).to_numpy()
        bootstrap = np.random.default_rng(seed + year).integers(len(weights), size=(2000, len(weights)))
        interval = np.percentile(100 * diff[bootstrap].sum(axis=1) / weights[bootstrap].sum(axis=1), [2.5, 97.5])
        rows.append(
            dict(year=year,
                 beta=beta,
                 people=g.person.nunique(),
                 tours=len(g) // args.draws,
                 draws=args.draws,
                 observed_core_pct=shares['observed'][0],
                 model_core_pct=shares['predicted'][0],
                 ring_error_pp=abs(shares['observed'] - shares['predicted']).sum() / 2,
                 mean_abs_leg_error_m=np.average(g.distance_error, weights=g.weight),
                 core_change_low_pp=interval[0],
                 core_change_high_pp=interval[1]))
    args.output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(args.output / 'lausanne_force_density.csv', index=False)
    print(results.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
