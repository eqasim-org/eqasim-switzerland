"""Test facility-type preferences for services and escorts on held-out MZ trips."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from synthesis.population.spatial.secondary.force_model.facility_preferences import examples, fit, read_visits
from scipy.special import softmax
from scipy.spatial import cKDTree

from cache_io import RunCache
from synthesis.population.spatial.secondary.force_model.subtypes import facility_catalog, PURPOSES

GROUPS = {5: 'services', 9: 'escort_child', 10: 'escort_other_person'}
HOLDOUT_CANTONS = [1, 22, 25]


def visits(folder):
    d = read_visits(folder)
    return d.assign(year=2015, subtype=d.wzweck1.map(GROUPS))


def facilities(cache):
    stops = cache.directory / (cache.resolve('data.gtfs.cleaned') + '.cache') / 'output/stops.txt'
    border = cache.stage('data.spatial.swiss_border').to_crs(2056).geometry.union_all()
    return facility_catalog(cache.stage('data.statent.statent'), stops, border)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--export-preferences', type=Path, help='Write the incremental coefficients for the pipeline')
    args = parser.parse_args()
    cache = RunCache(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    folder = Path(cache.yaml['config']['data_path']) / 'microcensus'
    d = visits(folder)
    counts = d.groupby(['year', 'subtype']).agg(visits=('WP', 'size'),
                                                people=('HHNR', 'nunique'),
                                                weighted=('WP', 'sum'),
                                                precise=('Z_QAL', lambda a: a.eq(1).sum()))
    counts['weighted_share_pct'] = 100 * counts.weighted / counts.groupby(level=0).weighted.transform('sum')
    counts.to_csv(args.output / 'survey_support.csv')
    d = d[d.Z_QAL.eq(1) & np.isfinite(d.x) & np.isfinite(d.y)].copy()
    catalog = facilities(cache)
    # Audit destination precision separately so that the site profile includes short and cross-border trips.
    distances = pd.DataFrame(
        {
            label: cKDTree(g[['x', 'y']]).query(d[['x', 'y']])[0]
            for label, g in catalog.groupby('sector')
        },
        index=d.index)
    near = distances.le(25.)
    footprint = []
    for (year, subtype), group in d.groupby(['year', 'subtype']):
        for label in near.columns:
            footprint.append(
                dict(year=year,
                     subtype=subtype,
                     sector=label,
                     visits=len(group),
                     within_25m_pct=100 * np.average(near.loc[group.index, label], weights=group.WP)))
    pd.DataFrame(footprint).to_csv(args.output / 'nearby_types_25m.csv', index=False)
    tags = near.idxmax(axis=1).where(near.sum(axis=1) == 1, 'multiple types').where(near.any(axis=1), 'no match')
    profile = d.assign(type_nearby=tags).groupby(['year', 'subtype', 'type_nearby']).WP.agg(['size', 'sum'])
    profile['share_pct'] = 100 * profile['sum'] / profile.groupby(level=[0, 1])['sum'].transform('sum')
    profile.to_csv(args.output / 'facility_profile_25m.csv')
    length = np.hypot(d.x - d.ox, d.y - d.oy)
    d = d[d.S_LND.eq(8100) & d.S_QAL.eq(1) & length.between(250., 30000.)]
    used, x, allowed = examples(d, catalog)
    # Keep all diaries of 2015 validation respondents out of estimation.
    holdout = used.Z_KANTON.isin(HOLDOUT_CANTONS)
    train = ~used.HHNR.isin(used.loc[holdout, 'HHNR'])
    split = used.wzweck1.replace({10: 9})
    results = []
    models = {}
    coefficients = []
    for method, labels in [('pooled', np.zeros(len(used))), ('services_escort', split.to_numpy()),
                           ('three_types', used.wzweck1.to_numpy())]:
        score = np.zeros(allowed.shape)
        for label in np.unique(labels):
            tr = train & (labels == label)
            b = fit(x[tr], allowed[tr], used.loc[tr, 'WP'].to_numpy())
            coefficients.extend(
                dict(method=method, code=int(label), sector=sector, coefficient=value)
                for sector, value in zip(sorted(catalog.sector.unique()), b, strict=True))
            score[labels == label] = x[labels == label] @ b
        prob = softmax(np.where(allowed, score, -np.inf), axis=1)
        models[method] = prob
        for label in ['all', *GROUPS.values()]:
            test = holdout if label == 'all' else holdout & used.subtype.eq(label)
            weights = used.loc[test, 'WP'].to_numpy()
            nll = -np.log(prob[test, 0])
            baseline = -np.log(models['pooled'][test, 0])
            # Paired household bootstrap of the reduction in held-out log loss.
            cluster = pd.DataFrame(dict(hh=used.loc[test, 'HHNR'], gain=(baseline - nll) * weights,
                                        w=weights)).groupby('hh').sum()
            rng = np.random.default_rng(3912)
            boot = []
            for _ in range(20):
                i = rng.integers(len(cluster), size=(50, len(cluster)))
                boot.extend(cluster.gain.to_numpy()[i].sum(axis=1) / cluster.w.to_numpy()[i].sum(axis=1))
            wins = (score[test, 0, None] >= np.where(allowed[test], score[test], -np.inf)).all(axis=1)
            ties = (allowed[test] & (score[test] == score[test, 0, None])).sum(axis=1)
            results.append(
                dict(method=method,
                     year=2015,
                     subtype=label,
                     trips=test.sum(),
                     people=used.loc[test, 'HHNR'].nunique(),
                     alternatives=np.average(allowed[test].sum(axis=1), weights=weights),
                     log_loss=np.average(nll, weights=weights),
                     gain_low=np.percentile(boot, 2.5),
                     gain_high=np.percentile(boot, 97.5),
                     first_rank_pct=100 * np.average(wins / ties, weights=weights)))
    pd.DataFrame(results).to_csv(args.output / 'heldout_facility_choice.csv', index=False)
    pd.DataFrame(coefficients).to_csv(args.output / 'facility_coefficients.csv', index=False)
    if args.export_preferences:
        table = pd.DataFrame(coefficients).pivot(index='sector', columns=['method', 'code'], values='coefficient')
        relative = table['three_types'].subtract(table['pooled'][0], axis=0).rename(columns=PURPOSES)
        relative.to_csv(args.export_preferences, float_format='%.12g')
    print(counts.round(2).to_string())
    print(pd.DataFrame(results).query("subtype=='all'").round(4).to_string(index=False))


if __name__ == '__main__':
    main()
