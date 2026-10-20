"""Weighted descriptive statistics; persons are the uncertainty clusters."""
import numpy as np
import pandas as pd


def quantile(values, weights, probabilities):
    values, weights = np.asarray(values), np.asarray(weights)
    order = np.argsort(values)
    return np.interp(probabilities, (np.cumsum(weights[order]) - .5 * weights[order]) / weights.sum(), values[order])


def mean_ci(frame, column='distance_km'):
    w = frame.weight.to_numpy(dtype=float)
    value = frame[column].to_numpy(dtype=float)
    valid = np.isfinite(value) & np.isfinite(w) & (w > 0)
    f = pd.DataFrame({'person_id': frame.person_id.to_numpy()[valid],
                      'w': w[valid], 's': w[valid] * value[valid]})
    totals = f.groupby('person_id', observed=True)[['w', 's']].sum()
    if totals.empty:
        return np.nan, np.nan, np.nan, np.nan
    mean = totals.s.sum() / totals.w.sum()
    g = len(totals)
    se = np.sqrt(np.sum((totals.s - mean * totals.w) ** 2) * g / max(g - 1, 1)) / totals.w.sum() if g > 1 else np.nan
    return float(mean), float(se), float(mean - 1.96 * se), float(mean + 1.96 * se)


def summaries(datasets):
    rows = []
    for name, frame in datasets.items():
        scopes = [('all', 'All secondary', frame)]
        for dimension in ['purpose', 'car_group', 'home_pt', 'age_group', 'municipality_type']:
            scopes.extend((dimension, str(level), group) for level, group in frame.groupby(dimension, observed=True))
        scopes.extend(('car_x_home_pt', str(car) + ' / ' + str(pt), group)
                      for (car, pt), group in frame.groupby(['car_group', 'home_pt'], observed=True))
        for dimension, group, f in scopes:
            mean, se, lower, upper = mean_ci(f)
            q = quantile(f.distance_km, f.weight, [.1, .5, .9, .95])
            rows.append(dict(dataset=name, dimension=dimension, group=group, trips=len(f),
                             persons=f.person_id.nunique(), sum_weights=f.weight.sum(),
                             mean_km=mean, mean_se_km=se, mean_low_km=lower, mean_high_km=upper,
                             p10_km=q[0], median_km=q[1], p90_km=q[2], p95_km=q[3],
                             short_share=np.average(f.distance_km <= 1.5, weights=f.weight),
                             local_share_5km=np.average(f.distance_km <= 5, weights=f.weight),
                             long_share_20km=np.average(f.distance_km > 20, weights=f.weight),
                             stationary_share=np.average(f.distance_km < .01, weights=f.weight),
                             mean_home_distance_km=np.average(f.home_distance_km, weights=f.weight)))
    return pd.DataFrame(rows)


def standardized_contrast(datasets, dimension, low, high, strata, min_people=10):
    """Same microcensus covariate mix and joint support for every method/group.

    Interval uses the person-level influence function of the standardized mean
    difference. Reference stratum weights are treated as fixed descriptive weights.
    """
    tables, frames = {}, {}
    for name, full in datasets.items():
        f = full.loc[full[dimension].isin([low, high])].copy()
        encoded = f[strata[0]].astype(str)
        for column in strata[1:]:
            encoded = encoded.str.cat(f[column].astype(str), sep='|')
        f['_stratum'] = encoded
        f['_sum'] = f.distance_km * f.weight
        tab = f.groupby(['_stratum', dimension], observed=True).agg(
            denominator=('weight', 'sum'), numerator=('_sum', 'sum'), persons=('person_id', 'nunique'))
        tab['mean'] = tab.numerator / tab.denominator
        tables[name], frames[name] = tab, f
    supported = None
    for table in tables.values():
        counts = table.persons.unstack(dimension).reindex(columns=[low, high]).fillna(0)
        valid = set(counts.index[(counts >= min_people).all(axis=1)])
        supported = valid if supported is None else supported & valid
    if not supported:
        return pd.DataFrame([dict(dataset=name, contrast=f'{high} − {low}', status='no_common_support') for name in datasets])
    reference = frames['survey'].groupby('_stratum', observed=True).weight.sum().reindex(sorted(supported))
    reference = reference / reference.sum()
    rows = []
    for name, f in frames.items():
        covered = f._stratum.isin(supported)
        coverage = f.loc[covered].weight.sum() / f.weight.sum()
        f = f.loc[covered].copy()
        table = tables[name].reset_index()
        f = f.merge(table[['_stratum', dimension, 'mean', 'denominator']], on=['_stratum', dimension], validate='many_to_one')
        f['_ref'] = f._stratum.map(reference)
        f['_influence'] = f['_ref'] * f.weight * (f.distance_km - f['mean']) / f.denominator
        f['_influence'] *= np.where(f[dimension] == high, 1., -1.)
        influence = f.groupby('person_id', observed=True)._influence.sum()
        se = np.sqrt((influence ** 2).sum() * len(influence) / max(len(influence) - 1, 1))
        means = {}
        for group in [low, high]:
            tab = tables[name].xs(group, level=dimension).reindex(reference.index)
            means[group] = float(np.sum(reference * tab['mean']))
        delta = means[high] - means[low]
        rows.append(dict(dataset=name, contrast=f'{high} − {low}', status='ok',
                         mean_low_group_km=means[low], mean_high_group_km=means[high],
                         difference_km=delta, se_km=se, low_km=delta - 1.96 * se,
                         high_km=delta + 1.96 * se, ratio=means[high] / means[low] if means[low] else np.nan,
                         common_strata=len(reference), retained_weight_share=coverage, persons=f.person_id.nunique()))
    return pd.DataFrame(rows)


def activity_spaces(frame):
    f = frame[['person_id', 'weight', 'destination_x', 'destination_y', 'distance_km', 'home_distance_km']].copy()
    # Local coordinates prevent catastrophic cancellation of large LV95 squares.
    f['x'] = (f.destination_x - 2600000.) / 1000
    f['y'] = (f.destination_y - 1200000.) / 1000
    f['r2'] = f.x ** 2 + f.y ** 2
    f['cell'] = (np.floor(f.destination_x / 250).astype('int64') * 100000
                  + np.floor(f.destination_y / 250).astype('int64'))
    out = f.groupby('person_id', observed=True).agg(weight=('weight', 'first'), secondary_legs=('person_id', 'size'),
        total_secondary_leg_km=('distance_km', 'sum'), max_home_distance_km=('home_distance_km', 'max'),
        x=('x', 'mean'), y=('y', 'mean'), r2=('r2', 'mean'), unique_cells_250m=('cell', 'nunique'))
    out['radius_gyration_km'] = np.sqrt(np.maximum(out.r2 - out.x ** 2 - out.y ** 2, 0))
    out['revisit_share_250m'] = 1 - out.unique_cells_250m / out.secondary_legs
    return out.drop(columns=['x', 'y', 'r2']).reset_index()


def transition_table(datasets):
    rows = []
    for name, frame in datasets.items():
        table = frame.groupby(['car_group', 'home_pt', 'destination_pt'], observed=True).weight.sum().rename('weight').reset_index()
        table['share'] = table.weight / table.groupby(['car_group', 'home_pt'], observed=True).weight.transform('sum')
        table['dataset'] = name
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def paired_replay_summary(observed, predictions):
    """Pair on survey trip, averaging model draws before comparing NN variants."""
    cols = ['person_id', 'trip_id']
    base = observed[cols + ['distance_km', 'weight']].rename(columns={'distance_km': 'observed_km'})
    aggregated = predictions.groupby(['dataset'] + cols, observed=True).agg(
        predicted_km=('distance_km', 'mean'), predicted_home_km=('home_distance_km', 'mean')).reset_index()
    paired = aggregated.merge(base, on=cols, validate='many_to_one')
    rows = []
    for name, f in paired.groupby('dataset', observed=True):
        error = f.predicted_km - f.observed_km
        rows.append(dict(dataset=str(name), trips=len(f), signed_distance_difference_km=np.average(error, weights=f.weight),
                         absolute_distance_difference_km=np.average(abs(error), weights=f.weight)))
    pivot = paired.pivot(index=cols, columns='dataset', values='predicted_km')
    if {'nn_distance', 'nn_no_distance'} <= set(pivot):
        delta = pivot.nn_no_distance - pivot.nn_distance
        f = base.set_index(cols).loc[delta.index].copy()
        f['delta'] = delta
        mean, se, low, high = mean_ci(f.reset_index(), 'delta')
        rows.append(dict(dataset='nn_no_distance_minus_nn_distance', trips=len(f),
                         signed_distance_difference_km=mean, difference_low_km=low, difference_high_km=high))
    return pd.DataFrame(rows), paired


def selector_diagnostics(observed, predictions, threshold_km=1.5):
    base = observed[['person_id', 'trip_id', 'crowfly_distance']].copy()
    base['observed_km'] = base.crowfly_distance / 1000
    # Stationary trips bypass selection; do not credit a classifier for them.
    base = base.loc[base.observed_km >= .01]
    trips = predictions.groupby(['dataset', 'person_id', 'trip_id'], observed=True).agg(
        selected_short=('selected_short', 'first'), weight=('weight', 'sum'),
        car_group=('car_group', 'first')).reset_index().merge(base, on=['person_id','trip_id'], validate='many_to_one')
    trips['observed_short'] = trips.observed_km <= threshold_km
    table = trips.groupby(['dataset','observed_short','selected_short'], observed=True).agg(
        weight=('weight','sum'),trips=('person_id','size'),persons=('person_id','nunique')).reset_index()
    table['row_share'] = table.weight / table.groupby(['dataset','observed_short'],observed=True).weight.transform('sum')
    return table, trips
