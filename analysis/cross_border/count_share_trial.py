"""Offline count-share/geography experiment; never called by production stages.

Inputs are a read-only demand snapshot and precomputed crossing-to-endpoint road
costs. Work on repeated OD groups rather than routing expanded agents one by one.
Run with: python -m analysis.cross_border.count_share_trial --input DIR --output DIR
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from data.cross_border.count_balancing import allocate, feasible


def quantile(values, weights, probability):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order[np.searchsorted(cumulative, probability*cumulative[-1])]])


def integer_draw(flow, agents, rng):
    """Draw whole people/tours, keeping a visitor's entry and return together."""
    mass = flow.sum(axis=1)
    probability = flow/mass[:, None]
    draw = np.zeros_like(flow)
    # Most rows are individual Swiss residents. Draw these in one vectorised
    # operation, leaving only repeated survey OD groups for multinomial draws.
    single = np.flatnonzero(agents == 1)
    cdf = np.cumsum(probability[single], axis=1)
    cdf[:, -1] = 1.
    chosen = (rng.random(len(single))[:, None] >= cdf).sum(axis=1)
    draw[single, chosen] = mass[single]
    for row in np.flatnonzero(agents > 1):
        draw[row] = rng.multinomial(int(agents[row]), probability[row])*(mass[row]/agents[row])
    return draw


def metrics(allocation, cost, road, target_total, fixed_exits):
    total = allocation.sum(axis=0) + fixed_exits
    mass = allocation.sum()
    extra = cost - cost.min(axis=1)[:, None]
    swiss_extra = road-road.min(axis=1)[:, None]
    return dict(total_crossings=float(total.sum()),
                share_displaced_pct=float(50*np.abs(total-target_total).sum()/total.sum()),
                largest_relative_count_gap_pct=float(100*np.max(np.abs(total-target_total)/target_total)),
                mean_swiss_road_minutes=float((allocation*road).sum()/mass),
                mean_extra_swiss_minutes=float((allocation*swiss_extra).sum()/mass),
                p95_extra_swiss_minutes=quantile(swiss_extra.ravel(),allocation.ravel(),.95),
                share_over_20_extra_swiss_minutes_pct=float(100*allocation[swiss_extra>20].sum()/mass),
                mean_extra_combined_minutes=float((allocation*extra).sum()/mass),
                p95_extra_combined_minutes=quantile(extra.ravel(), allocation.ravel(), .95),
                share_over_20_extra_minutes_pct=float(100*allocation[extra>20].sum()/mass))


def run(input_path, output):
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(input_path/'od.csv')
    sites = pd.read_csv(input_path/'sites.csv')
    reference = pd.read_json(input_path/'references.json').set_index('fid').loc[sites.fid]
    baseline = np.load(input_path/'baseline_allocation.npy')
    air = np.load(input_path/'air_costs.npz')
    roads = np.load(input_path/'road_costs.npz')
    fixed = air['fixed']
    mass = rows.mass.to_numpy()
    total = mass.sum() + fixed.sum()
    observed = ((reference.low+reference.high)/2).to_numpy()
    targets = total*observed/observed.sum()
    # Do not invent transit exits or reroute transit whose other border is
    # unlocated. Their current entries and all known exits reserve places first.
    frozen = rows.category.eq('transit').to_numpy() & ~roads['transit_known']
    fixed_entries = baseline[frozen].sum(axis=0)
    variable = ~frozen
    residual = targets-fixed-fixed_entries
    assert (residual>0).all()
    road = roads['road']
    # Returning visitors must respect the Swiss travel-time limit in both
    # directions, not merely on the average of their inward/outward legs.
    swiss_guard = road-road.min(axis=1)[:,None]
    visitors = rows.category.eq('foreign_visitor').to_numpy()
    swiss_guard[visitors] = np.maximum(roads['inward']-roads['inward'].min(axis=1)[:,None],
                                      roads['outward']-roads['outward'].min(axis=1)[:,None])[visitors]
    cost = road+roads['approach']
    scenarios = {'current_assignment': baseline.copy()}
    naive = baseline.copy()
    naive[variable] = mass[variable, None] * residual[None, :]/residual.sum()
    scenarios['count_shares_without_geography'] = naive
    fits, feasibility = {}, []
    for cap in (10., 15., 20., 30., 60., np.inf):
        support = cost[variable] <= cost[variable].min(axis=1)[:, None]+cap
        feasibility.append(dict(basis='combined',extra_minutes_cap=cap,
                                feasible=feasible(mass[variable], residual, support)))
    for cap in (10.,15.,20.,30.,40.,60.):
        support=swiss_guard[variable]<=cap
        feasibility.append(dict(basis='swiss_only',extra_minutes_cap=cap,feasible=feasible(mass[variable],residual,support)))
    cases = [('soft_10_cap15', 10., 15., 1.), ('soft_10_cap20', 10., 20., 1.),
             ('soft_20_cap20', 20., 20., 1.), ('hard_counts_unrestricted', None, np.inf, 1.),
             ('soft_10_cap20_half_foreign_access', 10., 20., .5),
             ('soft_10_cap20_swiss_only', 10., 20., 0.),
             ('soft_20_cap30_swiss_only', 20., 30., 0.),
             ('hard_counts_swiss_only', None, np.inf, 0.),
             ('hard_counts_swiss_cap20', None, 20., 0.),
             ('hard_counts_swiss_cap20_foreign_approach', None, 20., 1.),
             ('soft_20_cap30_half_foreign_access', 20., 30., .5)]
    for name, penalty, cap, approach_weight in cases:
        score = road+approach_weight*roads['approach']
        allowed = swiss_guard[variable]<=cap if name.startswith('hard_counts_swiss_cap20') else score[variable]<=score[variable].min(axis=1)[:,None]+cap
        flow, fit = allocate(mass[variable], score[variable], residual, allowed, penalty=penalty)
        allocation = baseline.copy()
        allocation[variable] = flow
        assert np.allclose(allocation.sum(axis=1), mass)
        scenarios[name] = allocation
        fits[name] = fit
    statistics = []
    counts = sites[['fid', 'name', 'point_id', 'importance', 'capacity']].copy()
    counts['official_reference'] = observed
    counts['normalised_share_target'] = targets
    counts['fixed_transit_exits'] = fixed
    counts['fixed_uncertain_transit_entries'] = fixed_entries
    categories = []
    for name, allocation in scenarios.items():
        counts[name] = allocation.sum(axis=0)+fixed
        statistics.append(dict(scenario=name, **metrics(allocation, cost, road, targets, fixed),
                               share_over_20_extra_either_swiss_direction_pct=float(100*allocation[swiss_guard>20].sum()/mass.sum())))
        for category, indices in rows.groupby('category').groups.items():
            columns = allocation[np.asarray(indices)].sum(axis=0)
            assert np.isclose(columns.sum(), rows.loc[indices, 'mass'].sum())
            for j, fid in enumerate(sites.fid):
                categories.append(dict(scenario=name, category=category, fid=int(fid), crossings=float(columns[j])))
        np.save(output/(name+'.npy'), allocation)
    counts.to_csv(output/'crossing_comparison.csv', index=False)
    pd.DataFrame(statistics).to_csv(output/'geography_comparison.csv', index=False)
    pd.DataFrame(categories).to_csv(output/'category_comparison.csv', index=False)
    # Freeze these inputs so later pipeline runs cannot silently change the trial.
    metadata = dict(total_crossings=total, official_total=float(observed.sum()),
                    reference_normalisation=float(total/observed.sum()),
                    fixed_uncertain_transit_entries=float(fixed_entries.sum()),
                    movable_od_cells=int(variable.sum()), fixed_od_cells=int(frozen.sum()),
                    fits=fits, joint_feasibility=feasibility,
                    road_timings=json.loads((input_path/'road_timings.json').read_text()),
                    extraction=json.loads((input_path/'extraction.json').read_text()))
    (output/'diagnostics.json').write_text(json.dumps(metadata, indent=2))
    print(counts[['fid','normalised_share_target',*scenarios.keys()]].round(1).to_string(index=False))
    print(pd.DataFrame(statistics).round(2).to_string(index=False))
    print('Joint road-time feasibility:', feasibility)
    print('Solve times:', {k:round(v['seconds'],4) for k,v in fits.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.input, args.output)
