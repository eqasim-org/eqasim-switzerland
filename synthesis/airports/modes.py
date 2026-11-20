"""Distance preferences with fixed airport mode totals and PT feasibility."""

import json

import numpy as np
import pandas as pd

from data.airports._shared.common import apportion, file_digest, normalized
from data.airports.geneva.model import distances


def read_distance_model(sources, model, geneva_ods, modes):
    """Fit one a+b*km preference per mode to the three Geneva survey groups."""
    metadata = json.loads(sources['regional_mode_metadata'].read_text())
    if (metadata.get('schema_version') != 1 or
            metadata['pdf_sha256'] != file_digest(sources['pdf']) or
            metadata['csv_sha256'] != file_digest(sources['regional_modes'])):
        raise ValueError('Review changed Geneva regional mode evidence and its metadata')
    table = pd.read_csv(sources['regional_modes'])
    return fit_distance_model(table, model, geneva_ods, modes, metadata)


def fit_distance_model(table, model, geneva_ods, modes, provenance=None):
    if not {'region', 'mode', 'percent'}.issubset(table):
        raise ValueError('Regional mode evidence requires region, mode and percent columns')
    features = model.municipalities
    distance = distances(features, model.airports['GVA'])
    column = 'full_population_daily_trips' if 'full_population_daily_trips' in geneva_ods else 'daily_trips'
    weights = features.municipality_id.map(geneva_ods.groupby('municipality_id')[column].sum()).fillna(0).to_numpy(float)
    fallback = model.probabilities('GVA')
    observations, mean_distances = [], []
    labels = ['GE', 'VD', 'CH_OTHER']
    for label in labels:
        mask = (features.canton_id.eq(25) if label == 'GE' else features.canton_id.eq(22)
                if label == 'VD' else ~features.canton_id.isin([25, 22])).to_numpy()
        rows = table.loc[table.region.eq(label)]
        if rows['mode'].duplicated().any() or set(rows['mode']) != set(modes):
            raise ValueError(f'Regional mode evidence needs one observation per mode for {label}')
        shares = rows.set_index('mode').loc[list(modes), 'percent'].to_numpy(float)
        if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 100).any():
            raise ValueError('Regional mode percentages must be between zero and 100')
        local_weights = weights[mask]
        if local_weights.sum() == 0:
            local_weights = fallback[mask]
        if not mask.any() or local_weights.sum() <= 0:
            raise ValueError(f'No municipal distance support for Geneva mode group {label}')
        mean_distances.append(float(np.average(distance[mask], weights=local_weights)))
        observations.append(normalized(shares))
    x = np.array(mean_distances)
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(3), x]), observations, rcond=None)[0]
    return {'method': 'Equal-weight least squares a+b*Euclidean_km; three regional means, not individual observations',
            'modes': list(modes), 'intercept': coefficients[0].tolist(), 'slope_per_km': coefficients[1].tolist(),
            'distance_range_km': [float(x.min()), float(x.max())],
            'regions': labels, 'regional_mean_distance_km': mean_distances,
            'regional_normalized_shares': np.asarray(observations).tolist(), 'provenance': provenance or {},
            'regional_linear_predictions': (np.column_stack([np.ones(3), x]) @ coefficients).tolist(),
            'transfer': 'Geneva preferences transferred to Zurich; airport-wide totals remain fixed',
            'prediction': 'Clamp distance to fitted range, clip preference to [0.01, 1]; use as allocation weights only'}


def preferences(distance, fitted, modes):
    if fitted is None:
        return np.ones((len(distance), len(modes)))
    if list(modes) != fitted['modes']:
        raise ValueError('Distance model and airport detailed modes are inconsistent')
    known = np.isfinite(distance)
    bounded = np.clip(np.where(known, distance, fitted['distance_range_km'][0]), *fitted['distance_range_km'])
    weights = np.clip(np.asarray(fitted['intercept']) + bounded[:, None] * np.asarray(fitted['slope_per_km']), .01, 1.)
    # Missing municipality rows still go to the existing unresolved-demand audit.
    weights[~known] = 1.
    return weights


def assign_modes(frame, homes, hotels, model, shares, modes, mode_map, rng, fitted=None):
    """Fill PT quotas from A/B/C, then D, and distribute any remainder to road modes."""
    result = frame.copy()
    pt = np.array([mode_map[m] == 'pt' for m in modes])
    road = np.array([mode_map[m] in ('car', 'car_passenger') for m in modes])
    if not (pt | road).all():
        raise ValueError('Airport allocation supports PT, car and car-passenger modes')

    # Keep home and visitor accommodation eligibility separate within a municipality.

    tiers = {}
    for kind, pool in [('resident', homes), ('visitor', hotels)]:
        abc = set(pool.loc[pool.ovgk.isin(['A', 'B', 'C']), 'municipality_id'])
        d = set(pool.loc[pool.ovgk.eq('D'), 'municipality_id']) - abc
        tiers[kind] = (abc, d)
    result['pt_access_tier'] = ['ABC' if m in tiers[k][0] else 'D' if m in tiers[k][1] else 'none'
                              for m, k in zip(result.municipality_id, result.traveller_type)]
    # Resident agents are already selected: use their own home, not another home in the municipality.
    if {'person_id', 'endpoint_ovgk', 'is_visitor'}.issubset(result):
        resident = ~result.is_visitor
        quality = result.loc[resident, 'endpoint_ovgk']
        result.loc[resident, 'pt_access_tier'] = np.where(quality.isin(['A','B','C']), 'ABC',
            np.where(quality.eq('D'), 'D', 'none'))
    result['pt_location_available'] = result.pt_access_tier.ne('none')
    result['access_mode'] = pd.Series('', index=result.index, dtype='str')
    distance_lookup = {a: pd.Series(distances(model.municipalities, coord), index=model.municipalities.municipality_id)
                       for a, coord in model.airports.items()}
    result['mode_assignment_distance_km'] = [distance_lookup[a].get(m, np.nan)
                                            for a, m in zip(result.airport, result.municipality_id)]
    diagnostics = {}
    for airport in ('GVA', 'ZRH'):
        rows = result.loc[result.airport.eq(airport)]
        requested = apportion(len(rows) * normalized(shares))
        adjusted = requested.copy()
        abc_positions = np.flatnonzero(rows.pt_access_tier.eq('ABC').to_numpy())
        d_positions = np.flatnonzero(rows.pt_access_tier.eq('D').to_numpy())
        capacity = len(abc_positions) + len(d_positions)
        requested_pt = int(requested[pt].sum())
        feasible_pt = min(requested_pt, capacity)
        adjusted[pt] = apportion(requested[pt] * feasible_pt / max(1, requested_pt))

        # Both car and car passenger can absorb demand when even A/B/C/D is insufficient.

        shortfall = requested_pt - feasible_pt
        road_weights = np.asarray(shares)[road]
        if road_weights.sum() == 0:
            road_weights = np.array([float(m == 'private_car') for m in np.asarray(modes)[road]])
        additions = apportion(shortfall * normalized(road_weights))
        adjusted[road] += additions
        score = preferences(rows.mode_assignment_distance_km.to_numpy(float), fitted, modes)

        # Exhaust A/B/C support before using only the necessary number of D movements.

        abc_counts = np.zeros(len(modes), dtype=int)
        abc_total = min(feasible_pt, len(abc_positions))
        abc_counts[pt] = apportion(adjusted[pt] * abc_total / max(1, feasible_pt))
        d_counts = np.zeros(len(modes), dtype=int)
        d_counts[pt] = adjusted[pt] - abc_counts[pt]
        _allocate(result, rows.index, abc_positions, abc_counts, score, modes, rng, require_all=False)
        _allocate(result, rows.index, d_positions, d_counts, score, modes, rng, require_all=False)
        remaining = np.flatnonzero(result.loc[rows.index, 'access_mode'].eq('').to_numpy())
        road_counts = np.where(road, adjusted, 0)
        _allocate(result, rows.index, remaining, road_counts, score, modes, rng)
        assigned = result.loc[rows.index, 'access_mode'].value_counts()
        no_pt = rows.pt_access_tier.eq('none')
        forced = result.loc[rows.index[no_pt], 'access_mode'].map(mode_map)
        extra = dict(zip(np.asarray(modes)[road], map(int, additions)))
        diagnostics[airport] = {'requested': dict(zip(modes, map(int, requested))),
            'feasible_target': dict(zip(modes, map(int, adjusted))),
            'assigned': {m: int(assigned.get(m, 0)) for m in modes},
            'pt_eligible_trips': capacity, 'pt_eligible_trips_abc': len(abc_positions),
            'pt_eligible_trips_d_only': len(d_positions), 'pt_trips_using_d': int(d_counts.sum()),
            'forced_road_trips': int(no_pt.sum()), 'forced_car_trips': int(forced.eq('car').sum()),
            'forced_car_passenger_trips': int(forced.eq('car_passenger').sum()),
            'pt_target_shortfall': shortfall, 'additional_road_trips': shortfall,
            'additional_car_trips': sum(v for m, v in extra.items() if mode_map[m] == 'car'),
            'additional_car_passenger_trips': sum(v for m, v in extra.items() if mode_map[m] == 'car_passenger'),
            'margins_preserved': bool(np.array_equal(requested, adjusted))}
    result['mode_detailed'] = result.access_mode
    result['mode'] = result.access_mode.map(mode_map)
    return result, diagnostics


def _allocate(output, index, positions, counts, weights, modes, rng, require_all=True):
    """Weighted draws without replacement, with exact integer category quotas."""
    remaining = np.asarray(positions)
    # Random mode order avoids systematically giving the first category priority.
    for j in rng.permutation(len(modes)):
        count = int(counts[j])
        if count == 0:
            continue
        if count > len(remaining):
            raise ValueError('Infeasible airport mode allocation')
        keys = -np.log(np.maximum(rng.random(len(remaining)), np.finfo(float).tiny)) / weights[remaining, j]
        chosen = np.argpartition(keys, count - 1)[:count]
        output.loc[index[remaining[chosen]], 'access_mode'] = modes[j]
        remaining = np.delete(remaining, chosen)
    if require_all and len(remaining):
        raise ValueError('Airport mode allocation left unassigned movements')
