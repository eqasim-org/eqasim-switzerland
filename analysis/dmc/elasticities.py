"""Local expected-share sensitivities using generated, not observed, histories."""
import numpy as np
import pandas as pd

from . import replay

TIME_COLUMNS = {
    'car': ['car_travel_time_min', 'parking_searching_duration_min'],
    'pt': ['pt_in_vehicle_time_min', 'pt_access_egress_time_min', 'pt_transfer_time_min'],
    'bike': ['bike_travel_time_min'], 'walk': ['walk_travel_time_min'],
    'car_passenger': ['car_passenger_travel_time_min'],
}
COST_COLUMNS = {'car': ['car_cost_CHF', 'parking_cost_CHF'], 'pt': ['pt_cost_CHF']}


def perturb(arrays, driver, factor):
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Intervention factor must be positive and finite')
    changed = dict(arrays)
    if driver in ('income', 'age'):
        columns = [driver]
    elif driver == 'distance':
        columns = [name for name in arrays if name.endswith('_distance_km')]
    elif driver == 'distance_car_passenger':
        columns = ['car_passenger_distance_km']
    elif driver == 'transfers':
        columns = ['pt_transfers']
    elif driver == 'pt_waiting':
        columns = ['pt_transfer_time_min']
    elif driver.startswith('time_') and driver[5:] in TIME_COLUMNS:
        columns = TIME_COLUMNS[driver[5:]]
    elif driver.startswith('cost_') and driver[5:] in COST_COLUMNS:
        columns = COST_COLUMNS[driver[5:]]
    else:
        raise ValueError(f'Unknown intervention: {driver}')
    for name in columns:
        value = np.asarray(arrays[name])
        changed[name] = value + (factor - 1) * np.maximum(value, 0) if driver.startswith('time_') or driver == 'pt_waiting' else value * factor
    return changed


def log_share_elasticity(minus, plus, step):
    if not 0 < step < 1:
        raise ValueError('Finite-difference step must be between zero and one')
    result = np.full_like(np.asarray(minus, dtype=float), np.nan)
    positive = (minus > 0) & (plus > 0)
    result[positive] = (np.log(plus[positive]) - np.log(minus[positive])) / (np.log1p(step) - np.log1p(-step))
    return result


def compute(arrays, evaluate, coefficients, modes, rows, weights, variant, step=.01):
    drivers = ['income', 'age', 'distance', *['time_' + mode for mode in modes],
               'cost_car', 'cost_pt', 'transfers', 'pt_waiting']
    if 'car_passenger' in modes:
        drivers.append('distance_car_passenger')

    def shares(inputs):
        return np.average(replay.predict(inputs, evaluate, coefficients, modes, rows).probabilities, axis=0, weights=weights)

    baseline = shares(arrays)
    values = []; checks = []; scenarios = []
    for driver in drivers:
        estimates = []
        for increment in [step, step / 2]:
            minus = shares(perturb(arrays, driver, 1 - increment))
            plus = shares(perturb(arrays, driver, 1 + increment))
            estimates.append(log_share_elasticity(minus, plus, increment))
        difference = np.abs(estimates[0] - estimates[1])
        checks.append({'variant': variant, 'driver': driver,
                       'max_absolute_difference_full_vs_half_step': float(np.nanmax(difference)) if np.isfinite(difference).any() else np.nan})
        scenario = shares(perturb(arrays, driver, 1.1))
        for index, mode in enumerate(modes):
            values.append({'variant': variant, 'driver': driver, 'response_mode': mode,
                           'elasticity': estimates[0][index], 'step_fraction': step, 'baseline_share': baseline[index]})
            scenarios.append({'variant': variant, 'scenario': driver + '_plus_10pct', 'response_mode': mode,
                              'baseline_share': baseline[index], 'scenario_share': scenario[index],
                              'change_percentage_points': 100 * (scenario[index] - baseline[index])})
    changed = dict(arrays)
    changed['pt_transfers'] = arrays['pt_transfers'] + (np.asarray(arrays['pt_availability']) > 0)
    scenario = shares(changed)
    for index, mode in enumerate(modes):
        scenarios.append({'variant': variant, 'scenario': 'one_extra_pt_transfer', 'response_mode': mode,
                          'baseline_share': baseline[index], 'scenario_share': scenario[index],
                          'change_percentage_points': 100 * (scenario[index] - baseline[index])})
    return pd.DataFrame(values), pd.DataFrame(checks), pd.DataFrame(scenarios)
