"""The promoted utility equations, shared by numerical and symbolic backends."""
from dmc.constants import constants
from dmc.models.specification import linear_terms


def build(context, arrays, beta, modes, *, power, maximum, exp):
    distance = arrays['euclidean_distance_km']
    density = sum(beta[f'beta_destination_{name}_density'] * power(arrays[f'destination_{name}_density'], exponent) / scale
                  for name, exponent, scale in [
                      ('employee', constants.EMPLOYEES_DENSITY_EXPONENT, constants.EMPLOYEES_DENSITY_SCALE),
                      ('population', constants.POPULATION_DENSITY_EXPONENT, constants.POPULATION_DENSITY_SCALE),
                      ('companies', constants.COMPANIES_DENSITY_EXPONENT, constants.COMPANIES_DENSITY_SCALE)])
    cost_factor = 1.
    if context.config('distance_cost_interaction'):
        cost_factor *= power(distance / context.config('reference_euclidean_distance_km'), beta['lambda_cost_distance'])
    if context.config('income_cost_interaction') and context.config('use_income_in_dmc'):
        cost_factor *= power(arrays['income'] / context.config('reference_income_chf'), beta['lambda_cost_income'])

    result = {}
    for mode in modes:
        age = (arrays['age'] - 40.) / 10.
        income = arrays['income'] / beta['income_reference_chf'] - 1.
        value = (beta[f'beta_{mode}_asc']
                 + sum(beta[f'beta_{mode}_{suffix}'] * arrays[column] for suffix, column in linear_terms(mode))
                 + beta[f'beta_{mode}_densities'] * density
                 + beta[f'age_linear_{mode}'] * age + beta[f'age_curvature_{mode}'] * age**2
                 + beta[f'income_linear_{mode}'] * income + beta[f'income_curvature_{mode}'] * income**2
                 + beta[f'short_amplitude_{mode}'] * power(maximum(1. - distance / 5., 0.), beta[f'short_power_{mode}']))
        if mode == 'car':
            time = arrays['car_travel_time_min'] + arrays['parking_searching_duration_min']
            value += beta['beta_car_travel_time_min'] * power(time / constants.TIME_SCALE_MIN, beta['lambda_car_travel_time'])
            value += beta['beta_cost_CHF'] * (arrays['car_cost_CHF'] + arrays['parking_cost_CHF']) * cost_factor
        elif mode == 'pt':
            for endpoint in ['origin', 'destination']:
                value += beta[f'beta_pt_{endpoint}_quality'] * power(arrays[f'{endpoint}PtQuality'], beta[f'gamma_pt_{endpoint}_quality'])
            for name in ['in_vehicle', 'access_egress']:
                value += beta[f'beta_pt_{name}_time_min'] * power(arrays[f'pt_{name}_time_min'] / constants.TIME_SCALE_MIN, beta[f'lambda_pt_{name}_time'])
            value += beta['beta_pt_transfer_time_min'] * maximum(arrays['pt_transfer_time_min'], 0.) / constants.TIME_SCALE_MIN
            count = maximum(arrays['pt_transfers'], 0.)
            q = beta['transfer_count_power']
            transformed = ((1. + count)**q - 1.) / (2.**q - 1.)
            x = maximum(distance, 0.) / beta['transfer_peak_km']
            transformed *= 1. + beta['transfer_amplitude'] * power(x * exp(1. - x), beta['transfer_shape'])
            value += beta['beta_pt_transfers'] * transformed
            value += beta['beta_cost_CHF'] * arrays['pt_cost_CHF'] * cost_factor
        else:
            exponent = 'lambda_car_passenger_travel_time' if mode == 'car_passenger' else f'lambda_{mode}'
            value += beta[f'beta_{mode}_travel_time_min'] * power(arrays[f'{mode}_travel_time_min'] / constants.TIME_SCALE_MIN, beta[exponent])
            if mode == 'car_passenger':
                value += beta['beta_passenger_distance'] * arrays['car_passenger_distance_km'] / constants.DISTANCE_SCALE_KM
                value += beta['beta_passenger_long_distance'] * maximum(
                    arrays['car_passenger_distance_km'] - beta['passenger_long_distance_threshold_km'], 0.)
                for endpoint, column in [('origin_home', 'passenger_origin_home'), ('origin_work', 'origin_work'),
                                         ('destination_home', 'destination_home'), ('destination_work', 'destination_work')]:
                    value += beta[f'beta_passenger_{endpoint}'] * arrays[column]
                value += beta['beta_passenger_history'] * arrays['passenger_history']
        result[mode] = value
    return result
