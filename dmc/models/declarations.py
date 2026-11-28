"""Shared declarations for the current JAX and Biogeme utility specification."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Parameter:
    name: str
    init_value: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    fixed: bool = False


# Initial values and bounds; fitted values always come from estimation.
PARAMETERS = [
    ('lambda_cost_distance', -0.08, None, -0.002, False),
    ('lambda_cost_income', 0.0, None, 0.0, False),
    ('lambda_car_travel_time', 0.7, 0.01, 2.0, False),
    ('lambda_pt_in_vehicle_time', 1.0, 0.01, 2.0, True),
    ('lambda_pt_access_egress_time', 1.0, 0.01, 2.0, False),
    ('lambda_pt_transfer_time', 1.0, 0.01, 2.0, True),
    ('lambda_car_passenger_travel_time', 1.0, 0.01, 2.0, False),
    ('lambda_bike', 1.0, 0.01, 2.0, False),
    ('lambda_walk', 1.0, 0.01, 2.0, False),
    ('beta_cost_CHF', -0.12, None, -0.002, False),
    ('beta_destination_employee_density', 0.1, None, None, False),
    ('beta_destination_population_density', 0.2, None, None, False),
    ('beta_destination_companies_density', 0.1, None, None, True),
    ('beta_car_asc', 3.48, None, None, False),
    ('beta_car_travel_time_min', -0.964, None, -0.002, False),
    ('beta_car_destination_work', 0.447, None, None, False),
    ('beta_car_destination_home', 0.0, None, None, True),
    ('beta_car_destination_education', -0.596, None, None, False),
    ('beta_car_destination_shopping', 0.608, None, None, False),
    ('beta_car_destination_leisure', 0.251, None, None, False),
    ('beta_car_destination_other', 0.9, None, None, False),
    ('beta_car_origin_home', 0.0, None, None, True),
    ('beta_car_destination_urban', -0.072, None, 0, False),
    ('beta_car_destination_urbancore', -1.039, None, 0, False),
    ('beta_car_sex', -0.574, None, None, False),
    ('beta_car_ownership_ratio', -2.25, None, None, False),
    ('beta_car_region_1', 0.223, None, None, False),
    ('beta_car_region_2', -0.467, None, None, False),
    ('beta_car_densities', -0.1, None, None, False),
    ('beta_car_destination_zurich', -0.1, None, 0, False),
    ('beta_car_destination_geneva', -0.1, None, 0, False),
    ('beta_car_destination_basel', 0.0, None, 0, False),
    ('beta_car_destination_lausanne', -0.01, None, 0, False),
    ('beta_car_destination_luzern', 0.0, None, 0, False),
    ('beta_car_destination_bern', 0.0, None, 0, False),
    ('beta_pt_asc', 0.0, None, None, True),
    ('beta_pt_access_egress_time_min', -0.716, None, -0.002, False),
    ('beta_pt_in_vehicle_time_min', -0.044, None, -0.002, False),
    ('beta_pt_transfers', -0.477, None, -0.002, False),
    ('beta_pt_transfer_time_min', -0.0234, None, -0.02, False),
    ('beta_pt_sex', 0.0, None, None, True),
    ('beta_pt_destination_work', 0.0, None, None, True),
    ('beta_pt_destination_home', 0.0, None, None, True),
    ('beta_pt_destination_education', 0.0, None, None, True),
    ('beta_pt_destination_shopping', 0.0, None, None, True),
    ('beta_pt_destination_leisure', 0.0, None, None, True),
    ('beta_pt_destination_other', 0.0, None, None, True),
    ('beta_pt_origin_home', 0.0, None, None, True),
    ('beta_pt_destination_urban', 0.0, None, None, True),
    ('beta_pt_destination_urbancore', 0.0, None, None, True),
    ('beta_pt_region_1', 0.0, None, None, True),
    ('beta_pt_region_2', 0.0, None, None, True),
    ('beta_pt_destination_quality', -0.05, None, 0.0, False),
    ('gamma_pt_destination_quality', 1.0, 0.25, 3.0, False),
    ('beta_pt_densities', 0.08, None, None, False),
    ('beta_bike_asc', 3.667, None, None, False),
    ('beta_bike_travel_time_min', -2.873, None, -0.002, False),
    ('beta_bike_sex', -0.441, None, None, False),
    ('beta_bike_destination_work', 0.0, None, None, True),
    ('beta_bike_destination_home', 0.0, None, None, True),
    ('beta_bike_destination_education', -0.45, None, None, False),
    ('beta_bike_destination_shopping', -0.353, None, None, False),
    ('beta_bike_destination_leisure', -0.093, None, None, False),
    ('beta_bike_destination_other', -0.515, None, None, False),
    ('beta_bike_origin_home', 0.228, None, None, False),
    ('beta_bike_destination_urban', -0.32, None, None, False),
    ('beta_bike_destination_urbancore', -0.655, None, None, False),
    ('beta_bike_region_1', -0.929, None, None, False),
    ('beta_bike_region_2', -0.413, None, None, False),
    ('beta_bike_densities', 0.0, None, None, False),
    ('beta_walk_asc', 10.58, None, None, False),
    ('beta_walk_travel_time_min', -8.164, None, -0.002, False),
    ('beta_walk_sex', -0.211, None, None, False),
    ('beta_walk_destination_work', 0.0, None, None, True),
    ('beta_walk_destination_home', 0.0, None, None, True),
    ('beta_walk_destination_education', -0.181, None, None, False),
    ('beta_walk_destination_shopping', 0.022, None, None, True),
    ('beta_walk_destination_leisure', 0.354, None, None, False),
    ('beta_walk_destination_other', 0.111, None, None, False),
    ('beta_walk_origin_home', 0.198, None, None, False),
    ('beta_walk_destination_urban', -0.18, None, None, False),
    ('beta_walk_destination_urbancore', -0.469, None, None, False),
    ('beta_walk_region_1', 0.241, None, None, False),
    ('beta_walk_region_2', -0.155, None, None, False),
    ('beta_walk_densities', 0.0, None, None, True),
    ('beta_car_passenger_asc', 0.46, None, None, False),
    ('beta_car_passenger_travel_time_min', -1.27, None, -0.002, False),
    ('beta_car_passenger_driving_permit', -0.339, None, None, False),
    ('beta_car_passenger_sex', 0.141, None, None, False),
    ('beta_car_passenger_destination_education', -0.558, None, None, False),
    ('beta_car_passenger_destination_shopping', 0.999, None, None, False),
    ('beta_car_passenger_destination_leisure', 1.276, None, None, False),
    ('beta_car_passenger_destination_other', 0.0, None, None, True),
    ('beta_car_passenger_destination_urban', -0.146, None, None, False),
    ('beta_car_passenger_destination_urbancore', -1.097, None, None, False),
    ('beta_car_passenger_region_1', 0.291, None, None, False),
    ('beta_car_passenger_region_2', -0.515, None, None, False),
    ('beta_car_passenger_ownership_ratio', 0.0, None, None, False),
    ('beta_car_passenger_has_car', 0.0, None, None, False),
    ('beta_car_passenger_densities', -0.15, None, None, False),
    ('age_linear_car', 0.0, -5.0, 5.0, False),
    ('age_curvature_car', 0.0, -2.0, 2.0, False),
    ('income_linear_car', 0.0, -5.0, 5.0, False),
    ('income_curvature_car', 0.0, -2.0, 2.0, False),
    ('short_amplitude_car', 0.0, -12.0, 12.0, False),
    ('short_power_car', 2.0, 1.0, 8.0, False),
    ('age_linear_pt', 0.0, -5.0, 5.0, True),
    ('age_curvature_pt', 0.0, -2.0, 2.0, True),
    ('income_linear_pt', 0.0, -5.0, 5.0, True),
    ('income_curvature_pt', 0.0, -2.0, 2.0, True),
    ('short_amplitude_pt', 0.0, -12.0, 12.0, True),
    ('short_power_pt', 2.0, 1.0, 8.0, True),
    ('age_linear_bike', 0.0, -5.0, 5.0, False),
    ('age_curvature_bike', 0.0, -2.0, 2.0, False),
    ('income_linear_bike', 0.0, -5.0, 5.0, False),
    ('income_curvature_bike', 0.0, -2.0, 2.0, False),
    ('short_amplitude_bike', 0.0, -12.0, 12.0, False),
    ('short_power_bike', 2.0, 1.0, 8.0, False),
    ('age_linear_walk', 0.0, -5.0, 5.0, False),
    ('age_curvature_walk', 0.0, -2.0, 2.0, False),
    ('income_linear_walk', 0.0, -5.0, 5.0, False),
    ('income_curvature_walk', 0.0, -2.0, 2.0, False),
    ('short_amplitude_walk', 0.0, -12.0, 12.0, False),
    ('short_power_walk', 2.0, 1.0, 8.0, False),
    ('age_linear_car_passenger', 0.0, -5.0, 5.0, False),
    ('age_curvature_car_passenger', 0.0, -2.0, 2.0, False),
    ('income_linear_car_passenger', 0.0, -5.0, 5.0, False),
    ('income_curvature_car_passenger', 0.0, -2.0, 2.0, False),
    ('short_amplitude_car_passenger', 0.0, -12.0, 12.0, False),
    ('short_power_car_passenger', 2.0, 1.0, 8.0, False),
    ('income_reference_chf', 5000.0, None, None, True),
    ('transfer_peak_km', 7.0, None, None, True),
    ('transfer_count_power', 0.65, 0.1, 1.0, False),
    ('transfer_amplitude', 0.5, 0.0, 5.0, False),
    ('transfer_shape', 2.0, 0.2, 15.0, False),
    ('beta_pt_origin_quality', -0.05, None, 0.0, False),
    ('gamma_pt_origin_quality', 1.0, 0.25, 3.0, False),
    ('beta_passenger_distance', -0.01, None, 0.0, False),
    ('beta_passenger_long_distance', -0.01, None, -0.01, False),
    ('passenger_long_distance_threshold_km', 25.0, None, None, True),
    ('beta_passenger_origin_home', 0.0, None, None, False),
    ('beta_passenger_origin_work', 0.0, None, None, False),
    ('beta_passenger_destination_home', 0.0, None, None, False),
    ('beta_passenger_destination_work', 0.0, None, None, False),
    ('beta_passenger_history', 0.0, None, None, False),
    ('beta_passenger_history_any', 0.0, None, None, False),
]


def define_parameters(ignore_car_passenger, use_exponents, use_income, *,
                      income_reference=5000., transfer_peak_km=7.,
                      distance_interaction=True, income_interaction=True):
    if not math.isfinite(income_reference) or not math.isfinite(transfer_peak_km) or income_reference <= 0 or transfer_peak_km <= 0:
        raise ValueError('Income reference and transfer peak must be positive')
    result = {}
    for row in PARAMETERS:
        p = Parameter(*row)
        if ignore_car_passenger and 'passenger' in p.name:
            continue
        if p.name == 'income_reference_chf':
            p = replace(p, init_value=float(income_reference))
        elif p.name == 'transfer_peak_km':
            p = replace(p, init_value=float(transfer_peak_km))
        elif not use_exponents and p.name.startswith('lambda_') and not p.name.startswith('lambda_cost_'):
            p = replace(p, init_value=1., fixed=True)
        elif not use_income and p.name.startswith(('income_linear_', 'income_curvature_')):
            p = replace(p, init_value=0., fixed=True)
        elif p.name == 'lambda_cost_income' and not (use_income and income_interaction):
            p = replace(p, init_value=0., fixed=True)
        elif p.name == 'lambda_cost_distance' and not distance_interaction:
            p = replace(p, init_value=0., upper_bound=0., fixed=True)
        result[p.name] = p
    return result
