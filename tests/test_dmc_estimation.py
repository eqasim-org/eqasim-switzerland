from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import rosen, rosen_der

from dmc.models.model_jax import Parameter, configure, optimize, projected_gradient_norm
from dmc.vot.functions import vot_utils


class Context:
    def __init__(self, values=None):
        self.values = dict(values or {})
        configure(self)

    def config(self, name, default=None, **kwargs):
        return self.values.setdefault(name, default)

    def stage(self, name, **kwargs):
        pass


def test_refinement_continues_after_function_plateau_with_shared_budget():
    context = Context({'dmc_jax_max_iterations': 300, 'dmc_jax_function_tolerance': .01,
                       'dmc_jax_gradient_tolerance': 1e-6, 'dmc_jax_refinement_passes': 2})
    def objective(x):
        return rosen(x), rosen_der(x)
    result = optimize(context, objective, np.array([-1.2, 1.0]), [(-2., 2.), (-1., 3.)])
    assert len(result.refinement_history) > 1
    assert result.fun < 1e-12
    assert result.gradient_converged
    assert result.projected_gradient_norm <= 1e-6
    assert result.nit <= 300


def test_convergence_allows_gradient_pointing_out_of_active_bound():
    def objective(x):
        return float((x[0] + 1)**2), np.array([2 * (x[0] + 1)])
    result = optimize(Context(), objective, np.array([1.]), [(0., None)])
    assert result.x[0] == pytest.approx(0.)
    assert result.jac[0] == pytest.approx(2.)
    assert result.gradient_converged
    assert projected_gradient_norm(result.x, result.jac, [(0., None)]) == 0.


@pytest.mark.parametrize('distance_interaction', [False, True])
@pytest.mark.parametrize('income_interaction', [False, True])
@pytest.mark.parametrize('use_income', [False, True])
def test_vot_matches_utility_derivatives_and_fixed_exponents(
        distance_interaction, income_interaction, use_income):
    context = Context({'distance_cost_interaction': distance_interaction,
                       'income_cost_interaction': income_interaction,
                       'use_income_in_dmc': use_income})
    parameters = {'beta_car_travel_time_min': -1.2, 'beta_pt_in_vehicle_time_min': -.3,
                  'beta_pt_access_egress_time_min': -.7, 'beta_pt_transfer_time_min': -.02,
                  'beta_cost_CHF': -.12, 'lambda_cost_distance': -.3, 'lambda_cost_income': -.2}
    # Both estimators omit fixed parameters from get_beta_values().
    fixed = {'lambda_car_travel_time': Parameter('lambda_car_travel_time', .7, fixed=True),
             'lambda_pt_access_egress_time': Parameter('lambda_pt_access_egress_time', .593, fixed=True)}
    result = SimpleNamespace(get_beta_values=lambda: parameters)
    frame = pd.DataFrame({'mode': [0, 1], 'car_availability': 1, 'pt_availability': 1,
                          'euclidean_distance_km': 7.5, 'income': 3000., 'person_weight': 1.,
                          'car_travel_time_min': 30., 'parking_searching_duration_min': 5.,
                          'pt_in_vehicle_time_min': 20., 'pt_access_egress_time_min': 10.,
                          'pt_transfer_time_min': 0.}, index=[10, 20])
    cost_factor = 1.
    if distance_interaction:
        cost_factor *= .5**(-.3)
    if income_interaction and use_income:
        cost_factor *= .5**(-.2)
    def derivative(function, value):
        h = 1e-4
        return (function(value + h) - function(value - h)) / (2 * h)
    cost_derivative = derivative(lambda cost: -.12 * cost_factor * cost, 2.)
    car_time_derivative = derivative(lambda time: -1.2 * ((time + 5.) / 10.)**.7, 30.)
    car, mean_car = vot_utils.get_car_vot(context, frame, result, ['car', 'pt'], fixed)
    assert car.index.tolist() == [10]
    assert mean_car == pytest.approx(60 * car_time_derivative / cost_derivative)

    # Increase total time proportionally across positive physical components.
    pt_derivative = derivative(lambda time: -.3 * (time * 2/3) / 10
                               - .7 * ((time / 3) / 10)**.593, 30.)
    pt, mean_pt, iv, _, _ = vot_utils.get_pt_vot(context, frame, result, ['car', 'pt'], fixed)
    assert pt.index.tolist() == [20]
    assert mean_pt == pytest.approx(60 * pt_derivative / cost_derivative)
    assert iv.iloc[0] == pytest.approx(60 * (-.3 / 10) / cost_derivative)


def test_newton_refinement_reaches_stationarity_and_keeps_active_bound():
    from scipy.optimize import OptimizeResult
    from dmc.models.model_jax import newton_polish

    def objective(x):
        displacement = x - np.array([1., 2., -1.])
        curvature = np.array([1., 1000., 1.])
        return float(np.sum(curvature * displacement**2)), 2 * curvature * displacement
    x = np.array([1.001, 1.99999, 0.])
    value, gradient = objective(x)
    result = OptimizeResult(x=x, fun=value, jac=gradient)
    bounds = [(None, None), (None, None), (0., None)]
    polished, steps, _, _ = newton_polish(objective, result, bounds, 1e-8, 2)
    np.testing.assert_allclose(polished.x, [1., 2., 0.], atol=1e-10)
    assert steps == 1
    assert projected_gradient_norm(polished.x, polished.jac, bounds) < 1e-8
    assert polished.fun <= value


def test_newton_refinement_skips_indefinite_curvature():
    from scipy.optimize import OptimizeResult
    from dmc.models.model_jax import newton_polish

    def objective(x):
        return float(-x[0]**2), -2 * x
    x = np.array([.001])
    value, gradient = objective(x)
    result = OptimizeResult(x=x, fun=value, jac=gradient)
    polished, steps, _, _ = newton_polish(objective, result, [(None, None)], 1e-8, 2)
    assert steps == 0
    np.testing.assert_equal(polished.x, x)


@pytest.fixture
def vot_constraint_sample():
    from dmc.models.model_jax import define_parameters
    parameters = define_parameters(False, True, True)
    free = [p for p in parameters.values() if not p.fixed]
    x = np.array([p.init_value for p in free])
    df = pd.DataFrame({
        'mode': [0, 0, 1, 1], 'car_availability': 1, 'pt_availability': 1,
        'euclidean_distance_km': [2., 30., 5., 20.], 'income': [3000., 7000., 4000., 5000.],
        'person_weight': [1., 3., 2., 5.], 'car_travel_time_min': [5., 40., 10., 15.],
        'parking_searching_duration_min': [2., 5., 2., 2.],
        'pt_in_vehicle_time_min': [10., 30., 20., 30.],
        'pt_access_egress_time_min': [3., 5., 0., 10.],
        'pt_transfer_time_min': [0., 2., -2., 3.],
    })
    return df, parameters, free, x


@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('distance', [False, True])
@pytest.mark.parametrize('income', [False, True])
def test_differentiable_constraints_match_vot_reporter(vot_constraint_sample, weighted, distance, income):
    from dmc.vot.constraints import create_vot_function
    df, parameters, free, x = vot_constraint_sample
    context = Context({'use_weights_for_vot': weighted, 'distance_cost_interaction': distance,
                       'income_cost_interaction': income})
    values = {p.name: value for p, value in zip(free, x)}
    result = SimpleNamespace(get_beta_values=lambda: values)
    _, car = vot_utils.get_car_vot(context, df, result, ['car', 'pt'], parameters)
    _, pt, *_ = vot_utils.get_pt_vot(context, df, result, ['car', 'pt'], parameters)
    means = create_vot_function(context, df, parameters, ['car', 'pt'], free)
    np.testing.assert_allclose(means(x), [car, pt])
    import jax
    jacobian = np.asarray(jax.jacrev(means)(x))
    assert np.isfinite(jacobian).all()
    index = next(i for i, p in enumerate(free) if p.name == 'beta_cost_CHF')
    plus, minus = x.copy(), x.copy()
    plus[index] += 1e-6
    minus[index] -= 1e-6
    np.testing.assert_allclose(jacobian[:, index], (means(plus) - means(minus)) / 2e-6, rtol=1e-6)


def test_equality_constrained_fit_reaches_selected_targets():
    from scipy.optimize import OptimizeResult
    from dmc.vot.constraints import constrain
    parameters = {
        'beta_car_travel_time_min': Parameter('beta_car_travel_time_min', -.3, -10, -.002),
        'beta_pt_in_vehicle_time_min': Parameter('beta_pt_in_vehicle_time_min', -.1, -10, -.002),
        'beta_cost_CHF': Parameter('beta_cost_CHF', -.1, fixed=True),
        'beta_pt_access_egress_time_min': Parameter('beta_pt_access_egress_time_min', -.5, fixed=True),
        'beta_pt_transfer_time_min': Parameter('beta_pt_transfer_time_min', -.1, fixed=True),
        **{name: Parameter(name, 1., fixed=True) for name in (
            'lambda_car_travel_time', 'lambda_pt_in_vehicle_time', 'lambda_pt_access_egress_time',
            'lambda_pt_transfer_time')},
    }
    free = [p for p in parameters.values() if not p.fixed]
    df = pd.DataFrame({'mode': [0, 1], 'car_availability': 1, 'pt_availability': 1,
                       'person_weight': 1., 'car_travel_time_min': 30.,
                       'parking_searching_duration_min': 0., 'pt_in_vehicle_time_min': 20.,
                       'pt_access_egress_time_min': 10., 'pt_transfer_time_min': 0.})
    baseline = np.array([-.3, -.1])
    def objective(x):
        delta = x - baseline
        return float(delta @ delta), 2 * delta
    initial = OptimizeResult(x=baseline, fun=0., jac=np.zeros(2), nit=1, nfev=1,
                             refinement_history=[])
    context = Context({'distance_cost_interaction': False, 'income_cost_interaction': False})
    result = constrain(context, df, parameters, ['car', 'pt'], free, objective, initial)
    np.testing.assert_allclose(result.x, [-32/60, -.2], atol=1e-8)
    assert result.gradient_converged
    assert result.vot_constraints['car_achieved_chf_h'] == pytest.approx(32.)
    assert result.vot_constraints['pt_combined_achieved_chf_h'] == pytest.approx(18.)
    assert result.vot_constraints['log_likelihood_loss'] > 0
    assert result.parameter_comparison.loc[lambda d: d.fixed, 'change'].eq(0).all()
