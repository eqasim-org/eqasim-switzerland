from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import pytest
from biogeme.database import Database
from biogeme.jax_calculator.single_formula import evaluate_expression

from calibration.dmc.default import default_mode_params, default_cost_params
from dmc.models import model_biogeme, model_jax
from dmc.models.specification import LINEAR_TERMS
from dmc.parameters import NAMES_CONVERSION, Parameters, writer
from dmc.pt_quality import encode_pt_quality, pt_quality_expression
from mode_choice.model.utilities import expressions


class Context:
    def __init__(self, path):
        self.directory = path
        self.values = {}

    def config(self, name, default=None, **kwargs):
        return self.values.setdefault(name, default)

    def stage(self, name, **kwargs):
        if name == 'data.constants':
            return SimpleNamespace(LOW_INCOME_THRESHOLD=2000, HIGH_INCOME_THRESHOLD=6000)

    def path(self):
        return str(self.directory)


QUALITY_PARAMETERS = ('beta_pt_home_quality', 'gamma_pt_home_quality',
                      'beta_pt_destination_quality', 'gamma_pt_destination_quality')


@pytest.mark.parametrize('categorical', [False, True])
def test_pandas_and_polars_encode_identical_quality(categorical):
    values = pd.Series(['A', 'B', 'C', 'D', 'None', None, 'a', ' b ', 'none', '', 'unclassified'],
                       index=range(20, 31))
    if categorical:
        values = values.astype('category')
    expected = [0, 1, 2, 3, 4, 4, 0, 1, 4, 4, 4]
    actual = encode_pt_quality(values)
    assert actual.tolist() == expected
    assert actual.index.equals(values.index)
    assert pl.from_pandas(values.to_frame('ovgk')).select(pt_quality_expression('ovgk')).to_series().to_list() == expected


@pytest.mark.parametrize('use_exponents', [False, True])
def test_both_estimators_train_quality_coefficients_and_exponents(use_exponents):
    jax_parameters = model_jax.define_parameters(False, use_exponents, True)
    biogeme_parameters = model_biogeme.define_betas(False, use_exponents, True)
    for name in QUALITY_PARAMETERS:
        assert not jax_parameters[name].fixed
        assert biogeme_parameters[name].status == 0
        expected_initial_value = 1.0 if name.startswith('gamma_') else -0.05
        assert jax_parameters[name].init_value == expected_initial_value
        assert biogeme_parameters[name].init_value == expected_initial_value
    for name in ('gamma_pt_home_quality', 'gamma_pt_destination_quality'):
        assert jax_parameters[name].lower_bound > 0
        assert biogeme_parameters[name].lower_bound > 0
    assert not any('good_service' in name or 'medium_service' in name for name in jax_parameters)


@pytest.fixture
def model_inputs(tmp_path):
    context = Context(tmp_path)
    model_jax.configure(context)
    context.values.update(distance_cost_interaction=False, income_cost_interaction=False)
    parameters = model_jax.define_parameters(False, True, True)
    # Isolate PT quality while evaluating the complete utility implementation.
    betas = {name: (1.0 if name.startswith(('lambda_', 'gamma_')) else 0.0) for name in parameters}
    betas.update(beta_pt_home_quality=-.7, gamma_pt_home_quality=1.3,
                 beta_pt_destination_quality=-.4, gamma_pt_destination_quality=.8)
    columns = {column for terms in LINEAR_TERMS.values() for _, column in terms}
    columns.update(model_biogeme.define_variables(None, False))
    columns.update(['destination_employee_density', 'destination_population_density',
                    'destination_companies_density', 'parking_cost_CHF', 'income'])
    frame = pd.DataFrame({column: np.ones(5) for column in columns})
    frame['homePtQuality'] = np.arange(5)
    frame['destinationPtQuality'] = np.arange(4, -1, -1)
    for mode in model_jax.MODES:
        frame[f'{mode}_availability'] = True
    return context, parameters, betas, frame


def test_writer_and_all_utility_backends_use_nonlinear_quality(model_inputs):
    context, parameters, betas, frame = model_inputs
    result = SimpleNamespace(get_beta_values=lambda: betas.copy())
    mode_path, cost_path = writer(context, result, parameters).write()
    exported = Parameters.from_yaml(mode_path, cost_path)
    for name in QUALITY_PARAMETERS:
        assert exported[NAMES_CONVERSION[name]] == betas[name]
    assert exported['pt.betaHomePtQuality_u'] == betas['beta_pt_home_quality']
    assert exported['pt.homePtQualityExponent'] == betas['gamma_pt_home_quality']
    assert not any('GoodService' in name or 'MediumService' in name for name in exported)
    # The parameter container must preserve the new fields through a second write/read.
    copy = Path(context.path()) / 'roundtrip.yml'
    exported.to_yaml(copy)
    assert Parameters.from_yaml(copy).to_dict() == exported.to_dict()

    expected = -.7 * np.arange(5)**1.3 - .4 * np.arange(4, -1, -1)**.8
    arrays = {column: jnp.asarray(frame[column].to_numpy()) for column in frame}
    jax_utilities, _ = model_jax.build_utilities(context, arrays, betas, model_jax.MODES, False)
    np.testing.assert_allclose(np.asarray(jax_utilities)[:, 1], expected)

    polars_utilities = pl.from_pandas(frame).select(expressions(
        exported, distance_interaction=False, income_interaction=False))
    np.testing.assert_allclose(polars_utilities['utility_pt'], expected)
    variables = model_biogeme.define_variables(None, False)
    biogeme_utilities, _ = model_biogeme.build_utilities(context, variables, betas, model_jax.MODES, False)
    numeric = frame.astype(float).rename(columns={'weight': 'person_weight'})
    actual = evaluate_expression(biogeme_utilities[1], database=Database('quality', numeric),
                                 aggregation=False, numerically_safe=True, use_jit=False)
    np.testing.assert_allclose(actual, expected)


def test_exponent_gradients_are_finite_at_quality_zero(model_inputs):
    context, _, betas, frame = model_inputs
    frame['homePtQuality'] = 0
    frame['destinationPtQuality'] = [0, 1, 2, 3, 4]
    arrays = {column: jnp.asarray(frame[column].to_numpy()) for column in frame}
    def utility(gammas):
        fitted = dict(betas, gamma_pt_home_quality=gammas[0], gamma_pt_destination_quality=gammas[1])
        return model_jax.build_utilities(context, arrays, fitted, model_jax.MODES, False)[0][:, 1].sum()
    gammas = jnp.array([.6, 1.4])
    gradient = np.asarray(jax.grad(utility)(gammas))
    assert np.isfinite(gradient).all()
    assert gradient[0] == 0
    q = np.arange(1, 5)
    assert gradient[1] == pytest.approx(-.4 * np.sum(q**1.4 * np.log(q)))


def test_default_parameters_support_new_utility_schema(model_inputs):
    _, _, _, frame = model_inputs
    defaults = Parameters({**default_mode_params, **default_cost_params})
    for name in QUALITY_PARAMETERS:
        assert NAMES_CONVERSION[name] in defaults
    frame = frame.assign(canton_name_en='Geneva')
    utilities = pl.from_pandas(frame).select(expressions(defaults))
    assert utilities['utility_pt'].is_finite().all()


@pytest.mark.parametrize('preprocess', [model_jax.preprocess_data, model_biogeme.preprocess_data])
def test_estimator_preprocessing_preserves_numeric_quality(model_inputs, preprocess):
    _, _, _, frame = model_inputs
    frame = frame.drop(columns='weight').assign(
        person_id=np.arange(5), trip_id=np.arange(5), person_weight=1.0,
        mode=model_jax.MODES, home_municipality='rural', origin_municipality='rural',
        destination_municipality='urban', sp_region=1, ms_region=1, ovgk='A',
        pt_egress_time_min=1.0, pt_access_time_min=1.0, income_class=1,
        actual_parking_duration_min=1.0, purpose='work')
    processed, modes = preprocess(frame, False)
    assert modes == model_jax.MODES
    assert processed['homePtQuality'].tolist() == [0, 1, 2, 3, 4]
    assert processed['destinationPtQuality'].tolist() == [4, 3, 2, 1, 0]
    assert not any('good_pt_service' in column or 'medium_pt_service' in column for column in processed)


def test_constrained_writer_preserves_fitted_precision(model_inputs):
    context, parameters, betas, _ = model_inputs
    context.values['dmc_vot_constraints'] = True
    betas['beta_cost_CHF'] = -.115884912345678
    betas['beta_pt_in_vehicle_time_min'] = -.511403912345678
    model = SimpleNamespace(get_beta_values=lambda: betas.copy())
    mode_path, cost_path = writer(context, model, parameters).write()
    exported = Parameters.from_yaml(mode_path, cost_path)
    assert exported['betaCost_u_MU'] == betas['beta_cost_CHF']
    assert exported['pt.betaInVehicleTime_u_min'] == betas['beta_pt_in_vehicle_time_min']
