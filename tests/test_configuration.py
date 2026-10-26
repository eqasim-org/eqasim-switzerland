"""Defaults must agree across stages when advanced YAML options are omitted."""
import ast
from pathlib import Path

import pytest
import yaml
from synpp.pipeline import ConfigurationContext

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('analysis', 'calibration', 'data', 'dmc', 'matsim', 'mode_choice', 'synthesis')


def test_literal_defaults_agree_across_stages():
    defaults = {}
    # These pre-existing options have stage-specific meanings/defaults.
    local_options = {'scaling_year', 'outbound_flows_perimeter'}
    for package in PACKAGES:
        for path in (ROOT / package).rglob('*.py'):
            if any(part.startswith('.') or part == '__pycache__' for part in path.relative_to(ROOT).parts):
                continue
            for call in ast.walk(ast.parse(path.read_text())):
                if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute)
                        or call.func.attr != 'config' or not call.args
                        or not isinstance(call.args[0], ast.Constant)):
                    continue
                key = call.args[0].value
                if key in local_options:
                    continue
                default = call.args[1] if len(call.args) > 1 else next(
                    (kw.value for kw in call.keywords if kw.arg == 'default'), None)
                if default is None:
                    continue
                try:
                    value = ast.literal_eval(default)
                except (ValueError, TypeError):
                    continue
                location = f'{path.relative_to(ROOT)}:{call.lineno}'
                if key in defaults:
                    previous, previous_location = defaults[key]
                    assert value == previous, f'{key}: {previous_location} and {location} disagree'
                defaults[key] = (value, location)


def test_omitted_dmc_settings_enable_selected_targets_and_explicit_override_works():
    from dmc.models import model_jax
    context = ConfigurationContext({})
    model_jax.configure(context)
    assert context.required_config['dmc_vot_constraints'] is True
    assert context.required_config['dmc_vot_car_target'] == 32.
    assert context.required_config['dmc_vot_pt_target'] == 18.
    context = ConfigurationContext({'dmc_vot_constraints': False})
    model_jax.configure(context)
    assert context.required_config['dmc_vot_constraints'] is False


def test_omitted_simulation_options_match_requested_defaults():
    from matsim.simulation import prepare
    context = ConfigurationContext({'input_downsampling': .1, 'threads': 2, 'random_seed': 1})
    prepare.configure(context)
    assert context.required_config['last_iteration'] == 140
    assert context.required_config['calibrate_alphas_in_matsim'] is True
    assert context.required_config['alphaCalibration.level'] == 'canton'
    assert context.required_config['network_calibration.activate'] is True
    assert context.required_config['useScheduleBasedTransport'] is False


def test_beginner_template_has_unique_keys_and_explanations():
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            assert key not in result, f'Duplicate config key: {key}'
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    text = (ROOT / 'config.yml').read_text()
    config = yaml.load(text, Loader=UniqueLoader)['config']
    assert not any(key.startswith(('dmc_jax_', 'dmc_vot_', 'network_calibration.')) for key in config)
    assert 'data_path' in config and 'external_population_folder' in config
    assert '/cluster/' not in text
    for line in text.splitlines():
        if line.startswith('    ') and not line.startswith('        ') and not line.lstrip().startswith(('#', '-')):
            if ':' in line:
                assert '#' in line, f'Missing config explanation: {line}'


@pytest.mark.parametrize('enabled', [True, False])
def test_uphill_option_controls_correction(monkeypatch, enabled):
    from types import SimpleNamespace
    from matsim.scenario.network.utils import network_handler
    calls = []
    class Context:
        def config(self, key):
            return {'correct_speed_uphill': enabled, 'assign_elevations': True}[key]
    class Corrector:
        def __init__(self, context, network):
            pass
        def run(self, kind):
            calls.append(kind)
            return 'corrected links'
    monkeypatch.setattr(network_handler, 'SpeedCorrector', Corrector)
    handler = network_handler.NetworkHandler.__new__(network_handler.NetworkHandler)
    handler.context = Context()
    handler.net = SimpleNamespace(links='original links')
    handler._adjust_uphill_speed_if_requested()
    assert calls == (['uphill'] if enabled else [])
    assert handler.net.links == ('corrected links' if enabled else 'original links')
