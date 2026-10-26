from pathlib import Path
import shutil

import pytest

from matsim import output
from matsim.config import tools
from matsim.simulation import prepare
from test_matsim_config import Context, base, pt_parameters, stage_inputs, fake_java, value


@pytest.mark.parametrize('active', [True, False])
def test_prepare_copies_assets_and_uses_relative_calibration_paths(
        tmp_path, monkeypatch, base, pt_parameters, active):
    stages = stage_inputs(tmp_path, base, pt_parameters)
    inputs = tmp_path / 'inputs'
    Path(stages['data.microcensus.shares'][0]).write_text(
        'mode,mode_share\ncar,0.4\npt,0.3\nwalk,0.15\nbike,0.1\ncar_passenger,0.05\n')
    geometry = inputs / 'geometry.csv'
    geometry.write_text('linkId,geometry\n1,LINESTRING(0 0;1 1)\n')
    stages['matsim.scenario.network.pt'] = stages['matsim.scenario.network.mapped']
    stages['matsim.scenario.network.pt']['geometry'] = str(geometry)
    source_files = {
        'analysis.counts.target': ['counts.csv'],
        'analysis.travel_times.APIs.target': ['trips.csv'],
        'calibration.road_regions.penalty_calibration': ['penalty_0.yml', 'penalty_1.yml'],
        'calibration.road_regions.freespeed_calibration': ['speed.yml'],
    }
    for stage, names in source_files.items():
        paths = []
        for name in names:
            path = inputs / name
            path.write_text(name)
            paths.append(str(path))
        stages[stage] = ';'.join(paths)

    context = Context(tmp_path / 'prepare', {
        'input_downsampling': .1, 'threads': 2, 'random_seed': 7,
        'network_calibration.activate': active,
    }, stages)
    java = fake_java([])
    def check_routing(ctx, command, arguments):
        if active and command.endswith('RunPopulationRouting'):
            assert value(ctx.directory / 'switzerland_config.xml',
                         'eqasim:networkCalibration.countsFile') == 'network_calibration_files/target_counts.csv'
        java(ctx, command, arguments)
    monkeypatch.setattr(prepare.eqasim, 'run', check_routing)
    monkeypatch.setattr(prepare, 'patch_directional_border_links', lambda *args, **kwargs: None)
    prepare.configure(context)
    config_path = prepare.execute(context)
    assert (context.directory / 'switzerland_detailed_network.csv').read_bytes() == geometry.read_bytes()
    expected = {
        'eqasim:networkCalibration.countsFile': ['target_counts.csv'],
        'eqasim:networkCalibration.freespeedCalibration.observedTripsFile': ['target_travel_times.csv'],
        'eqasim:networkCalibration.costCalibration.specialRegionPath': [
            'penalties_special_region_0.yml', 'penalties_special_region_1.yml'],
        'eqasim:networkCalibration.freespeedCalibration.specialRegionPath': ['freespeed_special_region_0.yml'],
    }
    if not active:
        assert not (context.directory / 'network_calibration_files').exists()
        assert not any(stage in context.dependencies for stage in source_files)
        return

    # The copied scenario must remain usable after removing upstream inputs.
    shutil.rmtree(inputs)
    for parameter, names in expected.items():
        relative_paths = value(config_path, parameter).split(';')
        assert relative_paths == [f'network_calibration_files/{name}' for name in names]
        for path in relative_paths:
            assert not Path(path).is_absolute()
            assert (context.directory / path).is_file()

    results = tmp_path / 'run' / 'simulation_output'
    results.mkdir(parents=True)
    shutil.copyfile(config_path, results / 'output_config.xml')
    (results / 'output_events.xml.gz').write_bytes(b'events')
    contracts = tmp_path / 'contracts.html'
    contracts.write_text('contracts')
    class OutputContext(Context):
        def path(self, stage=None):
            if stage == 'matsim.simulation.prepare':
                return str(context.directory)
            if stage == 'matsim.simulation.run':
                return str(results.parent)
            return super().path(stage)
    exported = OutputContext(tmp_path / 'export_stage', {
        'output_path': str(tmp_path / 'export'), 'output_id': 'scenario',
        'write_jar': False,
    }, {'contracts.contracts': str(contracts), 'matsim.cutter.run_scenario': ('', '')})
    output.configure(exported)
    output.execute(exported)
    destination = tmp_path / 'export' / 'scenario'
    assert (destination / 'switzerland_detailed_network.csv').read_bytes() == (
        context.directory / 'switzerland_detailed_network.csv').read_bytes()
    for parameter, names in expected.items():
        for path in value(destination / 'switzerland_config.xml', parameter).split(';'):
            assert not Path(path).is_absolute()
            assert (destination / path).is_file()
    assert (results / 'output_events.xml.gz').exists()
    assert not any(stage in exported.dependencies for stage in source_files)
    assert 'matsim.scenario.network.mapped' not in exported.dependencies
