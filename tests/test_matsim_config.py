from pathlib import Path
import gzip
import shutil
from lxml import etree
import pytest

from matsim.config import config, tools
from matsim.simulation import prepare, run, run_from_inputs
from matsim.config import pt as pt_config
from mode_choice.routing import config as mode_routing


class Context:
    def __init__(self, path, values=None, stages=None):
        self.directory = path
        path.mkdir(parents=True, exist_ok=True)
        self.values = dict(values or {})
        self.stages = stages or {}
        self.dependencies = set()
        self.working_directory = str(path.parent / "working")
    def config(self, key, default=None, **kwargs):
        return self.values.setdefault(key, default)
    def stage(self, key, **kwargs):
        self.dependencies.add(key)
        return self.stages.get(key)
    def path(self, stage=None):
        assert stage is None
        return str(self.directory)


def value(path, parameter):
    tree = tools.read_config(path)
    parts = parameter.split('.')
    parent = tree.find(f"module[@name='{parts[0]}']")
    for part in parts[1:-1]:
        parent = parent.find(f"parameterset[@type='{part}']")
    return parent.find(f"param[@name='{parts[-1]}']").get('value')


@pytest.fixture
def base(tmp_path):
    path = tmp_path / 'shared.xml'
    path.write_text('<?xml version="1.0"?><!DOCTYPE config SYSTEM "http://www.matsim.org/files/dtd/config_v2.dtd"><config><!-- keep comment --></config>')
    tools.change_params(path, {
        'plans.inputPlansFile': 'switzerland_population.xml.gz',
        'households.inputFile': 'switzerland_households.xml.gz',
        'facilities.inputFacilitiesFile': 'switzerland_facilities.xml.gz',
        'vehicles.vehiclesFile': 'switzerland_vehicles.xml.gz',
        'network.inputNetworkFile': 'switzerland_network.xml.gz',
        'transit.transitScheduleFile': 'switzerland_transit_schedule.xml.gz',
        'transit.vehiclesFile': 'switzerland_transit_vehicles.xml.gz',
        'DiscreteModeChoice.modeAvailability': 'original',
        'eqasim.routingDistanceUtility': .73,
        'eqasim.costParametersPath': 'cost_parameters.yml',
        'custom.marker': 'keep this',
    }, create=True)
    tools.add_ptZones_module(path)
    tree = tools.read_config(path)
    replanning = tools.get_module(tree, 'replanning')
    for group in ('freight', 'crossborder', 'null'):
        item = tools.get_parameter_set(replanning, 'strategysettings', {'subpopulation': group, 'strategyName': 'ReRoute'})
        tools.set_param('weight', .05, item)
    tools.write_config(tree, path)
    return path


@pytest.fixture
def pt_parameters():
    return {'travelTimeRail_u_h': -1.0, 'travelTimeBus_u_h': -1.5,
            'waitTime_u_h': -1.7, 'raptorPenalties:transfer_rail': .1,
            'raptorPenalties:transfer_intermodal': .2,
            'raptorPenalties:transfer_other': .3}


def test_tools_strict_creation_indices_comments_and_dedup(base):
    before = base.read_bytes()
    with pytest.raises(ValueError, match='Could not find'):
        tools.change_params(base, {'missing.a': 3})
    assert base.read_bytes() == before
    tools.change_params(base, {'new.nested.active': True, 'new.nested.path': None}, create=True)
    assert value(base, 'new.nested.active') == 'true'
    assert value(base, 'new.nested.path') == 'null'
    tree = tools.read_config(base)
    module = tools.get_module(tree, 'custom')
    for n in ('a', 'b'):
        item = tools.get_parameter_set(module, 'repeat', {'key': n})
        tools.set_param('v', n, item)
    tools.write_config(tree, base)
    tools.change_params(base, {'custom.repeat[1].v': 'updated'})
    assert tools.read_config(base).findall("module[@name='custom']/parameterset")[1].find("param[@name='v']").get('value') == 'updated'
    assert '<!-- keep comment -->' in base.read_text()
    assert '<!DOCTYPE config SYSTEM' in base.read_text()


def test_pt_updates_are_idempotent(base, pt_parameters):
    for _ in range(2):
        tools.add_SBBPT_module(base)
        tools.add_ptZones_module(base, base.parent)
        tools.adjust_pt_routing_parameters(base, pt_parameters)
    tree = tools.read_config(base)
    assert len(tree.findall("module[@name='SBBTransit']")) == 1
    assert len(tree.findall("module[@name='ptZones']")) == 1
    assert len(tree.findall("module[@name='swissRailRaptor']/parameterset")) == 81
    assert value(base, 'eqasim:raptor.travelTimeRail_u_h') == '-1.0'
    assert value(base, 'ptZones.ptZonesFilePath') == str(base.parent / 'gtfs_zones.csv')


@pytest.mark.parametrize('schedule', [False, True])
def test_shared_stage_runs_generation_once_without_population_dependencies(tmp_path, monkeypatch, base, schedule):
    context = Context(tmp_path / 'generate', {
        'input_downsampling': .1, 'random_seed': 7, 'threads': 4,
        'use_freight': True, 'useScheduleBasedTransport': schedule,
    }, {'calibration.car_routing_vot.optimal_value': .73})
    calls = []
    def java(ctx, command, args):
        flags = dict(zip(args[::2], args[1::2]))
        calls.append((command, flags))
        if command.endswith('RunGenerateConfig'):
            shutil.copyfile(base, flags['--output-path'])
    monkeypatch.setattr(config.eqasim, 'run', java)
    config.configure(context)
    result = config.execute(context)
    assert len(calls) == 2
    assert calls[0][0] == 'org.eqasim.core.scenario.config.RunGenerateConfig'
    assert calls[1][0] == 'org.eqasim.switzerland.ch_cmdp.scenario.RunAdaptConfig'
    assert calls[1][1]['--hasFreight'] == 'true'
    assert context.dependencies == {'matsim.runtime.java', 'matsim.runtime.eqasim', 'calibration.car_routing_vot.optimal_value'}
    assert calls[1][1]['--routingDistanceUtility'] == .73
    assert bool(tools.read_config(result).findall("module[@name='SBBTransit']")) == (not schedule)
    assert Path(result).is_absolute()


def stage_inputs(tmp_path, base, pt_parameters):
    inputs = tmp_path / 'inputs'
    inputs.mkdir()
    files = {}
    for name in ['network', 'schedule', 'population', 'facilities', 'vehicles', 'households', 'mode.yml', 'cost.yml', 'global.csv', 'cantonal.csv', 'sbb.csv', 'zones.csv', 'pricing.xml']:
        p = inputs / name
        p.write_text(name)
        files[name] = str(p)
    vehicles = inputs / 'transit_vehicles.xml.gz'
    with gzip.open(vehicles, 'wt') as stream:
        stream.write('<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd"><vehicleType id="bus"><networkMode networkMode="bus"/><passengerCarEquivalents pce="2"/></vehicleType></vehicleDefinitions>')
    return {
        'matsim.config.config': str(base),
        'matsim.scenario.network.mapped': {'network': files['network'], 'schedule': files['schedule'], 'vehicles': str(vehicles)},
        'matsim.scenario.population': files['population'],
        'matsim.scenario.facilities': files['facilities'],
        'matsim.scenario.vehicles': files['vehicles'],
        'matsim.scenario.households': files['households'],
        'dmc.parameter_files': (files['mode.yml'], files['cost.yml']),
        'data.microcensus.shares': (files['global.csv'], files['cantonal.csv']),
        'data.pt_pricing.pt_pricing': (files['sbb.csv'], files['zones.csv'], files['pricing.xml']),
        'calibration.pt_routing.pt_routing_parameters': pt_parameters,
        'calibration.car_routing_vot.optimal_value': .73,
        'analysis.counts.target': '/calibration/counts.csv',
        'analysis.travel_times.APIs.target': '/calibration/trips.csv',
        'calibration.road_regions.penalty_calibration': '/calibration/penalties.yml',
        'calibration.road_regions.freespeed_calibration': '/calibration/speeds.yml',
    }


def fake_java(calls):
    def run(context, command, args):
        assert not command.endswith(('RunGenerateConfig', 'RunAdaptConfig'))
        calls.append(command)
        flags = dict(zip(args[::2], args[1::2]))
        if command.endswith('RunPreparation'):
            for name in ('facilities', 'population', 'network'):
                (Path(context.path()) / flags[f'--output-{name}-path']).write_text(name)
        elif command.endswith(('RunCalculateStopCategories', 'RunPopulationRouting')):
            path = Path(flags['--output-path'])
            (path if path.is_absolute() else Path(context.path()) / path).write_text('output')
    return run


@pytest.mark.parametrize('active', [False, True])
@pytest.mark.parametrize('schedule', [False, True])
def test_prepare_preserves_local_calibration_and_files(tmp_path, monkeypatch, base, pt_parameters, active, schedule):
    stages = stage_inputs(tmp_path, base, pt_parameters)
    original = base.read_bytes()
    context = Context(tmp_path / 'prepare', {
        'input_downsampling': .1, 'threads': 4, 'random_seed': 7,
        'network_calibration.activate': active, 'useScheduleBasedTransport': schedule,
    }, stages)
    calls = []
    monkeypatch.setattr(prepare.eqasim, 'run', fake_java(calls))
    monkeypatch.setattr(prepare, 'patch_directional_border_links', lambda *args, **kwargs: None)
    prepare.configure(context)
    prepared = prepare.execute(context)
    result = Path(prepared['config'])
    assert base.read_bytes() == original
    assert 'matsim.config.config' in context.dependencies
    assert value(result, 'eqasim.routingDistanceUtility') == '0.73'
    assert value(result, 'custom.marker') == 'keep this'
    assert value(result, 'ptZones.ptZonesFilePath') == str(context.directory / 'gtfs_zones.csv')
    if active:
        assert value(result, 'eqasim:networkCalibration.countsFile') == '/calibration/counts.csv'
        assert value(result, 'eqasim:networkCalibration.freespeedCalibration.observedTripsFile') == '/calibration/trips.csv'
        assert value(result, 'eqasim:networkCalibration.costCalibration.specialRegionPath') == '/calibration/penalties.yml'
    assert (context.directory / 'switzerland_population.xml.gz').exists()
    assert not (context.directory / 'prepared_population.xml.gz').exists()


@pytest.mark.parametrize('schedule', [False, True])
def test_pt_configs_copy_base_and_disable_population(tmp_path, monkeypatch, base, pt_parameters, schedule):
    stage = pt_config
    stages = stage_inputs(tmp_path, base, pt_parameters)
    original = base.read_bytes()
    context = Context(tmp_path / 'pt', {'input_downsampling': .1, 'threads': 4, 'random_seed': 7,
                                      'useScheduleBasedTransport': schedule}, stages)
    monkeypatch.setattr(stage.eqasim, 'run', fake_java([]))
    stage.configure(context)
    result = stage.execute(context)
    assert base.read_bytes() == original
    for parameter in ('facilities.inputFacilitiesFile', 'households.inputFile', 'plans.inputPlansFile', 'vehicles.vehiclesFile'):
        assert value(result, parameter) == 'null'
    assert not tools.read_config(result).xpath("//parameterset[param[@name='subpopulation'][@value='freight']]")
    assert value(result, 'ptZones.ptZonesFilePath') == str(context.directory / 'gtfs_zones.csv')
    assert value(result, 'eqasim:raptor.travelTimeRail_u_h') == '-1.0'
    assert 'calibration.pt_routing.pt_routing_parameters' in context.dependencies
    assert value(result, 'eqasim.routingDistanceUtility') == '0.73'


def test_mode_choice_resolves_inputs_and_overrides_costs(tmp_path, base):
    original = base.read_bytes()
    context = Context(tmp_path / 'mode', stages={
        'matsim.config.pt': str(base),
        'dmc.parameter_files': ('/fit/mode.yml', '/fit/cost.yml'),
    })
    result = mode_routing.execute(context)
    assert value(result, 'transit.transitScheduleFile') == (base.parent / 'switzerland_transit_schedule.xml.gz').as_uri()
    assert value(result, 'eqasim.costParametersPath') == '/fit/cost.yml'
    assert base.read_bytes() == original


def test_external_scenario_keeps_custom_config_and_updates_local_inputs(tmp_path, monkeypatch, base):
    inputs = tmp_path / 'external'
    inputs.mkdir()
    shutil.copyfile(base, inputs / 'my_config.xml')
    for name in ('network','population','facilities','households','transit_schedule','transit_vehicles','vehicles'):
        (inputs / f'custom_{name}.xml.gz').write_text(name)
    original = (inputs / 'my_config.xml').read_bytes()
    context = Context(tmp_path / 'run_external', {
        'simulation_inputs_path': str(inputs), 'simulation_config_path': 'my_config.xml',
        'output_prefix': 'custom_', 'input_downsampling': .2, 'use_freight': False,
    })
    calls = []
    def java(ctx, command, arguments):
        assert command.endswith('RunSimulation')
        flags = dict(zip(arguments[::2], arguments[1::2]))
        folder = Path(flags['--config:controler.outputDirectory'])
        folder.mkdir()
        (folder / 'output_events.xml.gz').touch()
        calls.append(command)
    monkeypatch.setattr(run_from_inputs.eqasim, 'run', java)
    run_from_inputs.configure(context)
    run_from_inputs.execute(context)
    path = context.directory / 'experiment/my_config.xml'
    assert value(path, 'custom.marker') == 'keep this'
    assert value(path, 'qsim.flowCapacityFactor') == '0.2'
    assert value(path, 'vehicles.vehiclesFile') == 'custom_vehicles.xml.gz'
    assert value(path, 'eqasim.costParametersPath') == (inputs / 'cost_parameters.yml').as_uri()
    assert (inputs / 'my_config.xml').read_bytes() == original
    assert len(calls) == 1


def test_nested_paths_survive_config_relocation(tmp_path, base):
    tools.change_params(base, {
        'eqasim:networkCalibration.freespeedCalibration.observedTripsFile': 'targets/trips.csv',
        'eqasim:networkCalibration.costCalibration.specialRegionPath': 'regions/a.yml;regions/b.yml',
    }, create=True)
    result = tmp_path / 'relocated.xml'
    tools.copy_config(base, result, input_paths={
        'eqasim:networkCalibration.freespeedCalibration': ('observedTripsFile',),
        'eqasim:networkCalibration.costCalibration': ('specialRegionPath',),
    })
    assert value(result, 'eqasim:networkCalibration.freespeedCalibration.observedTripsFile') == (base.parent / 'targets/trips.csv').as_uri()
    assert value(result, 'eqasim:networkCalibration.costCalibration.specialRegionPath') == ';'.join((base.parent / f'regions/{letter}.yml').as_uri() for letter in ('a','b'))


def test_mapper_preserves_geometry_and_assignment_options(tmp_path):
    from matsim.scenario.network.mapped import write_mapping_config
    context = Context(tmp_path / 'mapper', {'threads': 3, 'route_bike': True, 'export_detailed_network': False})
    (context.directory / 'map_network_template.xml').write_text('<config><module name="PublicTransitMapping"><parameterset type="transportModeAssignment"><param name="scheduleMode" value="bus"/><param name="networkModes" value="wrong"/></parameterset></module></config>')
    result = write_mapping_config(context, '/network.xml.gz', '/schedule.xml.gz')
    assert value(result, 'PublicTransitMapping.inputNetworkFile') == '/network.xml.gz'
    assert value(result, 'PublicTransitMapping.splitLinksAtStops') == 'true'
    assert value(result, 'PublicTransitMapping.inputNetworkGeometryFile') == ''
    assert 'bike' in value(result, 'PublicTransitMapping.modesToKeepOnCleanUp')
    items = tools.read_config(result).findall("module/parameterset")
    assert len(items) == 5
    assert items[0].find("param[@name='networkModes']").get('value') == 'bus'
    rail = next(item for item in items if item.find("param[@name='scheduleMode']").get('value') == 'rail')
    assert rail.find("param[@name='strictLinkRule']").get('value') == 'true'


def test_osm_converter_preserves_modes_speeds_and_restrictions(tmp_path, monkeypatch):
    from matsim.scenario.network import convert_osm_pt2matsim as converter
    context = Context(tmp_path / 'osm', {'export_detailed_network': True}, {'data.osm.clean': '/data/input.osm.pbf'})
    def java(ctx, main, args, options):
        if main.endswith('CreateDefaultOsmConfig'):
            (context.directory / args[0]).write_text('<config><module name="OsmConverter"><parameterset type="routableSubnetwork"><param name="allowedTransportModes" value="bus, car"/><param name="subnetworkMode" value="bus"/></parameterset><parameterset type="wayDefaultParams"><param name="allowedTransportModes" value="car"/><param name="osmValue" value="motorway"/><param name="freespeed" value="1"/></parameterset></module></config>')
    monkeypatch.setattr(converter.pt2matsim, 'run', java)
    converter.configure(context)
    converter.execute(context)
    result = context.directory / 'convert_network.xml'
    assert value(result, 'OsmConverter.osmFile') == '/data/input.osm.pbf'
    assert value(result, 'OsmConverter.parseTurnRestrictions') == 'true'
    assert value(result, 'OsmConverter.outputDetailedLinkGeometryFile') == str(context.directory / 'detailed_network.csv')
    items = tools.read_config(result).findall("module/parameterset")
    way = next(item for item in items if item.get('type') == 'wayDefaultParams')
    assert way.find("param[@name='allowedTransportModes']").get('value') == 'car,taxi,bus'
    assert float(way.find("param[@name='freespeed']").get('value')) == converter.DEFAULT_SPEEDS['motorway']
    assert len([item for item in items if item.get('type') == 'routableSubnetwork']) == 4


@pytest.mark.parametrize('freight', [True,False])
def test_external_adaptation_is_idempotent(base, freight):
    for _ in range(2):
        tools.adapt_existing_scenario(base, .1, 'local_', freight)
    tree = tools.read_config(base)
    strategies = tree.findall("module[@name='replanning']/parameterset")
    assert len(strategies) == (6 if freight else 4)
    assert value(base, 'qsim.storageCapacityFactor') == str(.1 ** .75)
    assert value(base, 'custom.marker') == 'keep this'


@pytest.mark.parametrize('vdf', [False, True])
@pytest.mark.parametrize('calibration', ['none', 'alpha', 'beta'])
def test_prepare_writes_complete_simulation_config_and_run_only_launches(
        tmp_path, monkeypatch, base, pt_parameters, vdf, calibration):
    stages = stage_inputs(tmp_path, base, pt_parameters)
    global_shares, _ = stages['data.microcensus.shares']
    Path(global_shares).write_text('mode,mode_share\ncar,0.4\npt,0.3\nwalk,0.15\nbike,0.1\ncar_passenger,0.05\n')
    context = Context(tmp_path / 'prepare_full', {
        'input_downsampling': .1, 'threads': 24, 'random_seed': 7,
        'last_iteration': 101, 'use_vdf': vdf, 'useScheduleBasedTransport': False,
        'network_calibration.activate': True,
        'calibrate_alphas_in_matsim': calibration == 'alpha',
        'calibrate_betas_in_matsim': calibration == 'beta',
        'alphaCalibration.level': 'canton',
        'preventwaitingtoentertraffic': True, 'writeexperiencedplans': True,
        'activate_traffic_light_delays': False,
        'activate_unsignalized_intersections_delays': True,
    }, stages)
    original_java = fake_java([])
    def prepare_java(ctx, command, args):
        if command.endswith(('RunPopulationRouting', 'RunScenarioValidator')):
            config_path = context.directory / 'switzerland_config.xml'
            assert not tools.read_config(config_path).xpath(
                "module[@name='eqasim:alphaCalibration' or @name='eqasim:calibration']/param[@name='activate'][@value='true']")
        original_java(ctx, command, args)
    monkeypatch.setattr(prepare.eqasim, 'run', prepare_java)
    monkeypatch.setattr(prepare, 'patch_directional_border_links', lambda *args, **kwargs: None)
    prepare.configure(context)
    simulation = prepare.execute(context)
    path = simulation['config']
    assert value(path, 'controller.lastIteration') == '101'
    assert value(path, 'controller.writeEventsInterval') == '50'
    assert value(path, 'controller.writePlansInterval') == '50'
    assert value(path, 'controller.writeTripsInterval') == ('101' if vdf else '50')
    assert value(path, 'linkStats.writeLinkStatsInterval') == ('101' if vdf else '50')
    assert value(path, 'linkStats.averageLinkStatsOverIterations') == '8'
    assert value(path, 'qsim.numberOfThreads') == ('16' if vdf else '12')
    assert value(path, 'eqasim.useScheduleBasedTransport') == ('true' if vdf else 'false')
    assert value(path, 'eqasim.costParametersPath') == str(context.directory / 'cost_parameters.yml')
    assert value(path, 'eqasim.modeParametersPath') == str(context.directory / 'dmc_parameters.yml')
    assert value(path, 'eqasim:intersectionDelays.activateTl') == 'false'
    assert value(path, 'eqasim:intersectionDelays.activateUnsignalized') == 'true'
    assert value(path, 'eqasim:networkCalibration.calibrate') == 'penalty,freespeed,agent,subpopulations'
    assert value(path, 'eqasim:networkCalibration.minSpeed') == '1.0'
    if calibration == 'alpha':
        assert value(path, 'eqasim:alphaCalibration.carModeShare') == '0.4'
        assert value(path, 'eqasim:alphaCalibration.carPassengerModeShare') == '0.05'
        assert value(path, 'eqasim:alphaCalibration.level') == 'canton'
    elif calibration == 'beta':
        assert value(path, 'eqasim:calibration.runCalibration') == 'true'
        assert value(path, 'eqasim:calibration.maxEval') == '4000'
        assert value(path, 'eqasim:calibration.eqasimCachePath') == context.working_directory
    if not vdf:
        assert value(path, 'scoring.writeExperiencedPlans') == 'true'
    expected_arguments = ['--generateNetworkEvents', 'true'] if vdf else ['--preventwaitingtoentertraffic', 'y']
    assert simulation['arguments'] == expected_arguments
    run_context = Context(tmp_path / 'launch', stages={'matsim.simulation.prepare': simulation})
    snapshot = Path(path).read_bytes()
    calls = []
    def launch(ctx, main, arguments):
        calls.append((main, arguments))
        folder = ctx.directory / 'simulation_output'
        folder.mkdir()
        (folder / 'output_events.xml.gz').touch()
    monkeypatch.setattr(run.eqasim, 'run', launch)
    run.configure(run_context)
    run.execute(run_context)
    assert calls == [(simulation['main_class'], ['--config-path', path, *expected_arguments])]
    assert run_context.values.keys() == {'java_memory'}
    assert Path(path).read_bytes() == snapshot


def test_pt_calibration_requests_bootstrap_parameters(tmp_path):
    from calibration.pt_routing import calibrate
    context = Context(tmp_path / 'calibration')
    requests = []
    def stage(key, **kwargs):
        requests.append((key, kwargs))
    context.stage = stage
    calibrate.configure(context)
    assert ('matsim.config.pt', {'config': {'calibrate_pt_routing_params': False},
                                 'alias': 'pt_calibration_config'}) in requests


def test_beta_calibration_keeps_priority_over_alpha(tmp_path):
    context = Context(tmp_path, {'calibrate_alphas_in_matsim': True, 'calibrate_betas_in_matsim': True})
    params = tools.mode_shares_calibration_parameters(context)
    assert params['eqasim:calibration.runCalibration'] is True
    assert 'eqasim:alphaCalibration.activate' not in params
