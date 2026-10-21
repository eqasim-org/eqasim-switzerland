"""Small native end-to-end loading and boundary-cutting regressions."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from lxml import etree
from shapely.geometry import box, LineString

from test_sumo_config import Context, runtime, network, trip_file, validate_schema
from sumo import run as selected_run
from sumo.runtime import sumo
from sumo.scenario import demand, modes
from sumo.scenario.tools import elements, read_routes
from sumo.scenario.network.tools import read_network, validate_routes
from sumo.simulation import load_network
from sumo.simulation import run_microscopic, run_mesoscopic
from sumo.cutter import geometry, split_network, network as cut_network, trips, scenario
from sumo.cutter.tools import inside_intervals, cut_vehicle


@pytest.mark.parametrize('method', ['dua', 'incremental', 'SUE'])
def test_crossing_trip_loaded_cut_and_simulated(tmp_path, runtime, network, method):
    region = tmp_path / 'region.gpkg'
    gpd.GeoDataFrame(geometry=[box(50, -20, 250, 20)], crs=2056).to_file(region)
    config = {'sumo_network_file': str(network), 'sumo_region_geometry': str(region),
              'sumo_network_crs': 'EPSG:2056', 'random_seed': 42, 'sumo_end': 1000,
              'sumo_assignment_iterations': 3, 'sumo_loading_convergence_window': 2,
              'sumo_loading_convergence_tolerance': 0.5, 'sumo_loading_method': method}
    stages = {'sumo.runtime.sumo': runtime}
    stages['sumo.cutter.geometry'] = geometry.execute(Context(geometry, tmp_path / 'geometry', config, stages))
    stages['sumo.cutter.split_network'] = split_network.execute(Context(split_network, tmp_path / 'split', config, stages))
    source = trip_file(tmp_path, count=6)
    stages['sumo.scenario.demand'] = demand.execute(Context(demand, tmp_path / 'demand',
        {'sumo_assignment_trip_files': [str(source)]}))
    stages['sumo.simulation.load_network'] = load_network.execute(Context(load_network, tmp_path / 'load', config, stages))
    loaded = stages['sumo.simulation.load_network']
    assert loaded['source_count'] == loaded['count'] == 6
    assert loaded['boundary_split']
    if method == 'dua':
        assert loaded['convergence']['iterations'] >= 2
    stages['sumo.cutter.network'] = cut_network.execute(Context(cut_network, tmp_path / 'cutnet', config, stages))
    stages['sumo.cutter.trips'] = trips.execute(Context(trips, tmp_path / 'cuttrips', config, stages))
    result = stages['sumo.cutter.trips']
    assert result['count'] == 6
    assert result['interpolated_visits'] == 0
    net = read_network(result['network'])
    assert sum(edge['length'] for edge in net['edges'].values()) == pytest.approx(200, abs=.01)
    _, full_vehicles = read_routes(loaded['routes'])
    _, cut_vehicles = read_routes(result['routes'])
    full = {v['attributes']['id']: v for v in full_vehicles}
    for cut in cut_vehicles:
        parent = full[cut['attributes']['id']]
        assert float(cut['attributes']['depart']) == pytest.approx(parent['exit_times'][0])
        assert cut['parameters']['sumo_original_trip_id'] == cut['attributes']['id']
    bundle = scenario.execute(Context(scenario, tmp_path / 'bundle', config, stages))
    sumo.run(Context(sumo, tmp_path / 'run', stages=stages), ['-c', bundle['config'], '--no-step-log', 'true'])
    assert Path(bundle['config']).is_file()
    if method == 'dua':
        stages['sumo.cutter.scenario'] = bundle
        before = {path.name: path.read_bytes() for path in Path(bundle['config']).parent.iterdir()}
        for model, module in [('microscopic', run_microscopic), ('mesoscopic', run_mesoscopic)]:
            execution_config = {**config, 'sumo_micro_step_length': 0.25,
                                'sumo_micro_action_step_length': 0.5, 'sumo_meso_segment_length': 50}
            execution = Context(module, tmp_path / model, execution_config, stages)
            result = module.execute(execution)
            assert result['model'] == model
            assert result['network'] == bundle['network']
            assert result['input_routes'] == bundle['routes']
            validate_schema(result['config'], runtime, 'sumo')
            recipe = etree.parse(result['config'])
            if model == 'microscopic':
                assert recipe.find('mesoscopic/mesosim') is None
                assert recipe.find('time/step-length').get('value') == '0.25'
            else:
                assert recipe.find('mesoscopic/mesosim').get('value') == 'true'
                assert recipe.find('mesoscopic/meso-edgelength').get('value') == '50.0'
            completed = list(elements(result['tripinfo'], 'tripinfo'))
            # Streaming elements are cleared after consumption; count separately below.
            assert len(completed) == 6
            assert all(float(trip.get('arrival')) >= 0 for trip in elements(result['tripinfo'], 'tripinfo'))
            for output in ('config', 'tripinfo', 'edge_data', 'summary', 'routes'):
                assert Path(result[output]).parent == tmp_path / model
            selector = Context(selected_run, tmp_path / 'selector', {'sumo_simulation_model': model},
                               {'sumo.simulation.run_' + model: result})
            assert selected_run.execute(selector) is result
        assert before == {path.name: path.read_bytes() for path in Path(bundle['config']).parent.iterdir()}


def test_modes_and_holes():
    assert modes.selected(['truck']) == {'truck', 'lcv'}
    assert modes.includes('lcv', modes.selected(['car', 'truck']))
    assert not modes.includes('car_passenger', modes.selected(['car', 'truck']))
    with pytest.raises(ValueError):
        modes.selected(['typo'])
    polygon = box(0, -10, 100, 10).difference(box(40, -5, 60, 5))
    assert inside_intervals(LineString([(-10, 0), (110, 0)]), polygon) == pytest.approx([(1/12, 5/12), (7/12, 11/12)])


def test_regional_dispatch_dependencies_and_invalid_model():
    from synpp.pipeline import ConfigurationContext
    for model in ('microscopic', 'mesoscopic'):
        context = ConfigurationContext({'sumo_simulation_model': model})
        selected_run.configure(context)
        assert [stage['descriptor'] for stage in context.required_stages] == ['sumo.simulation.run_' + model]
    context = ConfigurationContext({})
    selected_run.configure(context)
    assert context.required_config['sumo_simulation_model'] == 'microscopic'
    with pytest.raises(ValueError, match='sumo_simulation_model'):
        selected_run.configure(ConfigurationContext({'sumo_simulation_model': 'macroscopic'}))
    for module in (run_microscopic, run_mesoscopic):
        context = ConfigurationContext({'random_seed': 42})
        module.configure(context)
        assert {stage['descriptor'] for stage in context.required_stages} == {'sumo.cutter.scenario', 'sumo.runtime.sumo'}


@pytest.mark.parametrize('method,scale,expected', [('dua', 2, 6), ('incremental', 2, 6), ('SUE', 1, 3)])
def test_loading_methods_and_scale(tmp_path, runtime, network, method, scale, expected):
    source = trip_file(tmp_path, count=3)
    selected = demand.execute(Context(demand, tmp_path / 'demand', {'sumo_assignment_trip_files': [str(source)]}))
    result = load_network.execute(Context(load_network, tmp_path / 'load', {
        'sumo_network_file': str(network), 'sumo_loading_method': method,
        'sumo_assignment_scale': scale, 'random_seed': 42, 'sumo_end': 1000,
        'sumo_assignment_iterations': 3, 'sumo_loading_convergence_window': 2},
        {'sumo.runtime.sumo': runtime, 'sumo.scenario.demand': selected}))
    assert result['source_count'] == 3
    assert result['count'] == expected
    _, vehicles = read_routes(result['routes'])
    assert {v['parameters']['sumo_original_trip_id'] for v in vehicles} == {'t0', 't1', 't2'}


def test_mode_filter_vehicle_classes_and_conditional_dependencies(tmp_path, runtime, network):
    from synpp.pipeline import ConfigurationContext
    from sumo.scenario.population import trips as commercial
    source = tmp_path / 'modes.xml'
    source.write_text('<routes>' + ''.join(f'<vType id="{c}" vClass="{c}"/>' for c in ('passenger', 'truck', 'delivery', 'bicycle'))
        + ''.join(f'<trip id="{c}" type="{c}" depart="0" from="in" to="out"/>' for c in ('passenger', 'truck', 'delivery', 'bicycle')) + '</routes>')
    result = demand.execute(Context(demand, tmp_path / 'demand', {'sumo_assignment_trip_files': [str(source)]}))
    assert result['count'] == result['loading_count'] == 3
    loaded = load_network.execute(Context(load_network, tmp_path / 'load', {
        'sumo_network_file': str(network), 'random_seed': 42, 'sumo_end': 1000,
        'sumo_assignment_iterations': 2, 'sumo_loading_convergence_window': 2},
        {'sumo.runtime.sumo': runtime, 'sumo.scenario.demand': result}))
    assert loaded['count'] == 3
    assert {vehicle['attributes']['type'] for vehicle in read_routes(loaded['routes'])[1]} == {'passenger', 'truck', 'delivery'}
    context = ConfigurationContext({'sumo_modes': ['car']})
    commercial.configure(context)
    assert not context.required_stages
    context = ConfigurationContext({'sumo_modes': ['lcv']})
    commercial.configure(context)
    assert [stage['descriptor'] for stage in context.required_stages] == ['synthesis.lcv.trips']


def test_reentry_keeps_separate_visits_and_positions():
    source = {'edges': {e: {'length': 100} for e in 'abcde'}}
    clipped = {'connections': [{'from': 'a', 'to': 'b'}, {'from': 'd', 'to': 'e'}]}
    mapping = {e: [{'id': e, 'start': 0., 'end': 1., 'length': 100.}] for e in 'abde'}
    vehicle = {'attributes': {'id': 'p', 'depart': '10', 'departPos': '20', 'arrivalPos': '80'},
               'edges': list('abcde'), 'exit_times': [18., 28., 38., 48., 56.], 'parameters': {'person_id': 'p'}, 'route_attributes': {}}
    visits = cut_vehicle(vehicle, mapping, source, clipped)
    assert [v['edges'] for v in visits] == [list('ab'), list('de')]
    assert [float(v['attributes']['depart']) for v in visits] == [10., 38.]
    assert visits[0]['attributes']['departPos'] == '20'
    assert visits[1]['attributes']['arrivalPos'] == '80'
    assert visits[1]['parameters']['person_id'] == 'p'
    assert 'departSpeed' not in visits[0]['attributes']
    assert visits[1]['attributes']['departLane'] == 'best'
    assert visits[1]['attributes']['departSpeed'] == 'max'
    assert [v['route_attributes']['routeLength'] for v in visits] == ['180', '180']


@pytest.mark.parametrize('policy', ['largest', 'all', 'error'])
def test_disconnected_components(tmp_path, runtime, network, policy):
    from sumo.cutter.tools import build_network
    polygon = box(0, -10, 40, 10).union(box(260, -10, 300, 10))
    context = Context(cut_network, tmp_path / 'cut', {'sumo_cutter_components': policy}, {'sumo.runtime.sumo': runtime})
    if policy == 'error':
        with pytest.raises(ValueError, match='disconnected'):
            build_network(context, network, polygon, keep_outside=False, components=policy)
    else:
        result = build_network(context, network, polygon, keep_outside=False, components=policy)
        assert result['components_before_filter'] == 2
        assert result['edge_count'] == (1 if policy == 'largest' else 2)
        assert result['discarded_edges'] == (1 if policy == 'largest' else 0)


def test_geometry_crs_offset_and_holes(tmp_path):
    from sumo.cutter.tools import load_region
    from pyproj import Transformer
    source = tmp_path / 'region.geojson'
    poly = box(8, 47, 8.002, 47.002).difference(box(8.0005, 47.0005, 8.0015, 47.0015))
    gpd.GeoDataFrame(geometry=[poly], crs=4326).to_file(source)
    transformed = load_region(source, None, {'projParameter': 'EPSG:2056', 'netOffset': '-2600000,-1200000'})
    assert len(transformed.interiors) == 1
    x, y = Transformer.from_crs(4326, 2056, always_xy=True).transform(8, 47)
    assert transformed.bounds[:2] == pytest.approx((x-2600000, y-1200000), abs=2)


def test_native_turns_signals_lanes_and_curves_survive(tmp_path, runtime):
    from sumo.cutter.tools import build_network
    nodes = tmp_path / 'nodes.xml'
    nodes.write_text('<nodes><node id="a" x="-100" y="0"/><node id="b" x="0" y="0" type="traffic_light"/>'
        '<node id="c" x="100" y="0"/><node id="d" x="0" y="100"/><node id="e" x="200" y="0"/></nodes>')
    edges = tmp_path / 'edges.xml'
    edges.write_text('<edges><edge id="in" from="a" to="b" numLanes="2" shape="-100,0 -50,5 0,0">'
        '<lane index="0" allow="truck delivery"/><lane index="1" allow="passenger"/></edge>'
        '<edge id="through" from="b" to="c" numLanes="2"><lane index="0" allow="truck delivery"/>'
        '<lane index="1" allow="passenger"/></edge><edge id="side" from="b" to="d"/>'
        '<edge id="out" from="c" to="e" numLanes="2"/></edges>')
    connections = tmp_path / 'connections.xml'
    connections.write_text('<connections><connection from="in" to="through" fromLane="0" toLane="0"/>'
        '<connection from="in" to="through" fromLane="1" toLane="1"/>'
        '<connection from="through" to="out" fromLane="0" toLane="0"/>'
        '<connection from="through" to="out" fromLane="1" toLane="1"/></connections>')
    source = tmp_path / 'source.net.xml.gz'
    context = Context(cut_network, tmp_path / 'native', stages={'sumo.runtime.sumo': runtime})
    sumo.run(context, ['--node-files', nodes, '--edge-files', edges, '--connection-files', connections,
        '--offset.disable-normalization', 'true', '--no-turnarounds', 'true', '-o', source], binary='netconvert')
    import gzip
    with gzip.open(source, 'rb') as stream:
        original = etree.parse(stream)
    result = build_network(context, source, box(-75, -20, 150, 70), keep_outside=True)
    with gzip.open(result['network'], 'rb') as stream:
        rebuilt = etree.parse(stream)
    assert etree.tostring(original.find('tlLogic')) == etree.tostring(rebuilt.find('tlLogic'))
    assert rebuilt.find("edge[@id='in']/lane[@index='0']").get('allow') == 'delivery truck'
    assert rebuilt.find("edge[@id='in']/lane[@index='1']").get('allow') == 'passenger'
    mapping = json.loads(Path(result['mapping']).read_text())
    reverse = {piece['id']: source_id for source_id, pieces in mapping.items() for piece in pieces}
    rebuilt_data = read_network(result['network'])
    assert not any(reverse[c['from']] == 'in' and reverse[c['to']] == 'side' for c in rebuilt_data['connections'])
    assert any(len(rebuilt_data['edges'][piece['id']]['shape'].coords) > 2 for piece in mapping['in'])


def test_nonconvergence_and_incomplete_horizon_are_explicit(tmp_path, runtime, network):
    source = trip_file(tmp_path, count=1)
    selected = demand.execute(Context(demand, tmp_path / 'demand', {'sumo_assignment_trip_files': [str(source)]}))
    settings = {'sumo_network_file': str(network), 'random_seed': 42, 'sumo_end': 1000,
                'sumo_assignment_iterations': 1, 'sumo_loading_require_convergence': True,
                'sumo_loading_method': 'dua'}
    stages = {'sumo.runtime.sumo': runtime, 'sumo.scenario.demand': selected}
    with pytest.raises(RuntimeError, match='stability'):
        load_network.execute(Context(load_network, tmp_path / 'unstable', settings, stages))
    settings.update(sumo_loading_require_convergence=False, sumo_end=1)
    with pytest.raises(ValueError, match='unfinished|undeparted'):
        load_network.execute(Context(load_network, tmp_path / 'incomplete', settings, stages))


def test_projected_native_network_retains_coordinate_frame(tmp_path, runtime):
    from test_sumo_population import OSM
    from sumo.scenario.network import road_network
    from sumo.cutter.tools import build_network
    folder = tmp_path / 'osm'
    folder.mkdir()
    source = folder / 'input.osm'
    source.write_text(OSM)
    config = {'data_path': str(tmp_path), 'sumo_osm_file': str(source), 'random_seed': 42}
    stages = {'sumo.runtime.sumo': runtime}
    source_network = road_network.execute(Context(road_network, tmp_path / 'net', config, stages))
    before = read_network(source_network)
    bounds = next(iter(before['edges'].values()))['shape'].bounds
    left, low, right, high = bounds
    region = box((left+right)/2, low-1000, right+1000, high+1000)
    result = build_network(Context(cut_network, tmp_path / 'split', stages=stages), source_network, region, keep_outside=True)
    after = read_network(result['network'])
    assert [float(value) for value in after['location']['netOffset'].split(',')] == pytest.approx(
        [float(value) for value in before['location']['netOffset'].split(',')], abs=1e-8)
    assert after['location']['projParameter'] == before['location']['projParameter']


def test_repeated_directed_geometry_crossings():
    line = LineString([(0, 0), (100, 0), (0, 0)])
    assert inside_intervals(line, box(20, -10, 40, 10)) == pytest.approx([(0.1, 0.2), (0.8, 0.9)])


def test_dua_updates_route_choices_using_loaded_costs(tmp_path, runtime):
    nodes = tmp_path / 'nodes.xml'
    nodes.write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="100" y="0"/>'
        '<node id="c" x="200" y="0"/><node id="f" x="190" y="0"/>'
        '<node id="d" x="300" y="0"/><node id="e" x="150" y="50"/></nodes>')
    edges = tmp_path / 'edges.xml'
    edges.write_text('<edges><edge id="in" from="a" to="b" length="100" speed="10" numLanes="3"/>'
        '<edge id="fast" from="b" to="f" length="500" speed="10" numLanes="3"/>'
        '<edge id="neck" from="f" to="c" length="100" speed="10"/>'
        '<edge id="slow1" from="b" to="e" length="300" speed="8"/>'
        '<edge id="slow2" from="e" to="c" length="300" speed="8"/>'
        '<edge id="out" from="c" to="d" length="100" speed="10" numLanes="3"/></edges>')
    stages = {'sumo.runtime.sumo': runtime}
    net = tmp_path / 'network.net.xml'
    sumo.run(Context(sumo, tmp_path / 'native', stages=stages), ['--node-files', nodes, '--edge-files', edges,
        '--output-file', net, '--no-internal-links', 'true'], binary='netconvert')
    source = trip_file(tmp_path, count=150)
    source.write_text(source.read_text().replace(' depart=', ' departLane="free" depart='))
    stages['sumo.scenario.demand'] = demand.execute(Context(demand, tmp_path / 'demand',
        {'sumo_assignment_trip_files': [str(source)]}))
    loaded = load_network.execute(Context(load_network, tmp_path / 'load', {
        'sumo_network_file': str(net), 'random_seed': 42, 'sumo_end': 2000,
        'sumo_assignment_interval': 300, 'sumo_assignment_iterations': 5,
        'sumo_loading_convergence_tolerance': 0, 'sumo_loading_method': 'dua'}, stages))
    routes = read_routes(loaded['routes'])[1]
    assert any('fast' in vehicle['edges'] for vehicle in routes)
    assert any('slow1' in vehicle['edges'] for vehicle in routes)
    assert loaded['convergence']['mean_travel_times_s'][0] > 30


@pytest.mark.parametrize('restricted', [False, True])
def test_default_macro_keeps_assignment_distribution_without_postrouting(tmp_path, runtime, monkeypatch, restricted):
    nodes = tmp_path / 'nodes.xml'
    nodes.write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="100" y="0"/>'
        '<node id="c" x="200" y="0"/><node id="d" x="300" y="0"/>'
        '<node id="e" x="150" y="50"/></nodes>')
    edge_file = tmp_path / 'edges.xml'
    permission = ' allow="passenger"' if restricted else ''
    edge_file.write_text('<edges><edge id="in" from="a" to="b" length="100" speed="10"/>'
        f'<edge id="fast" from="b" to="c" length="100" speed="10"{permission}/>'
        '<edge id="slow1" from="b" to="e" length="70" speed="8"/>'
        '<edge id="slow2" from="e" to="c" length="70" speed="8"/>'
        '<edge id="out" from="c" to="d" length="100" speed="10"/></edges>')
    stages = {'sumo.runtime.sumo': runtime}
    network = tmp_path / 'network.net.xml'
    sumo.run(Context(sumo, tmp_path / 'native', stages=stages), ['--node-files', nodes, '--edge-files', edge_file,
        '--output-file', network, '--no-internal-links', 'true'], binary='netconvert')
    source = trip_file(tmp_path, count=200)
    if restricted:
        source.write_text(source.read_text().replace('<routes>', '<routes><vType id="truck" vClass="truck"/>')
                          .replace(' depart=', ' type="truck" depart='))
    stages['sumo.scenario.demand'] = demand.execute(Context(demand, tmp_path / 'demand',
        {'sumo_assignment_trip_files': [str(source)]}))
    native_run = sumo.run
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs.get('binary', 'sumo'))
        return native_run(*args, **kwargs)
    monkeypatch.setattr(sumo, 'run', record)
    settings = {'sumo_network_file': str(network), 'random_seed': 42,
                'sumo_end': 3600, 'sumo_assignment_scale': 10}
    context = Context(load_network, tmp_path / 'load', settings, stages)
    assert context.config('sumo_loading_method') == 'incremental'
    if restricted:
        with pytest.raises(ValueError, match='Invalid native marouter assignment.*forbidden to truck'):
            load_network.execute(context)
        assert calls == ['duarouter', 'marouter']
        return
    result = load_network.execute(context)
    assert calls == ['duarouter', 'marouter']
    assert result['source_count'] == 200
    assert result['count'] == 2000
    _, vehicles = read_routes(result['routes'])
    assert {vehicle['parameters']['sumo_original_trip_id'] for vehicle in vehicles} == {f't{i}' for i in range(200)}
    assert all(vehicle['parameters']['source_travel_time_s'] == '32' for vehicle in vehicles)
    assigned_paths = {tuple(route.get('edges').split()) for vehicle in elements(context.cache / 'assigned.flows.xml.gz', 'vehicle')
                      for route in vehicle.findall('routeDistribution/route') if float(route.get('probability')) > 0}
    assert {tuple(vehicle['edges']) for vehicle in vehicles} == assigned_paths
    weighted = {edge.get('id'): (float(edge.get('entered')), float(edge.get('traveltime')))
                for interval in elements(result['weights'], 'interval') for edge in interval.findall('edge')}
    for edge in ('fast', 'slow1'):
        count = sum(edge in vehicle['edges'] for vehicle in vehicles)
        assert count > 0
        assert abs(count - weighted[edge][0]) < 1
    for vehicle in vehicles:
        duration = vehicle['exit_times'][-1] - float(vehicle['attributes']['depart'])
        assert duration == pytest.approx(sum(weighted[edge][1] for edge in vehicle['edges']))
    first = [(v['attributes']['id'], v['edges'], v['exit_times']) for v in vehicles]
    second = load_network.execute(Context(load_network, tmp_path / 'repeat', settings, stages))
    assert first == [(v['attributes']['id'], v['edges'], v['exit_times']) for v in read_routes(second['routes'])[1]]


def test_macro_clock_uses_departure_slice_and_partial_endpoints(tmp_path):
    from sumo.scenario.assignment import materialize_assignment
    from lxml import etree
    # All edges appear in an explicitly timed route; never fall back to free flow.
    network = tmp_path / 'network.xml'
    network.write_text('<net>' + ''.join(f'<edge id="{e}" from="a" to="b"><lane id="{e}_0" index="0" '
        'speed="10" length="100" shape="0,0 100,0"/></edge>' for e in ('a', 'b'))
        + '<connection from="a" to="b" fromLane="0" toLane="0"/></net>')
    source = tmp_path / 'trips.xml'
    source.write_text('<routes><trip id="p" depart="3595" departPos="50" arrivalPos="25" from="a" to="b"/></routes>')
    assigned = tmp_path / 'assigned.xml'
    assigned.write_text('<routes><vehicle id="p" depart="3595"><routeDistribution>'
        '<route edges="a b" probability="7"/></routeDistribution></vehicle></routes>')
    weights = tmp_path / 'weights.xml'
    weights.write_text('<meandata><interval begin="0" end="3600"><edge id="a" traveltime="20"/>'
        '<edge id="b" traveltime="40"/></interval><interval begin="3600" end="7200">'
        '<edge id="a" traveltime="200"/><edge id="b" traveltime="400"/></interval></meandata>')
    output = tmp_path / 'routes.xml'
    materialize_assignment(assigned, weights, source, network, output, seed=42)
    vehicle = read_routes(output)[1][0]
    assert vehicle['exit_times'] == [3605, 3615]
    tree = etree.parse(str(weights))
    tree.find('interval/edge').getparent().remove(tree.find('interval/edge'))
    tree.write(str(weights))
    with pytest.raises(ValueError, match='No loaded travel time'):
        materialize_assignment(assigned, weights, source, network, output, seed=42)
