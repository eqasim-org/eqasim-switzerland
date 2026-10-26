"""Unified network inputs: bounded detection and real downstream conversion."""

import gzip
from pathlib import Path

import geopandas as gpd
import osmium
import pytest
import synpp
from lxml import etree
from synpp.pipeline import ConfigurationContext
from shapely.geometry import box

from test_sumo_population import OSM, runtime
from sumo.runtime import sumo
from sumo.scenario.network import road_network
from sumo.scenario.network.tools import detect_network_format, resolve_input_path
from sumo.scenario.tools import elements


@pytest.mark.parametrize('root,expected', [('net', 'sumo'), ('osm', 'osm')])
@pytest.mark.parametrize('compressed', [False, True])
def test_xml_content_overrides_filename(tmp_path, root, expected, compressed):
    path = tmp_path / ('wrong.osm' if root == 'net' else 'wrong.net.xml')
    content = ('<?xml version="1.0" encoding="utf-8"?>\n<!--' + 'comment ' * 1000 + '-->\n<' + root + '/>').encode()
    path.write_bytes(gzip.compress(content) if compressed else content)
    assert detect_network_format(path) == expected


@pytest.mark.parametrize('contents', [b'', b'garbage', b'<routes/>', b'<!--' + b'x' * 70000 + b'--><osm/>'])
def test_unknown_or_excessive_header_rejected(tmp_path, contents):
    path = tmp_path / 'input.xml'
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        detect_network_format(path)


@pytest.mark.parametrize('kind', ['xml', 'gzip', 'pbf', 'renamed_pbf', 'misleading_xml', 'fallback'])
def test_osm_inputs_converted_before_downstream_config(tmp_path, runtime, monkeypatch, kind):
    xml = tmp_path / 'source.osm'
    xml.write_text(OSM)
    source = xml
    configured = str(xml)
    if kind in {'pbf', 'renamed_pbf'}:
        source = tmp_path / 'input.osm.pbf'
        with osmium.io.Reader(str(xml)) as reader, osmium.SimpleWriter(str(source)) as writer:
            osmium.apply(reader, writer)
        if kind == 'renamed_pbf':
            source = source.rename(tmp_path / 'input.blob')
        assert detect_network_format(source) == 'osm_pbf'
        configured = str(source)
    elif kind == 'gzip':
        source = tmp_path / 'input.gz'
        source.write_bytes(gzip.compress(xml.read_bytes()))
        configured = str(source)
    elif kind == 'misleading_xml':
        source = tmp_path / 'input.pbf'
        source.write_bytes(xml.read_bytes())
        configured = str(source)
    elif kind == 'fallback':
        source = tmp_path / 'osm' / 'input.osm'
        source.parent.mkdir()
        source.write_bytes(xml.read_bytes())
        configured = 'input.osm'
    trips = tmp_path / 'trips.xml'
    trips.write_text('<routes/>')
    original = source.read_bytes()
    monkeypatch.setattr(sumo, 'execute', lambda context: runtime)
    options = dict(definitions=[{'descriptor': 'sumo.config.microscopic'}],
        config={'sumo_network_file': configured, 'data_path': str(tmp_path), 'sumo_route_files': [str(trips)], 'random_seed': 42},
        working_directory=str(tmp_path / 'cache'), ensure_working_directory=True, rerun_required=False)
    result = synpp.run(**options)[0]
    network = Path(etree.parse(result).find('input/net-file').get('value'))
    assert network != source
    assert detect_network_format(network) == 'sumo'
    before = network.stat().st_mtime_ns
    assert synpp.run(**options)[0] == result
    assert network.stat().st_mtime_ns == before
    assert source.read_bytes() == original


def test_prepared_network_skips_runtime_and_tracks_source(tmp_path, monkeypatch):
    prepared = tmp_path / 'unusual.filename'
    prepared.write_text('<net/>')
    monkeypatch.setattr(sumo, 'execute', lambda context: pytest.fail('Prepared network installed SUMO'))
    context = ConfigurationContext({'sumo_network_file': str(prepared)})
    road_network.configure(context)
    assert not context.required_stages
    assert road_network.execute(context) == str(prepared)
    before = road_network.validate(context)
    prepared.write_text('<net>\n</net>')
    assert road_network.validate(context) != before
    result = synpp.run([{'descriptor': 'sumo.scenario.network.road_network'}],
        config={'sumo_network_file': str(prepared)}, working_directory=str(tmp_path / 'cache'),
        ensure_working_directory=True)
    assert result == [str(prepared)]


def test_direct_path_precedes_osm_directory(tmp_path, monkeypatch):
    (tmp_path / 'osm').mkdir()
    direct = tmp_path / 'input.xml'
    direct.write_text('<net/>')
    (tmp_path / 'osm' / 'input.xml').write_text('<osm/>')
    monkeypatch.chdir(tmp_path)
    assert resolve_input_path('input.xml', tmp_path) == direct
    direct.unlink()
    assert resolve_input_path('input.xml', tmp_path) == tmp_path / 'osm' / 'input.xml'


@pytest.mark.parametrize('model', ['microscopic', 'mesoscopic'])
def test_actual_synpp_cut_and_run_with_unified_osm_input(tmp_path, runtime, monkeypatch, model):
    source = tmp_path / 'source.osm'
    source.write_text(OSM)
    trips = tmp_path / 'trips.xml'
    trips.write_text('<routes><trip id="crossing" depart="0" fromLonLat="8.0001,47" toLonLat="8.0019,47"/></routes>')
    region = tmp_path / 'region.gpkg'
    gpd.GeoDataFrame(geometry=[box(8.0005, 46.999, 8.0015, 47.002)], crs=4326).to_file(region)
    monkeypatch.setattr(sumo, 'execute', lambda context: runtime)
    config = {'sumo_network_file': str(source), 'sumo_assignment_trip_files': [str(trips)],
              'sumo_region_geometry': str(region), 'sumo_modes': ['car'], 'sumo_simulation_model': model,
              'sumo_assignment_iterations': 2, 'sumo_loading_convergence_window': 2,
              'sumo_loading_convergence_tolerance': 1., 'sumo_end': 300, 'random_seed': 42,
              'sumo_micro_options': {'routing': {'device.rerouting.threads': 1}}}
    result = synpp.run([{'descriptor': 'sumo.run'}], config=config,
        working_directory=str(tmp_path / 'cache'), ensure_working_directory=True)[0]
    assert result['model'] == model
    assert result['count'] == 1
    assert all(float(trip.get('arrival')) >= 0 for trip in elements(result['tripinfo'], 'tripinfo'))
    recipe = etree.parse(result['config'])
    if model == 'microscopic':
        assert recipe.find('routing/device.rerouting.threads').get('value') == '1'
