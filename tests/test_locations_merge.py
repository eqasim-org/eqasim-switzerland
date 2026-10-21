import csv
import gzip
from types import SimpleNamespace

import pandas as pd
from shapely import from_wkt

from matsim.scenario.network.utils import pt_route_check
from matsim.scenario.network.utils.network_cleaner import networkCleaner


def test_pt_correction_exports_new_and_reversed_detailed_geometry(tmp_path, monkeypatch):
    network = tmp_path / 'network.xml'
    network.write_text('''<network><nodes>
<node id="1" x="0" y="0"/>
<node id="2" x="10" y="0"/>
<node id="3" x="20" y="0"/>
</nodes><links>
<link id="road" from="1" to="2" length="12" freespeed="10" capacity="600" permlanes="1" modes="car"/>
</links></network>''')
    schedule = tmp_path / 'schedule.xml'
    schedule.write_text('<transitSchedule><transitStops/></transitSchedule>')
    geometry = tmp_path / 'geometry.csv'
    with geometry.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['LinkId', 'Geometry'])
        writer.writerow(['road', 'LINESTRING (0 0, 5 3, 10 0)'])
    monkeypatch.setattr(pt_route_check, 'Corridors', lambda *args: SimpleNamespace(bounds=(-100,-100,100,100)))
    def correct(schedule, net, *args, **kwargs):
        reverse = net.add_reverse_link('road', {})
        added = net.add_bus_links('2', '3', [(10,0),(15,5),(20,0)], {})
        return [], {'road'}, {reverse, *added}
    monkeypatch.setattr(pt_route_check, 'check_and_correct', correct)
    patched = tmp_path / 'patched.xml.gz'
    output_geometry = tmp_path / 'patched.csv'
    _, changes = pt_route_check.run(
        str(schedule), str(network), 'unused', str(tmp_path/'report.csv'),
        corrected_schedule_path=str(tmp_path/'corrected.xml.gz'), patched_network_path=str(patched),
        input_geometry_path=str(geometry), output_geometry_path=str(output_geometry), log=lambda *args: None)
    assert changes >= 3
    with output_geometry.open() as stream:
        shapes = {row['LinkId']: from_wkt(row['Geometry']) for row in csv.DictReader(stream)}
    assert list(shapes['road'].coords) == [(0,0),(5,3),(10,0)]
    assert list(shapes['pt_bus_link_1'].coords) == [(10,0),(5,3),(0,0)]
    from lxml import etree
    with gzip.open(patched) as stream:
        tree=etree.parse(stream)
    assert set(shapes) == {link.get('id') for link in tree.findall('links/link')}
    assert 'bus' in tree.find('links/link[@id="road"]').get('modes').split(',')


def test_pt_stop_node_breaks_chain_without_changing_geometry_or_tolls():
    links = pd.DataFrame([
        dict(link_id='a', from_node='1', to_node='2', length=10., capacity=600., freespeed=10.,
             permlanes=1., modes='car,truck', attributes={'toll':1.}),
        dict(link_id='b', from_node='2', to_node='3', length=10., capacity=600., freespeed=10.,
             permlanes=1., modes='car,truck', attributes={'toll':2.}),
    ])
    cleaner = networkCleaner(SimpleNamespace(links=links, nodes=pd.DataFrame()))
    merged, _ = cleaner.merge_link_chains(links)
    protected, _ = cleaner.merge_link_chains(links, protected_nodes={'2'})
    assert len(merged) == 1
    assert merged.iloc[0]['attributes']['toll'] == 3.
    assert protected.link_id.tolist() == ['a', 'b']


import pytest
from pathlib import Path


@pytest.mark.parametrize('agencies', [['881'], []])
def test_corrected_stage_reads_cleaned_gtfs_output_directory(tmp_path, monkeypatch, agencies):
    from matsim.scenario.network import corrected
    cleaned = tmp_path / 'cleaned'
    (cleaned / 'output').mkdir(parents=True)
    (cleaned / 'output/routes.txt').write_text('route_id,agency_id\nroute1,881\n')
    directory = tmp_path / 'corrected'
    directory.mkdir()
    mapped = {key: f'mapped_{key}' for key in ('network','schedule','vehicles','road_network','geometry')}
    class Context:
        def __init__(self):
            self.values={'pt_route_agencies': agencies, 'data_path': str(tmp_path)}
            self.dependencies=set()
        def config(self,key,default=None):
            return self.values.setdefault(key,default)
        def stage(self,name,**kwargs):
            self.dependencies.add(name)
            return mapped if name=='matsim.scenario.network.mapped' else 'gtfs'
        def path(self,stage=None):
            return str(cleaned if stage=='data.gtfs.cleaned' else directory)
    def check(**kwargs):
        if agencies:
            assert Path(kwargs['gtfs_routes_path'])==cleaned/'output/routes.txt'
            assert pt_route_check.read_line_agencies(kwargs['gtfs_routes_path'])=={'route1':'881'}
        else:
            assert kwargs['gtfs_routes_path'] is None
        Path(kwargs['corrected_schedule_path']).touch()
        return [],0
    monkeypatch.setattr(corrected.pt_route_check,'run',check)
    context=Context()
    corrected.configure(context)
    result=corrected.execute(context)
    assert result['network']==mapped['network']
    assert result['geometry']==mapped['geometry']
    assert ('data.gtfs.cleaned' in context.dependencies)==bool(agencies)
