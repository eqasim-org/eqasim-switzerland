"""PT corrections read route agency data from the cleaned GTFS output."""

from pathlib import Path

import pytest
from matsim.scenario.network.utils import pt_route_check


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
