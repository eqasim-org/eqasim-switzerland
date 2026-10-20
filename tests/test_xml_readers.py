import gzip
from pathlib import Path
import xml.etree.ElementTree as stdlib_xml

import pytest
from matsim import readers


def write_xml(path, content):
    payload=content.encode('utf-8')
    if path.suffix=='.gz':
        with gzip.open(path,'wb') as stream:
            stream.write(payload)
    else:
        path.write_bytes(payload)


@pytest.mark.parametrize('compressed',[False,True])
@pytest.mark.parametrize('namespace',['',' xmlns="http://www.matsim.org/files/dtd"'])
@pytest.mark.parametrize('backend',['lxml','stdlib'])
def test_households_read_binary_xml_and_close_files(tmp_path,monkeypatch,compressed,namespace,backend):
    if backend=='stdlib':
        monkeypatch.setattr(readers,'ET',stdlib_xml)
    path=tmp_path/('households.xml.gz' if compressed else 'households.xml')
    write_xml(path,f'''<?xml version="1.0" encoding="UTF-8"?>
<households{namespace}><household id="5"><members><personId refId="7"/></members>
<attributes><attribute name="incomeClass" class="java.lang.Integer">2</attribute>
<attribute name="censusId" class="java.lang.Integer">15</attribute>
<attribute name="household_income" class="java.lang.Double">6000.5</attribute>
<attribute name="location" class="java.lang.String">Genève</attribute></attributes>
</household></households>''')
    opened=[]
    original=readers.xopen.xopen
    def track(filename,mode):
        assert mode=='rb'
        stream=original(filename,mode)
        opened.append(stream)
        return stream
    monkeypatch.setattr(readers.xopen,'xopen',track)
    result=readers.houshold_reader(str(path)).households
    assert len(result)==1
    row=result.iloc[0]
    assert row['id']==5
    assert row['members']==[7]
    assert row['incomeClass']=='2'
    assert row['household_income']==6000.5
    assert row['location']=='Genève'
    assert all(stream.closed for stream in opened)


@pytest.mark.parametrize('compressed',[False,True])
def test_both_plan_readers_accept_binary_xml(tmp_path,compressed):
    path=tmp_path/('plans.xml.gz' if compressed else 'plans.xml')
    write_xml(path,'''<?xml version="1.0" encoding="UTF-8"?>
<population><person id="7"><plan selected="yes">
<activity type="home" x="1" y="2"/><leg mode="walk"><route type="generic">walk</route></leg>
<activity type="work" x="3" y="4"/></plan></person></population>''')
    iterator=readers.plan_reader(str(path))
    person,plan=next(iterator)
    assert person.get('id')=='7'
    assert plan.get('selected')=='yes'
    iterator.close()
    frames=readers.plan_reader_dataframe(str(path))
    assert len(frames.persons)==1
    assert len(frames.plans)==1
    assert frames.activities['type'].tolist()==['home','work']
    assert frames.legs['mode'].tolist()==['walk']


@pytest.mark.parametrize('prefix', ['switzerland_', 'geneva_'])
@pytest.mark.parametrize('output_has_income', [False, True])
def test_analysis_falls_back_to_parent_households(tmp_path, prefix, output_has_income):
    from types import SimpleNamespace
    from analysis.mode_shares.utils import ModeShareAnalyzer
    scenario = tmp_path / 'scenario'
    results = scenario / 'simulation_output'
    results.mkdir(parents=True)
    (results / 'output_persons.csv.gz').write_bytes(gzip.compress(
        b'person;subpopulation;age;sex;cantonId;cantonName\n7;resident;40;m;25;Geneva\n'))
    (results / 'output_trips.csv.gz').write_bytes(gzip.compress(
        b'trip_id;person;main_mode;euclidean_distance;end_activity_type\n1;7;car;1000;work\n'))
    xml = '<households><household id="5"><members><personId refId="7"/></members>{}</household></households>'
    attributes = '<attributes><attribute name="incomeClass">2</attribute></attributes>'
    write_xml(results / 'output_households.xml.gz', xml.format(attributes if output_has_income else ''))
    # An absent parent file must not matter when the output already contains income.
    if not output_has_income:
        write_xml(scenario / f'{prefix}households.xml.gz', xml.format(attributes))
    values = {'analysis.counts.simulation_path': '', 'output_path': str(tmp_path),
              'output_id': 'scenario', 'output_prefix': prefix, 'simulation_directory': 'simulation_output'}
    context = SimpleNamespace(config=lambda key: values[key])
    analyzer = ModeShareAnalyzer.__new__(ModeShareAnalyzer)
    analyzer.load_matsim_data(context)
    assert analyzer.trips['income_class'].tolist() == [2]
    assert analyzer.trips['mode'].tolist() == ['car']
