"""Bounded route publication: restore metadata, validate, sort Arrow payloads and export."""

from pathlib import Path
import tempfile

from lxml import etree
import polars as pl

from sumo.scenario.assignment import source_trip_id
from sumo.scenario.network.tools import read_network, route_length, validate_routes
from sumo.scenario.tools import elements, iter_routes, write_serialized_trips

SCHEMA = {'id': pl.String, 'depart': pl.Float64, 'xml': pl.Binary}


def publish(context, trips, routes, network, method):
    definitions, types = {}, {}
    for element in elements(trips, ('trip','vType')):
        if element.tag=='vType':
            types[element.get('id')]=etree.fromstring(etree.tostring(element))
        else:
            definitions[element.get('id')] = (element.get('depart'),
                {p.get('key'):p.get('value') for p in element.findall('param')})
    data=read_network(network,context=context)
    state={};seen=set();count=0
    cache=Path(context.path()).resolve()
    with tempfile.TemporaryDirectory(prefix='publish_',dir=cache) as temporary:
        paths=[];rows=[]
        def flush():
            target=Path(temporary)/f'part_{len(paths):06d}.arrow'
            pl.DataFrame(rows,schema=SCHEMA).write_ipc(target)
            paths.append(target);rows.clear()
        for tag,vehicle in iter_routes(routes):
            if tag=='vType':
                types[vehicle.get('id')]=vehicle
                continue
            attrs=vehicle['attributes']
            parent=source_trip_id(attrs['id'],definitions);seen.add(parent)
            depart,metadata=definitions[parent]
            vehicle['parameters']={**metadata,**vehicle['parameters'],'sumo_original_trip_id':parent,
                                   'sumo_requested_depart':depart}
            vehicle['route_attributes']['cost']=format(vehicle['exit_times'][-1]-float(attrs['depart']),'.12g')
            distance=(float(attrs['routeLength']) if method=='dua' and 'routeLength' in attrs else
                      float(vehicle['route_attributes']['routeLength']) if method!='dua' and 'routeLength' in vehicle['route_attributes'] else
                      route_length(vehicle['edges'],data,attrs))
            vehicle['route_attributes']['routeLength']=format(distance,'.12g')
            validate_routes([vehicle],data,types,state=state)
            element=etree.Element('vehicle',attrib=attrs)
            etree.SubElement(element,'route',attrib={**vehicle['route_attributes'],'edges':' '.join(vehicle['edges']),
                'exitTimes':' '.join(format(t,'.12g') for t in vehicle['exit_times'])})
            for key,value in vehicle['parameters'].items(): etree.SubElement(element,'param',key=key,value=str(value))
            rows.append({'id':attrs['id'],'depart':float(attrs['depart']),'xml':etree.tostring(element)})
            count+=1
            if len(rows)>=10000:flush()
        if rows:flush()
        if seen!=set(definitions):
            raise ValueError('Loading did not complete every source trip; extend sumo_end or repair network deadlocks')
        target=cache/'published.arrow'
        if paths:
            frame=pl.concat([pl.scan_ipc(path) for path in paths]).sort('depart','id')
            frame.sink_ipc(target)
        else:pl.DataFrame(schema=SCHEMA).write_ipc(target)
    frame=pl.read_ipc(target)
    if frame['id'].n_unique()!=frame.height:raise ValueError('Duplicate routed vehicle IDs')
    output=cache/'loaded.routes.xml.gz'
    write_serialized_trips(output,types,frame['xml'])
    return output,count
