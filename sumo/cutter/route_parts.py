"""Stream national routes into bounded, independently processed regional Arrow parts."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

from lxml import etree
import polars as pl

from sumo.cutter.tools import cut_vehicle
from sumo.scenario.network.tools import read_network
from sumo.scenario.tools import iter_routes

SCHEMA = {'id':pl.String,'depart':pl.Float64,'xml':pl.Binary}


def configure(context):
    context.stage('sumo.simulation.load_network')
    context.stage('sumo.cutter.network')
    context.config('threads', volatile=True)
    context.config('sumo_native_routes', True)
    if context.config('sumo_native_routes'):
        context.stage('sumo.native.routes')


def execute(context):

    # Read reusable network caches and immutable mapping once for all worker threads.

    loaded=context.stage('sumo.simulation.load_network')
    region=context.stage('sumo.cutter.network')
    source=read_network(loaded['network'],context=context)
    cut=read_network(region['network'],context=context)
    mapping={key:value for key,value in json.loads(Path(region['mapping']).read_text()).items() if value}
    transitions={(c['from'],c['to']) for c in cut['connections']}
    kernel=None
    if context.config('sumo_native_routes'):
        from sumo.native.kernels import RouteKernel
        kernel=RouteKernel(context.stage('sumo.native.routes'),mapping,source,transitions)
    cache=Path(context.path()).resolve()
    workers=max(1,int(context.config('threads')))
    types={}; pending=deque(); parts=[]
    report={'source_count':0,'count':0,'outside_or_discarded':0,'repeated_visits':0,'interpolated_visits':0}

    # Keep only a bounded number of batches in flight, with deterministic collection order.

    def collect(future):
        rows,counts=future.result()
        path=cache/f'part_{len(parts):06d}.arrow'
        pl.DataFrame(rows,schema=SCHEMA).write_ipc(path)
        parts.append(str(path))
        for key,value in counts.items(): report[key]+=value
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batch=[]
        for tag,vehicle in iter_routes(loaded['routes'],selected_edges=set(mapping)):
            if tag=='vType':
                types[vehicle.get('id')]=vehicle
                continue
            report['source_count']+=1
            if tag=='outside':
                report['outside_or_discarded']+=1
                continue
            batch.append(vehicle)
            if len(batch)>=1024:
                pending.append(pool.submit(_process,batch,mapping,source,cut,transitions,kernel))
                batch=[]
                if len(pending)>=2*workers: collect(pending.popleft())
        if batch: pending.append(pool.submit(_process,batch,mapping,source,cut,transitions,kernel))
        while pending: collect(pending.popleft())

    # Publish cached parts; the export stage validates the complete regional result.
    definitions=cache/'types.xml'
    definitions.write_bytes(b''.join(etree.tostring(value) for value in types.values()))
    result={'parts':parts,'types':str(definitions),'network':region['network'],**report}
    (cache/'parts.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate(context):
    from sumo.cutter import tools
    from sumo.scenario import tools as scenario_tools
    from sumo.native import kernels
    return [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (tools,scenario_tools,kernels)]


def _process(batch,mapping,source,cut,transitions,kernel):
    rows=[];counts={'count':0,'outside_or_discarded':0,'repeated_visits':0,'interpolated_visits':0}
    for vehicle in batch:
        visits=cut_vehicle(vehicle,mapping,source,cut,transitions,kernel=kernel)
        counts['outside_or_discarded']+=not visits
        counts['repeated_visits']+=len(visits)>1
        counts['count']+=len(visits)
        for visit in visits:
            counts['interpolated_visits']+=visit['parameters']['sumo_boundary_timing']=='interpolated'
            element=etree.Element('vehicle',attrib=visit['attributes'])
            etree.SubElement(element,'route',attrib={**visit['route_attributes'],'edges':' '.join(visit['edges']),
                'exitTimes':' '.join(format(t,'.12g') for t in visit['exit_times'])})
            for key,value in visit['parameters'].items(): etree.SubElement(element,'param',key=key,value=str(value))
            rows.append({'id':visit['attributes']['id'],'depart':float(visit['attributes']['depart']),'xml':etree.tostring(element)})
    return rows,counts
