"""Sort and export cached regional route parts without repeating national cutting."""

import json
from pathlib import Path

from lxml import etree
import polars as pl

from sumo.scenario.network.tools import read_network, validate_routes
from sumo.scenario.tools import read_routes, write_serialized_trips


def configure(context):
    context.stage('sumo.cutter.route_parts')


def execute(context):

    # Merge only retained regional Arrow rows and sort with Polars.

    source=context.stage('sumo.cutter.route_parts')
    frames=[pl.scan_ipc(path) for path in source['parts']]
    frame=(pl.concat(frames).sort('depart','id').collect() if frames else
           pl.DataFrame(schema={'id':pl.String,'depart':pl.Float64,'xml':pl.Binary}))
    if frame['id'].n_unique()!=frame.height:
        raise ValueError('Duplicate regional vehicle IDs')
    types={element.get('id'):element for element in etree.fromstring(b'<routes>'+Path(source['types']).read_bytes()+b'</routes>')}

    # Write and validate regional routes, retaining every contiguous routed visit.

    cache=Path(context.path()).resolve()
    output=cache/'region.routes.xml.gz'
    write_serialized_trips(output,types,frame['xml'])
    definitions,vehicles=read_routes(output)
    validate_routes(vehicles,read_network(source['network'],context=context),definitions)
    result={key:value for key,value in source.items() if key not in {'parts','types'}}
    result['routes']=str(output)
    (cache/'cut_report.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
