"""Versioned Arrow network cache owned by consumers, never by input directories."""

import hashlib
import json
from pathlib import Path
import tempfile

import polars as pl
import shapely

SCHEMA = 1


def read_cached(path, context, reader):
    source = Path(path).resolve()
    stat = source.stat()
    identity = [SCHEMA, str(source), stat.st_mtime_ns, stat.st_size,
                hashlib.sha256(Path(reader.__code__.co_filename).read_bytes()).hexdigest()]
    root = Path(context.path()).resolve().parent / '_sumo_networks'
    root.mkdir(exist_ok=True)
    target = root / hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    if not (target / 'complete.json').exists():
        data = reader(source)
        with tempfile.TemporaryDirectory(dir=root, prefix='building_') as temporary:
            temporary = Path(temporary)
            rows = [{"id": key, "from_node": edge['from'], "to_node": edge['to'],
                     "length": edge['length'], "speed": edge['speed'],
                     "lanes": json.dumps(edge['lanes']), "geometry": shapely.to_wkb(edge['shape'])}
                    for key, edge in data['edges'].items()]
            schema = {'id': pl.String, 'from_node': pl.String, 'to_node': pl.String,
                      'length': pl.Float64, 'speed': pl.Float64, 'lanes': pl.String, 'geometry': pl.Binary}
            pl.DataFrame(rows, schema=schema).write_ipc(temporary / 'edges.arrow')
            pl.DataFrame({'attributes': [json.dumps(c) for c in data['connections']]},
                         schema={'attributes': pl.String}).write_ipc(temporary / 'connections.arrow')
            (temporary / 'complete.json').write_text(json.dumps({'identity': identity, 'location': data['location']}))
            try:
                temporary.rename(target)
                temporary.mkdir()  # TemporaryDirectory still owns its cleanup directory.
            except OSError:
                if not (target / 'complete.json').exists():
                    raise
        return data
    frame = pl.read_ipc(target / 'edges.arrow')
    shapes = shapely.from_wkb(frame['geometry'].to_numpy())
    edges = {}
    for row, shape in zip(frame.drop('geometry').iter_rows(named=True), shapes):
        edges[row['id']] = {'from': row['from_node'], 'to': row['to_node'], 'length': row['length'],
                            'speed': row['speed'], 'lanes': json.loads(row['lanes']), 'shape': shape}
    return {'edges': edges,
            'connections': [json.loads(row) for row in pl.read_ipc(target / 'connections.arrow')['attributes']],
            'location': json.loads((target / 'complete.json').read_text())['location']}
