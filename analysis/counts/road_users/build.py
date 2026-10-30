"""Build disk indexes without materialising event or population XML trees."""

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from lxml import etree
from pyproj import Transformer


logger = logging.getLogger(__name__)
STAT_DTYPE = np.dtype([
    (name, '<u4') for name in ['entries', 'initial', 'car', 'freight', 'transit', 'bike', 'other', 'stuck']
] + [('travel_time', '<f8'), ('completed_links', '<u4'), ('padding', '<u4'), ('hourly', '<u4', 32)])
assert STAT_DTYPE.itemsize == 176
STATION_LINKS = {'24': ['2391498', '2927869'], '505': ['2391502', '1289297'], '504': ['1289295', '106813']}
GROUPS = ['Swiss residents', 'French residents', 'Other cross-border', 'Freight', 'LCV',
          'Swiss cross-border', 'Other / unknown']


def connect(path):
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=OFF')
    connection.execute('PRAGMA synchronous=OFF')
    connection.execute('PRAGMA cache_size=-64000')
    return connection


def build_network(output, network_path, geometry_path):
    """Retain directed road metadata and index geometry by viewport bounds."""
    output = Path(output)
    ids = (output / 'link_ids.txt').read_text().splitlines()
    lookup = {link: index for index, link in enumerate(ids)}
    daily = np.load(output / 'reference_daily_counts.npy', mmap_mode='r')
    temporary = output / 'network.build.sqlite'
    if temporary.exists():
        temporary.unlink()
    connection = connect(temporary)
    connection.executescript('''
        CREATE TABLE links (
            idx INTEGER PRIMARY KEY, link_id TEXT UNIQUE, from_node TEXT, to_node TEXT,
            length REAL, speed REAL, capacity REAL, lanes REAL, modes TEXT,
            highway TEXT, osm_id TEXT, from_x REAL, from_y REAL, to_x REAL, to_y REAL,
            daily_count REAL);
        CREATE TABLE shapes (idx INTEGER PRIMARY KEY, coordinates TEXT);
        CREATE VIRTUAL TABLE spatial USING rtree(idx, minx, maxx, miny, maxy);
    ''')
    nodes, records = {}, []
    import gzip
    with gzip.open(network_path, 'rb') as stream:
        for _, element in etree.iterparse(stream, events=('end',), tag=('node', 'link'),
                                         load_dtd=False, no_network=True):
            if element.tag == 'node':
                nodes[element.get('id')] = (float(element.get('x')), float(element.get('y')))
            else:
                index = lookup.get(element.get('id'))
                if index is not None:
                    attributes = {a.get('name'): a.text for a in element.findall('./attributes/attribute')}
                    start, end = nodes[element.get('from')], nodes[element.get('to')]
                    records.append((index, element.get('id'), element.get('from'), element.get('to'),
                                    float(element.get('length')), float(element.get('freespeed')),
                                    float(element.get('capacity')), float(element.get('permlanes')),
                                    element.get('modes'), attributes.get('osm:way:highway') or 'unclassified',
                                    attributes.get('osm:way:id'), *start, *end, float(daily[index])))
                    if len(records) >= 20000:
                        connection.executemany('INSERT INTO links VALUES (' + ','.join(['?'] * 16) + ')', records)
                        records.clear()
            # Remove processed siblings as well as children to bound parser memory.
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    if records:
        connection.executemany('INSERT INTO links VALUES (' + ','.join(['?'] * 16) + ')', records)
    del nodes
    connection.commit()
    logger.info('Network metadata: %d links', connection.execute('SELECT count(*) FROM links').fetchone()[0])

    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    seen = 0
    for chunk in pd.read_csv(geometry_path, usecols=['LinkId', 'Geometry'], dtype=str, chunksize=30000):
        indices = chunk.LinkId.map(lookup)
        valid = indices.notna()
        indices = indices[valid].astype(int).to_numpy()
        geometries = shapely.from_wkt(chunk.loc[valid, 'Geometry'].to_numpy())
        # Keep genuine bends while avoiding excessive browser geometry detail.
        geometries = shapely.simplify(geometries, 1.0, preserve_topology=True)
        geometries = shapely.transform(geometries, transformer.transform, interleaved=False)
        bounds = shapely.bounds(geometries)
        shape_rows, spatial_rows = [], []
        for index, geometry, bound in zip(indices, geometries, bounds):
            coords = np.round(shapely.get_coordinates(geometry), 6).tolist()
            if len(coords) < 2 or not np.isfinite(bound).all():
                raise ValueError('Invalid detailed shape for link %s' % ids[index])
            shape_rows.append((int(index), json.dumps(coords, separators=(',', ':'))))
            spatial_rows.append((int(index), float(bound[0]), float(bound[2]), float(bound[1]), float(bound[3])))
        connection.executemany('INSERT INTO shapes VALUES (?,?)', shape_rows)
        connection.executemany('INSERT INTO spatial VALUES (?,?,?,?,?)', spatial_rows)
        seen += len(shape_rows)
        if seen % 300000 < 30000:
            logger.info('Indexed detailed shapes: %d', seen)
        connection.commit()
    connection.execute('CREATE INDEX road_type ON links(highway)')
    connection.execute('ANALYZE')
    connection.commit()
    missing = connection.execute('''SELECT count(*) FROM links l LEFT JOIN shapes s USING(idx)
        WHERE s.idx IS NULL AND l.highway IN ('motorway','motorway_link','trunk','trunk_link','primary','primary_link')''').fetchone()[0]
    if missing:
        raise ValueError('%d major road links have no matching geometry' % missing)
    connection.close()
    temporary.replace(output / 'network.sqlite')
    logger.info('Finished network index: %d shapes', seen)


def build_people(output):
    """Keep person and trip information on disk; unavailable statuses stay unknown."""
    output = Path(output)
    temporary = output / 'people.build.sqlite'
    if temporary.exists():
        temporary.unlink()
    connection = connect(temporary)
    for filename, table in [('persons.tsv', 'persons'), ('trips.tsv', 'trips')]:
        first = True
        count = 0
        for frame in pd.read_csv(output / filename, sep='\t', chunksize=50000,
                                 dtype={'person': str, 'vehicle': str, 'canton': str}, keep_default_na=False):
            for field in ['idx', 'person_idx', 'sequence', 'station_mask', 'has_work_activity', 'has_education_activity']:
                if field in frame:
                    frame[field] = pd.to_numeric(frame[field], errors='raise').astype('int64')
            for field in ['age', 'home_x', 'home_y', 'departure', 'origin_x', 'origin_y', 'destination_x', 'destination_y']:
                if field in frame:
                    frame[field] = pd.to_numeric(frame[field], errors='coerce')
            frame.to_sql(table, connection, if_exists='replace' if first else 'append', index=False)
            first = False
            count += len(frame)
        logger.info('Indexed %d %s', count, table)
    connection.execute('CREATE UNIQUE INDEX person_idx ON persons(idx)')
    connection.execute('CREATE UNIQUE INDEX person_id ON persons(person)')
    connection.execute('CREATE INDEX trip_person ON trips(person_idx, vehicle, departure)')
    connection.execute('CREATE INDEX corridor_plan ON trips(station_mask)')
    connection.commit()
    connection.close()
    temporary.replace(output / 'people.sqlite')


def link_statistics(output):
    return np.memmap(Path(output) / 'link_stats.bin', dtype=STAT_DTYPE, mode='r')


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def index_person_features(output, swiss_border):
    """Use exported flags and home geography; never infer student status from age."""
    output = Path(output)
    with sqlite3.connect(output / 'people.sqlite') as connection:
        people = pd.read_sql_query('SELECT * FROM persons ORDER BY idx', connection)
    if not np.array_equal(people.idx, np.arange(len(people))):
        raise ValueError('Person indexes are not contiguous')
    groups = np.zeros(len(people), dtype='uint8')
    external = people.is_external_fr.astype(str).str.lower().isin(['true', '1'])
    border = people.is_crossborder.astype(str).str.lower().isin(['true', '1'])
    freight = people.is_freight.astype(str).str.lower().isin(['true', '1'])
    lcv = people.person.str.startswith('lcv_')
    groups[external] = 1
    groups[border] = 2
    # Swiss cross-border commuters remain separately visible. Missing homes are
    # not silently treated as Swiss, nor classified from a destination canton.
    points = shapely.points(people.home_x.fillna(np.nan), people.home_y.fillna(np.nan))
    shapely.prepare(swiss_border)
    inside = shapely.covers(swiss_border, points)
    groups[border & inside] = 5
    groups[freight] = 3
    groups[lcv] = 4
    canton_codes, canton_labels = pd.factorize(people.canton.fillna('Unknown').replace('', 'Unknown'), sort=True)
    def status(field):
        text = people[field].astype(str).str.lower()
        return np.where(text.isin(['true', '1']), 1, np.where(text.isin(['false', '0']), 0, -1)).astype('int8')
    age = pd.to_numeric(people.age, errors='coerce').to_numpy(dtype='float32')
    age[(age < 0) | (age > 120)] = np.nan
    # Unknown ages in freight agents remain unknown rather than becoming adults.
    np.savez_compressed(output / 'person_features.npz', age=age, group=groups,
        canton=canton_codes.astype('int16'), home_x=people.home_x.to_numpy(dtype='float64'),
        home_y=people.home_y.to_numpy(dtype='float64'), sex=people.sex.fillna('').to_numpy(dtype='U8'),
        employed=status('employed'), student=status('student'),
        work_activity=people.has_work_activity.to_numpy(dtype='uint8'),
        education_activity=people.has_education_activity.to_numpy(dtype='uint8'))
    metadata = {'groups': GROUPS, 'cantons': canton_labels.tolist(), 'persons': len(people),
                'person_group_counts': dict(zip(GROUPS, np.bincount(groups, minlength=len(GROUPS)).tolist())),
                'employment_known': int((status('employed') >= 0).sum()),
                'student_status_known': int((status('student') >= 0).sum())}
    (output / 'person_features.json').write_text(json.dumps(metadata, indent=2))
    logger.info('Person groups: %s', metadata['person_group_counts'])


def index_user_groups(output):
    """Count exact distinct simulation agents by group, with small memory chunks."""
    output = Path(output)
    offsets = np.memmap(output / 'link_member_offsets.bin', dtype='<u8', mode='r')
    members = np.memmap(output / 'link_members.bin', dtype='<u4', mode='r')
    with np.load(output / 'person_features.npz') as data:
        groups = data['group']
    counts = np.lib.format.open_memmap(output / 'link_group_users.npy', mode='w+', dtype='<u4',
                                     shape=(len(offsets) - 1, len(GROUPS)))
    for lo in range(0, len(counts), 20000):
        hi = min(lo + 20000, len(counts))
        local = np.repeat(np.arange(hi - lo, dtype='uint32'), np.diff(offsets[lo:hi + 1]).astype('int64'))
        selected = members[int(offsets[lo]):int(offsets[hi])]
        values = np.bincount(local * len(GROUPS) + groups[selected], minlength=(hi - lo) * len(GROUPS))
        counts[lo:hi] = values.reshape(-1, len(GROUPS))
    counts.flush()
    if int(counts.sum(dtype='uint64')) != len(members):
        raise ValueError('Driver group totals do not conserve exact memberships')
    logger.info('Indexed %d distinct link/agent pairs by group', len(members))
