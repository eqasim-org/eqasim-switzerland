"""Serve viewport geometry and compact driver summaries from read-only indexes."""

import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory
from pyproj import Transformer

from .build import GROUPS, STATION_LINKS, link_statistics
from .purposes import purpose_summary


def create_app(output, sample):
    """Queries read compact indexes; no request reparses plans or events."""
    output = Path(output).resolve()
    app = Flask(__name__)
    features = dict(np.load(output / 'person_features.npz'))
    metadata = json.loads((output / 'person_features.json').read_text())
    stats = link_statistics(output)
    groups = np.load(output / 'link_group_users.npy', mmap_mode='r')
    offsets = np.memmap(output / 'link_member_offsets.bin', dtype='<u8', mode='r')
    members = np.memmap(output / 'link_members.bin', dtype='<u4', mode='r')
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)

    @contextmanager
    def database(name):
        connection = sqlite3.connect('file:' + str(output / name) + '?mode=ro&immutable=1', uri=True)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def row_for(link_id):
        with database('network.sqlite') as connection:
            row = connection.execute('SELECT l.*,s.coordinates FROM links l JOIN shapes s USING(idx) WHERE link_id=?', (link_id,)).fetchone()
        if row is None:
            abort(404)
        return dict(row)

    def distribution(values, labels=None):
        unique, counts = np.unique(values, return_counts=True)
        rows = []
        for value, count in zip(unique, counts):
            key = (labels[int(value)] if int(value) >= 0 else 'Unknown') if labels is not None else str(value or 'Unknown')
            rows.append({'label': key, 'count': int(count)})
        return sorted(rows, key=lambda item: -item['count'])

    @app.get('/')
    def index():
        return send_from_directory(Path(__file__).parent, 'map.html')

    @app.get('/assets/<path:filename>')
    def assets(filename):
        return send_from_directory(output / 'assets', filename)

    @app.get('/report/<filename>')
    def report_file(filename):
        allowed = {'diagnosis.html', 'corridor_diagnosis.json', 'corridor_trip_profiles.csv',
                   'cross_border_source_audit.csv', 'cross_border_source_audit.json',
                   'browser_validation.json', 'validation.json'}
        if filename not in allowed:
            abort(404)
        return send_from_directory(output, filename)

    @app.get('/api/meta')
    def meta():
        with database('network.sqlite') as connection:
            roads = connection.execute('SELECT highway,count(*) AS n FROM links GROUP BY highway ORDER BY n DESC').fetchall()
        return jsonify(groups=GROUPS, roads=[dict(row) for row in roads], sample=sample,
                       has_report=(output / 'diagnosis.html').exists(),
                       stations=STATION_LINKS, population=metadata,
                       notes=['Traffic volume comes from iteration-120 daily counts, scaled by 1/sample.',
                              'Agent profiles use event times 0 <= t < 24:00. The CSV keeps its original inclusive-midnight counting convention.',
                              'Group colours count distinct simulation agents in the sample, not passages or passenger occupancy.',
                              'Freight and LCV agents represent shipment/trip agents; distinct agents need not be distinct physical vehicles.',
                              'Employment/student status is unknown when not exported. Activity presence is a separate field.',
                              'Swiss cross-border users are identified by the cross-border flag and a home inside Switzerland.'])

    @app.get('/api/links')
    def links():
        try:
            west, south, east, north = map(float, request.args['bbox'].split(','))
            minimum = max(0., float(request.args.get('minimum', 0)))
            group = int(request.args.get('group', -1))
        except (KeyError, ValueError):
            abort(400)
        if not all(math.isfinite(x) for x in [west, south, east, north, minimum]) or not -180 <= west < east <= 180 or not -90 <= south < north <= 90 or not -1 <= group < len(GROUPS):
            abort(400)
        types = request.args.get('types', '').split(',')
        if not types or len(types) > 50:
            abort(400)
        limit = min(20000, max(500, int(request.args.get('limit', 12000))))
        placeholders = ','.join('?' for _ in types)
        with database('network.sqlite') as connection:
            rows = connection.execute(f'''SELECT l.idx,l.link_id,l.highway,l.daily_count,s.coordinates
                FROM spatial r JOIN links l USING(idx) JOIN shapes s USING(idx)
                WHERE r.minx<=? AND r.maxx>=? AND r.miny<=? AND r.maxy>=?
                AND l.highway IN ({placeholders}) AND l.daily_count>=?
                ORDER BY l.daily_count DESC LIMIT ?''',
                [east, west, north, south, *types, minimum * sample, limit + 1]).fetchall()
        result = []
        for row in rows[:limit]:
            index = row['idx']
            value = row['daily_count'] / sample if group < 0 else int(groups[index, group])
            # Keep zero-group roads faintly visible so overlap selection remains
            # useful and an absent group is not mistaken for a missing road.
            result.append([row['link_id'], index, row['highway'], round(row['daily_count'] / sample, 1),
                           value, json.loads(row['coordinates'])])
        return jsonify(links=result, truncated=len(rows) > limit, limit=limit,
                       metric='Daily traffic, full population' if group < 0 else GROUPS[group] + ': distinct sample agents')

    @app.get('/api/link/<path:link_id>')
    def link(link_id):
        row = row_for(link_id)
        index = row['idx']
        users = members[int(offsets[index]):int(offsets[index + 1])]
        ages = features['age'][users]
        bounds = [0, 6, 15, 18, 24, 30, 39, 65, np.inf]
        age_labels = ['0–5', '6–14', '15–17', '18–23', '24–29', '30–38', '39–64', '65+', 'Unknown']
        age_bin = np.searchsorted(bounds[1:], ages, side='right')
        age_bin[~np.isfinite(ages)] = 8
        s = stats[index]
        mean_time = float(s['travel_time'] / s['completed_links']) if s['completed_links'] else None
        sample_ids = users[np.linspace(0, len(users) - 1, min(25, len(users)), dtype=int)] if len(users) else users
        with database('people.sqlite') as connection:
            persons = [dict(r) for r in connection.execute('SELECT idx,person,age,canton,subpopulation FROM persons WHERE idx IN (' + ','.join('?' for _ in sample_ids) + ')', sample_ids.astype(int).tolist())] if len(sample_ids) else []
        # Homeless freight agents and invalid coordinates have no invented home.
        x, y = features['home_x'][users], features['home_y'][users]
        valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        cells, counts = np.unique(np.column_stack((np.floor(x[valid] / 2000), np.floor(y[valid] / 2000))), axis=0, return_counts=True)
        order = np.argsort(-counts)[:300]
        if len(order):
            lon, lat = transformer.transform((cells[order, 0] + .5) * 2000, (cells[order, 1] + .5) * 2000)
            homes = [[float(a), float(b), int(c)] for a, b, c in zip(lon, lat, counts[order])]
        else:
            homes = []
        summary = dict(row)
        summary['coordinates'] = json.loads(summary['coordinates'])
        summary.update(daily_flow=row['daily_count'] / sample, unique_agents=len(users),
            group_users=[{'label': name, 'count': int(groups[index, i])} for i, name in enumerate(GROUPS)],
            age=distribution(age_bin, age_labels), canton=distribution(features['canton'][users], metadata['cantons']),
            sex=distribution(features['sex'][users]),
            employed=distribution(features['employed'][users], ['No', 'Yes']),
            student=distribution(features['student'][users], ['No', 'Yes']),
            work_activity_users=int(features['work_activity'][users].sum()),
            education_activity_users=int(features['education_activity'][users].sum()),
            hourly=s['hourly'].astype(int).tolist(), event_entries=int(s['entries']),
            initial_vehicle_placements=int(s['initial']), stuck_events=int(s['stuck']),
            mean_time_seconds=mean_time, mean_speed_kmh=3.6 * row['length'] / mean_time if mean_time and mean_time > 0 else None,
            event_modes={name: int(s[name]) for name in ['car', 'freight', 'transit', 'bike', 'other']},
            persons=persons, homes=homes, homes_known=int(valid.sum()), homes_shown=int(counts[order].sum()) if len(order) else 0,
            home_grid_metres=2000)
        return jsonify(summary)

    @app.get('/api/person/<int:index>')
    def person(index):
        with database('people.sqlite') as connection:
            row = connection.execute('SELECT * FROM persons WHERE idx=?', (index,)).fetchone()
            if row is None:
                abort(404)
            trips = [dict(t) for t in connection.execute('SELECT * FROM trips WHERE person_idx=? ORDER BY sequence', (index,))]
        return jsonify(person=dict(row), trips=trips)

    @app.get('/api/purposes/link/<path:link_id>')
    def link_purposes(link_id):
        row = row_for(link_id)
        if not (output / 'purposes.sqlite').exists():
            return jsonify(available=False, groups=[])
        with database('purposes.sqlite') as connection:
            return jsonify(purpose_summary(connection, row['idx']))

    @app.get('/api/diagnosis')
    def diagnosis():
        path = output / 'corridor_diagnosis.json'
        return jsonify(json.loads(path.read_text()) if path.exists() else {'status': 'Extraction running'})

    @app.get('/api/corridor/link/<path:link_id>')
    def corridor_link(link_id):
        if not (output / 'corridor.sqlite').exists():
            return jsonify(count=0, purposes=[], journeys=[])
        row = row_for(link_id)
        with database('corridor.sqlite') as connection:
            purposes = [dict(r) for r in connection.execute('''SELECT coalesce(j.destination_purpose,'Unknown') AS label,count(*) AS count
                FROM route_links r JOIN journeys j USING(journey_id) WHERE r.link_idx=?
                GROUP BY j.destination_purpose ORDER BY count(*) DESC''', (row['idx'],))]
            journeys = [dict(r) for r in connection.execute('''SELECT j.journey_id,j.person,j.mode,j.start,j.status,j."group",
                j.origin_purpose,j.destination_purpose FROM route_links r JOIN journeys j USING(journey_id)
                WHERE r.link_idx=? ORDER BY j.start LIMIT 12''', (row['idx'],))]
        return jsonify(count=sum(r['count'] for r in purposes), purposes=purposes, journeys=journeys)

    @app.get('/api/journey/<int:index>')
    def journey(index):
        if not (output / 'corridor.sqlite').exists():
            abort(404)
        with database('corridor.sqlite') as connection:
            trip = connection.execute('SELECT * FROM journeys WHERE journey_id=?', (index,)).fetchone()
        if trip is None:
            abort(404)
        trip = dict(trip)
        route = list(map(int, trip.pop('links').split()))
        shapes = {}
        with database('network.sqlite') as connection:
            for lo in range(0, len(route), 500):
                batch = route[lo:lo + 500]
                for r in connection.execute('SELECT idx,coordinates FROM shapes WHERE idx IN (' + ','.join('?' for _ in batch) + ')', batch):
                    shapes[r['idx']] = json.loads(r['coordinates'])
        trip['paths'] = [shapes[k] for k in route if k in shapes]
        return jsonify(trip)

    return app
