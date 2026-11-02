"""Conserve directional station flows and trace the journeys between them."""

import json
import logging
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .build import GROUPS, STATION_LINKS, connect, link_statistics


logger = logging.getLogger(__name__)


def diagnose(output, sample):
    """Use full journeys to distinguish exits, arrivals after midnight and aborts."""
    output = Path(output)
    full = output / 'all_30h'
    source = full if (full / 'corridor_journeys.tsv').exists() else output
    journeys = pd.read_csv(source / 'corridor_journeys.tsv', sep='\t', keep_default_na=False)
    with sqlite3.connect(output / 'people.sqlite') as connection:
        planned = pd.read_sql_query('SELECT * FROM trips WHERE station_mask>0', connection)
        people = pd.read_sql_query('SELECT idx,person,age,canton,subpopulation FROM persons', connection)
    journeys = journeys.merge(planned, on=['person_idx', 'sequence'], how='left', suffixes=('', '_plan'), validate='one_to_one')
    matched = journeys.vehicle.eq(journeys.vehicle_plan)
    # An unmatched selected-plan leg must not supply a guessed trip purpose.
    plan_columns = ['origin_purpose', 'origin_x', 'origin_y', 'origin_municipality',
                    'destination_purpose', 'destination_x', 'destination_y', 'destination_municipality']
    journeys.loc[~matched, plan_columns] = None
    journeys = journeys.merge(people, left_on='person_idx', right_on='idx', how='left', validate='many_to_one')
    features = dict(np.load(output / 'person_features.npz'))
    journeys['group'] = journeys.person_idx.map(lambda i: GROUPS[int(features['group'][i])] if i < len(features['group']) else 'Other / unknown')
    lookup = _road_lookup(output, journeys.links)
    ids = {row['link_id']: index for index, row in lookup.items()}
    times = journeys.station_times.map(lambda text: [(int(part.split(':')[0]), float(part.split(':')[1])) for part in text.split(',') if part])
    station_rows = []
    stats = link_statistics(output)
    for station, links in STATION_LINKS.items():
        for link in links:
            index = ids[link]
            journeys[link] = times.map(lambda sequence: any(k == index and t < 86400 for k, t in sequence))
            journeys[link + '_later'] = times.map(lambda sequence: any(k == index and t >= 86400 for k, t in sequence))
            row = lookup[index]
            direction = 'Geneva' if row['from_y'] > row['to_y'] else 'Lausanne'
            count = int(stats[index]['hourly'][:24].sum())
            station_rows.append({'station': station, 'direction': direction, 'link_id': link,
                                 'sample_passages': count, 'scaled_passages': count / sample,
                                 'reference_scaled': row['daily_count'] / sample,
                                 'stuck_on_link': int(stats[index]['stuck'])})
    cohorts = {}
    definitions = [
        ('south_exit', '2391502', '1289295', 'after'),
        ('south_entry', '1289295', '2391502', 'before'),
        ('north_entry', '1289297', '106813', 'before'),
        ('north_exit', '106813', '1289297', 'after'),
    ]
    transformer = Transformer.from_crs(2056, 4326, always_xy=True)
    journeys['cohort'] = ''
    for name, present, absent, position in definitions:
        mask = journeys[present] & ~journeys[absent]
        journeys.loc[mask, 'cohort'] = name
        subset = journeys.loc[mask]
        ramps = []
        for route in subset.links:
            path = np.fromstring(route, sep=' ', dtype='uint32').tolist()
            anchor = path.index(ids[present])
            candidates = path[anchor + 1:] if position == 'after' else path[:anchor][::-1]
            ramp = next((k for k in candidates if lookup[k]['highway'] == 'motorway_link'), None)
            ramps.append(ramp)
        top_ramps = []
        for index, count in Counter(ramps).most_common(12):
            if index is None:
                top_ramps.append({'link_id': 'No motorway ramp found', 'sample_trips': count})
                continue
            row = lookup[index]
            lon, lat = transformer.transform((row['from_x'] + row['to_x']) / 2, (row['from_y'] + row['to_y']) / 2)
            top_ramps.append({'link_id': row['link_id'], 'sample_trips': count, 'lon': lon, 'lat': lat})
        cohorts[name] = {
            'sample_trips': len(subset), 'scaled_trips': len(subset) / sample,
            'unique_agents': int(subset.person_idx.nunique()),
            'groups': subset.group.value_counts().to_dict(),
            'event_status': subset.status.value_counts().to_dict(),
            'destination_purpose': subset.destination_purpose.fillna('Unknown').value_counts().to_dict(),
            'origin_purpose': subset.origin_purpose.fillna('Unknown').value_counts().to_dict(),
            'destination_municipality': subset.destination_municipality.fillna('Unknown').value_counts().head(12).to_dict(),
            'origin_municipality': subset.origin_municipality.fillna('Unknown').value_counts().head(12).to_dict(),
            'counterpart_crossed_after_midnight': int(subset[absent + '_later'].sum()),
            'ramps': top_ramps,
            'plan_match_fraction': float(matched.loc[mask].mean()) if len(subset) else None,
        }
        bit = 1 << [item for values in STATION_LINKS.values() for item in values].index(absent)
        planned_counterpart = pd.to_numeric(subset.station_mask_plan, errors='coerce').fillna(0).astype(int).map(lambda value: bool(value & bit))
        cohorts[name]['matched_plans_including_counterpart'] = int((planned_counterpart & matched.loc[mask]).sum())
        cohorts[name]['matched_plans_including_counterpart_without_late_crossing'] = int((planned_counterpart & matched.loc[mask] & ~subset[absent + '_later']).sum())
    directional = {row['link_id']: row['sample_passages'] for row in station_rows}
    conservation = {
        'south_505_minus_504': directional['2391502'] - directional['1289295'],
        'south_exits_minus_entries': cohorts['south_exit']['sample_trips'] - cohorts['south_entry']['sample_trips'],
        'north_505_minus_504': directional['1289297'] - directional['106813'],
        'north_entries_minus_exits': cohorts['north_entry']['sample_trips'] - cohorts['north_exit']['sample_trips'],
    }
    if conservation['south_505_minus_504'] != conservation['south_exits_minus_entries'] or conservation['north_505_minus_504'] != conservation['north_entries_minus_exits']:
        raise ValueError('Repeated station passages require passage-level rather than journey-level cohort accounting')
    contributions = {}
    for group in GROUPS:
        contributions[group] = sum(sign * cohorts[name]['groups'].get(group, 0) for name, sign in
                                  [('south_exit', 1), ('south_entry', -1), ('north_entry', 1), ('north_exit', -1)]) / sample
    diagnosis = {
        'window_seconds': [0, 86400],
        'full_events_end_seconds': json.loads((source / 'events_summary.json').read_text()).get('last_event_time'),
        'sample': sample, 'corridor_journeys_full_simulation': len(journeys),
        'selected_plan_match_fraction': float(matched.mean()),
        'stations': station_rows, 'cohorts': cohorts, 'conservation': conservation,
        'contribution_to_505_minus_504_full_population': contributions,
        'interpretation_limits': [
            'These observations locate simulated interchange traffic; they do not establish that any individual trip is wrong.',
            'The available observed reference is a combined-direction daily total, not an observed origin-destination matrix.',
            'Foreign cross-border agents can have border proxies as their simulated home, rather than a true residential address.',
            'Selected-plan purposes are attached only when person, network-leg sequence and vehicle agree.',
            'An unfinished journey at the end of the available event window is not evidence of removal or failure.',
        ],
    }
    (output / 'corridor_diagnosis.json').write_text(json.dumps(diagnosis, indent=2))
    pd.DataFrame(station_rows).to_csv(output / 'station_directional_comparison.csv', index=False)
    keep = ['person_idx', 'sequence', 'person', 'age', 'group', 'canton', 'vehicle', 'mode',
            'start', 'end', 'status', 'cohort', *plan_columns]
    journeys[keep].to_csv(output / 'corridor_trip_profiles.csv', index=False)
    _index_corridor(output, journeys)
    logger.info('Directional flow conservation: %s', conservation)
    logger.info('Group contributions to 505 minus 504: %s', contributions)
    return diagnosis


def _road_lookup(output, routes):
    wanted = set()
    for route in routes:
        wanted.update(map(int, route.split()))
    lookup = {}
    with sqlite3.connect(Path(output) / 'network.sqlite') as connection:
        connection.row_factory = sqlite3.Row
        wanted = list(wanted)
        for lo in range(0, len(wanted), 800):
            batch = wanted[lo:lo + 800]
            for row in connection.execute('SELECT * FROM links WHERE idx IN (' + ','.join('?' for _ in batch) + ')', batch):
                lookup[row['idx']] = dict(row)
    return lookup


def _index_corridor(output, full_profiles):
    """Retain only actual daily routes, so purpose inspection uses the same window."""
    output = Path(output)
    daily = pd.read_csv(output / 'corridor_journeys.tsv', sep='\t', keep_default_na=False)
    metadata = full_profiles[['person_idx', 'sequence', 'person', 'group', 'origin_purpose', 'destination_purpose',
                              'origin_x', 'origin_y', 'destination_x', 'destination_y']]
    daily = daily.merge(metadata, on=['person_idx', 'sequence'], how='left', validate='one_to_one')
    daily['journey_id'] = np.arange(len(daily))
    temporary = output / 'corridor.build.sqlite'
    if temporary.exists():
        temporary.unlink()
    connection = connect(temporary)
    daily.to_sql('journeys', connection, index=False)
    connection.execute('CREATE TABLE route_links(link_idx INTEGER,journey_id INTEGER,position INTEGER)')
    rows = []
    for trip in daily.itertuples():
        route = list(map(int, trip.links.split()))
        first = 1 if route and route[0] == trip.start_link else 0
        seen = set()
        for position, index in enumerate(route[first:], start=first):
            if index not in seen:
                rows.append((index, trip.journey_id, position))
                seen.add(index)
        if len(rows) > 50000:
            connection.executemany('INSERT INTO route_links VALUES (?,?,?)', rows)
            rows.clear()
    connection.executemany('INSERT INTO route_links VALUES (?,?,?)', rows)
    connection.execute('CREATE UNIQUE INDEX route_by_link ON route_links(link_idx,journey_id)')
    connection.execute('CREATE UNIQUE INDEX journey_by_id ON journeys(journey_id)')
    connection.commit()
    connection.close()
    temporary.replace(output / 'corridor.sqlite')
