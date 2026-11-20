"""Airport passenger diary chains from the raw 2015 and 2021 microcensuses."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from data.airports.geneva.model import AIRPORTS, resolve
from data.activity_purposes import microcensus_purpose_map, format_activity_chain

VERSION = 3
MODES = {
    2015: {1: 'walk', 2: 'bike', 3: 'moped', 4: 'motorcycle_driver', 5: 'motorcycle_driver',
           6: 'motorcycle_passenger', 7: 'car_driver', 8: 'car_passenger', 9: 'rail',
           10: 'postbus', 11: 'bus', 12: 'tram', 13: 'taxi', 14: 'coach', 15: 'truck',
           16: 'ship', 17: 'air', 18: 'special_rail', 19: 'other', 20: 'ebike', 21: 'ebike', 95: 'other'},
    2021: {1: 'walk', 2: 'bike', 3: 'ebike', 4: 'ebike', 5: 'moped', 6: 'motorcycle_driver',
           7: 'motorcycle_driver', 8: 'motorcycle_passenger', 9: 'car_driver', 10: 'car_passenger',
           11: 'rail', 12: 'bus', 13: 'tram', 14: 'taxi', 15: 'ridehail', 16: 'coach',
           17: 'truck', 18: 'ship', 19: 'air', 20: 'special_rail', 21: 'other', 95: 'other'}}
PT_MODES = {'rail', 'bus', 'postbus', 'tram', 'special_rail', 'ship'}
# Raw survey names occur only in these input-column mappings.
SOURCE_COLUMNS = {
    'HHNR': 'source_person_id', 'WEGNR': 'trip_id', 'ETNR': 'stage_id',
    'alter': 'age', 'gesl': 'sex_code', 'USTag': 'diary_date_source', 'WP': 'person_weight',
    'f40800_01': 'employment_status_code', 'wzweck1': 'purpose_code',
    'wzweck2': 'return_home_code', 'f51100': 'departure_minutes', 'f51400': 'arrival_minutes',
    'f51300': 'mode_code', 'f51310a': 'car_ownership_code', 'f52900': 'stage_purpose_code',
    'f52950': 'previous_activity_code', 'rdist': 'distance_km',
    'S_LND': 'origin_country_code', 'Z_LND': 'destination_country_code',
    **{f'f4100{branch}{suffix}': f'{label}_activity_status_{number}'
       for branch, label in [(0, 'employed'), (1, 'nonemployed')]
       for number, suffix in enumerate(('a', 'b', 'c'), 1)}}
STATUS_FIELDS = [name for name in SOURCE_COLUMNS.values() if '_activity_status_' in name]


def configure(context):
    context.config('data_path')
    context.config('airport_model_airports', default=AIRPORTS)
    context.config('microcensus_airports_2015_folder', default='microcensus')
    context.config('microcensus_airports_2021_folder', default='microcensus/21')
    context.config('microcensus_airports_radius_m', default=3000.)


def execute(context):

    # Resolve the raw inputs when Synpp executes the stage.

    sources, settings = inputs(context)

    # Read both surveys independently, harmonize codes, and retain connected Swiss diary segments.

    surveys = []
    for year, folder in sources.items():
        persons, trips, stages = read_survey(folder, year)
        surveys.append(extract_chains(persons, trips, stages, year, settings['airports'], settings['radius_m']))
    result = pd.concat(surveys, ignore_index=True).convert_dtypes()
    summary = {'settings': settings, 'crs': 'EPSG:2056',
               'sources': {str(year): str(folder) for year, folder in sources.items()},
               'rows': len(result), 'by_survey': {str(y): len(f) for y, f in zip(sources, surveys)},
               'respondents': int(result.person_id.nunique()),
               'unit': 'One connected Swiss airport-related segment per respondent; chain_id uniquely identifies segments',
               'sex': '0=male, 1=female; missing/unknown stays null',
               'limitations': ['Airport identification uses air-stage endpoints within a configurable radius, not terminal polygons.',
                   'Ground access/egress must be observed in the same journey with its other endpoint in Switzerland.',
                   'Flights and foreign stages are excluded; the new airport purpose marks the boundary of each Swiss chain.',
                   'Raw survey weights are retained; editions are not pooled into an airport demand estimate.',
                   'Missing employment/student responses remain null. The household survey does not represent foreign hotel visitors.']}

    # Return the data and its provenance directly, without writing source-folder exports.

    result.attrs['summary'] = summary
    return result


def inputs(context):
    root = Path(context.config('data_path'))
    radius = context.config('microcensus_airports_radius_m')
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('microcensus_airports_radius_m must be finite and positive')
    sources = {year: resolve(root, context.config(f'microcensus_airports_{year}_folder')) for year in (2015, 2021)}
    settings = {'radius_m': float(radius), 'airports': context.config('airport_model_airports'), 'version': VERSION}
    return sources, settings


def read_survey(folder, year):
    """Rename source columns immediately, then work only with descriptive names."""
    sep = ',' if year == 2015 else ';'
    def read(name, names):
        return pd.read_csv(folder / name, sep=sep, encoding='latin1',
                           usecols=list(names), low_memory=False).rename(columns=names)

    # Attach the reported household home, independently of where the diary starts.

    person_names = {raw: name for raw, name in SOURCE_COLUMNS.items()
                    if name in ['source_person_id', 'age', 'sex_code', 'diary_date_source',
                                'person_weight', 'employment_status_code', *STATUS_FIELDS]}
    persons = read('zielpersonen.csv', person_names)
    home_names = {'HHNR': 'source_person_id',
                  'W_X_CH1903' if year == 2015 else 'W_X': 'home_source_x',
                  'W_Y_CH1903' if year == 2015 else 'W_Y': 'home_source_y'}
    homes = read('haushalte.csv', home_names)
    homes['home_x'], homes['home_y'] = project(homes.home_source_x, homes.home_source_y, year)
    persons = persons.merge(homes[['source_person_id', 'home_x', 'home_y']],
                            on='source_person_id', how='left', validate='one_to_one')

    # Retain every journey/stage; harmonize coordinate fields before projecting them.

    coordinate_names = {f'{prefix}_{axis}_CH1903' if year == 2015 else f'{prefix}_{axis}':
                        f'{side}_source_{axis.lower()}'
                        for prefix, side in [('S', 'origin'), ('Z', 'destination')] for axis in ('X', 'Y')}
    trip_fields = ['source_person_id', 'trip_id', 'purpose_code', 'return_home_code',
                   'departure_minutes', 'arrival_minutes']
    stage_fields = ['source_person_id', 'trip_id', 'stage_id', 'mode_code', 'car_ownership_code',
                    'stage_purpose_code', 'previous_activity_code', 'departure_minutes',
                    'arrival_minutes', 'distance_km', 'origin_country_code', 'destination_country_code']
    trips = read('wege.csv', {**{k: v for k, v in SOURCE_COLUMNS.items() if v in trip_fields}, **coordinate_names})
    stages = read('etappen.csv', {**{k: v for k, v in SOURCE_COLUMNS.items() if v in stage_fields}, **coordinate_names})
    for frame in (trips, stages):
        for side in ('origin', 'destination'):
            frame[side + '_x'], frame[side + '_y'] = project(frame[side + '_source_x'], frame[side + '_source_y'], year)
        frame['departure_seconds'] = frame.departure_minutes.where(frame.departure_minutes.ge(0)) * 60
        frame['arrival_seconds'] = frame.arrival_minutes.where(frame.arrival_minutes.ge(0)) * 60
    stages['mode_detailed'] = stages.mode_code.map(MODES[year]).fillna('unknown')
    car_type = stages.car_ownership_code.map({1: 'own_car', 2: 'company_car', 3: 'rental_car', 4: 'carshare', 5: 'other_car'})
    drivers = stages.mode_detailed.eq('car_driver') & car_type.notna()
    stages.loc[drivers, 'mode_detailed'] = car_type.loc[drivers]
    return persons, trips, stages


def project(x, y, year):
    x, y = pd.to_numeric(x, errors='coerce'), pd.to_numeric(y, errors='coerce')
    missing = [-99, -98, -97, -96, -95, -9, 0]
    valid = np.isfinite(x) & np.isfinite(y) & ~x.isin(missing) & ~y.isin(missing)
    if year == 2021:
        valid &= x.between(-180, 180) & y.between(-90, 90)
    east, north = np.full(len(x), np.nan), np.full(len(y), np.nan)
    transform = Transformer.from_crs(21781 if year == 2015 else 4326, 2056, always_xy=True)
    east[valid], north[valid] = transform.transform(x.loc[valid].to_numpy(), y.loc[valid].to_numpy())
    return east, north


def airport_events(stages, airports, radius):
    """Identify observed ground blocks adjacent to a flight at a Swiss airport."""
    events = []
    for (person, trip), chain in stages.groupby(['source_person_id', 'trip_id'], sort=True):
        chain = chain.sort_values('stage_id')
        for _, flight in chain.loc[chain.mode_detailed.eq('air')].iterrows():
            for side, direction in [('origin', 'to_airport'), ('destination', 'from_airport')]:
                distances = {a: np.hypot(flight[side + '_x'] - xy[0], flight[side + '_y'] - xy[1]) for a, xy in airports.items()}
                airport = min(distances, key=distances.get)
                if not np.isfinite(distances[airport]) or distances[airport] > radius:
                    continue
                ground = chain.loc[chain.stage_id.lt(flight.stage_id) if side == 'origin' else chain.stage_id.gt(flight.stage_id)]
                if ground.empty or ground.mode_detailed.eq('air').any():
                    continue
                endpoint = ground.iloc[0] if side == 'origin' else ground.iloc[-1]
                if endpoint['origin_country_code' if side == 'origin' else 'destination_country_code'] != 8100:
                    continue
                # Prefer a motorized mode over access walking; retain every stage mode too.
                distances_by_mode = ground.groupby('mode_detailed').distance_km.sum().clip(lower=0)
                motorized = distances_by_mode.drop(labels=['walk', 'bike', 'ebike', 'unknown'], errors='ignore')
                detailed = (motorized if not motorized.empty else distances_by_mode).idxmax()
                events.append({'person': int(person), 'trip': int(trip), 'flight_stage': int(flight.stage_id),
                    'airport': airport, 'direction': direction, 'mode_detailed': detailed,
                    'airport_match_distance_m': float(distances[airport]),
                    'mode': 'pt' if detailed in PT_MODES else detailed,
                    'mode_chain': ground.mode_detailed.tolist(),
                    'departure_seconds': finite(ground.iloc[0].departure_seconds),
                    'arrival_seconds': finite(ground.iloc[-1].arrival_seconds),
                    'boundary_stage': int(ground.iloc[-1 if side == 'origin' else 0].stage_id)})
    return events


def extract_chains(persons, trips, stages, year, coordinates, radius):
    """Return connected Swiss diary segments, preserving respondent and survey IDs."""
    transform = Transformer.from_crs(4326, 2056, always_xy=True)
    airports = {a: transform.transform(c['longitude'], c['latitude']) for a, c in coordinates.items()}
    # Inspect air travellers' raw days, then retain only the connected Swiss portions.
    ids = stages.loc[stages.mode_detailed.eq('air'), 'source_person_id'].unique()
    events = airport_events(stages.loc[stages.source_person_id.isin(ids)], airports, radius)

    # Group events by respondent before splitting each raw diary at flights and foreign stages.

    by_person = {}
    for event in events:
        by_person.setdefault(event['person'], []).append(event)
    people = persons.set_index('source_person_id')
    if not people.index.is_unique:
        raise ValueError('Microcensus respondent IDs must be unique within each survey')
    trip_groups = {int(p): g.sort_values('trip_id') for p, g in trips.loc[trips.source_person_id.isin(by_person)].groupby('source_person_id')}
    stage_groups = {int(p): g.sort_values(['trip_id', 'stage_id']) for p, g in stages.loc[stages.source_person_id.isin(by_person)].groupby('source_person_id')}
    output = []
    for person, selected in sorted(by_person.items()):
        attributes = people.loc[person]
        diary, legs = trip_groups[person], stage_groups[person]
        home = (attributes.home_x, attributes.home_y)
        names = sorted({e['airport'] for e in selected})
        distances = {a: finite(np.hypot(home[0] - airports[a][0], home[1] - airports[a][1])) for a in names}
        for event in selected:
            event['home_airport_distance_m'] = distances[event['airport']]
            event.pop('person')

        # Derive attributes from reported status codes; skipped answers stay unknown.

        status = attributes[STATUS_FIELDS]
        student, employed = pd.NA, pd.NA
        if status.eq(32).any() or attributes.employment_status_code == 5:
            student = True  # Reported education or apprenticeship.
        elif status.ge(0).any():
            student = False
        if attributes.employment_status_code in (1, 2, 3, 4, 5):
            employed = True
        elif attributes.employment_status_code == -99 and attributes.age >= 15:
            employed = False  # Computed status says not employed; children were not asked.
        base = {'person_id': f'mz{year}_{person}', 'source_person_id': person, 'survey_year': year,
            'age': attributes.age if attributes.age >= 0 else pd.NA,
            'sex': {1: 0, 2: 1}.get(attributes.sex_code, pd.NA), 'employed': employed, 'is_student': student,
            'home_x': finite(home[0]), 'home_y': finite(home[1]),
            'source_diary_trip_count': len(diary),
            'diary_date': str(pd.to_datetime(attributes.diary_date_source, format='%m/%d/%Y' if year == 2015 else '%d.%m.%Y')),
            'person_weight': float(attributes.person_weight)}

        # Split at every flight or foreign stage, so no artificial leg joins Swiss fragments.

        for index, segment in enumerate(swiss_chains(diary, legs, home, selected, year)):
            events = segment.pop('events')
            names = sorted({e['airport'] for e in events})
            segment_distances = {a: distances[a] for a in names}
            output.append({**base, **segment, 'chain_id': f'mz{year}_{person}_segment{index}',
                'segment_index': index,
                'home_airport_distance_m': segment_distances[names[0]] if len(names) == 1 else None,
                'home_airport_distances_m': json.dumps(segment_distances, allow_nan=False),
                'airport': '|'.join(names), 'direction': '|'.join(sorted({e['direction'] for e in events})),
                'mode': '|'.join(sorted({e['mode'] for e in events})),
                'mode_detailed': '|'.join(sorted({e['mode_detailed'] for e in events})),
                'airport_event_count': len(events), 'airport_events': json.dumps(events, allow_nan=False)})
    columns = ['person_id', 'chain_id', 'segment_index', 'source_person_id', 'survey_year', 'age', 'sex', 'employed', 'is_student',
        'home_x', 'home_y', 'home_airport_distance_m', 'home_airport_distances_m', 'airport', 'direction',
        'mode', 'mode_detailed', 'airport_event_count', 'airport_events', 'activity_chain', 'mode_chain',
        'activity_chain_json', 'diary_trip_count', 'source_diary_trip_count', 'diary_date', 'person_weight']
    return pd.DataFrame(output, columns=columns)


def broad_mode(detailed):
    """Use the same mode vocabulary as the cleaned microcensus chains."""
    if detailed in PT_MODES or detailed in ('taxi', 'ridehail', 'coach'):
        return 'pt'
    if detailed in ('bike', 'ebike'):
        return 'bike'
    if detailed in ('walk', 'car_passenger', 'unknown'):
        return detailed
    return 'car' if detailed in ('own_car', 'company_car', 'rental_car', 'carshare', 'other_car',
                                 'car_driver', 'truck', 'moped', 'motorcycle_driver', 'motorcycle_passenger') else 'unknown'


def swiss_chains(trips, stages, home, events, year):
    """Keep only connected Swiss ground stages; airports terminate or start chains."""
    stages = stages.sort_values(['trip_id', 'stage_id']).copy()
    keep = (stages.origin_country_code.eq(8100) & stages.destination_country_code.eq(8100)
            & stages.mode_detailed.ne('air'))
    # Every excluded stage breaks the day: never create a trip across an omitted flight.
    stages['segment'] = (~keep).cumsum()
    purposes = microcensus_purpose_map(year)
    segments = []
    for _, ground in stages.loc[keep].groupby('segment', sort=False):
        keys = set(zip(ground.trip_id.astype(int), ground.stage_id.astype(int)))
        selected = [e for e in events if (e['trip'], e['boundary_stage']) in keys]
        if not selected:
            continue
        arrivals = {(e['trip'], e['boundary_stage']): e for e in selected if e['direction'] == 'from_airport'}
        departures = {(e['trip'], e['boundary_stage']): e for e in selected if e['direction'] == 'to_airport'}
        first = ground.iloc[0]
        first_event = arrivals.get((int(first.trip_id), int(first.stage_id)))
        if first_event:
            initial = 'airport'
        elif np.hypot(first.origin_x - home[0], first.origin_y - home[1]) <= 100:
            initial = 'home'
        else:
            initial = purposes.get(first.previous_activity_code, 'other')
            if initial == 'unknown':
                initial = 'other'  # Same fallback as the ordinary reader's missing initial purpose.
        activities = [{'purpose': initial, 'x': finite(first.origin_x), 'y': finite(first.origin_y),
                       'arrival_seconds': None, 'departure_seconds': finite(first.departure_seconds)}]
        if first_event:
            activities[0]['airport'] = first_event['airport']
        legs = []
        for trip_id, block in ground.groupby('trip_id', sort=False):
            start, end = block.iloc[0], block.iloc[-1]
            original = trips.loc[trips.trip_id.eq(trip_id)].iloc[0]
            last_event = departures.get((int(trip_id), int(end.stage_id)))
            purpose = 'airport' if last_event else purposes.get(original.purpose_code, 'unknown')
            if not last_event and original.return_home_code > 1:
                purpose = 'home'
            distances = block.groupby('mode_detailed').distance_km.sum().clip(lower=0)
            vehicular = distances.drop(labels=['walk', 'bike', 'ebike', 'unknown'], errors='ignore')
            detailed = (vehicular if not vehicular.empty else distances).idxmax()
            mode = 'car_passenger' if block.mode_detailed.eq('car_passenger').any() else broad_mode(detailed)
            activities[-1]['departure_seconds'] = finite(start.departure_seconds)
            activity = {'purpose': purpose, 'x': finite(end.destination_x), 'y': finite(end.destination_y),
                        'arrival_seconds': finite(end.arrival_seconds)}
            if last_event:
                activity['airport'] = last_event['airport']
            activities.append(activity)
            legs.append({'source_trip_id': int(trip_id), 'mode': mode, 'mode_detailed': detailed,
                         'departure_seconds': finite(start.departure_seconds), 'arrival_seconds': finite(end.arrival_seconds)})
        fields = ['trip_id', 'stage_id', 'mode_detailed', 'origin_x', 'origin_y', 'destination_x', 'destination_y',
                  'origin_country_code', 'destination_country_code', 'departure_seconds', 'arrival_seconds',
                  'stage_purpose_code', 'previous_activity_code']
        records = json.loads(ground[fields].to_json(orient='records'))
        segments.append({'activity_chain': format_activity_chain([a['purpose'] for a in activities]),
            'mode_chain': format_activity_chain([leg['mode'] for leg in legs]),
            'diary_trip_count': len(legs), 'events': selected,
            'activity_chain_json': json.dumps({'activities': activities, 'trips': legs, 'stages': records}, allow_nan=False)})
    return segments


def finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None
