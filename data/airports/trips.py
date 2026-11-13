"""Sample daily Geneva/Zurich passenger access trips from municipality ODs."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from data.airports import common, geneva, geneva_times, zurich
from data.airports.common import apportion, atomic_write, export_signature, file_digest, normalized
from data.airports.model import resolve

logger = logging.getLogger('synpp')
MODES = ('dropoff', 'private_car', 'rail', 'bus_tram_aerobus', 'taxi', 'ridehail', 'other')
SPEEDS = dict(zip(MODES, (60., 60., 70., 30., 50., 50., 40.)))
VERSION = 3


def configure(context):
    context.stage('data.airports.geneva')
    context.stage('data.airports.zurich')
    context.stage('data.airports.model')
    context.stage('data.airports.locations')
    context.config('data_path')
    context.config('random_seed')
    context.config('airport_trips_day', default='all')
    context.config('airport_trips_sample_rate', default=1.)
    context.config('airport_trips_speeds_kmh', default=SPEEDS)
    context.config('airport_trips_detour_factor', default=1.35)
    context.config('airport_trips_missing_hotel', default='unresolved')
    context.config('airport_trips_mode_data', default='airports/geneva/geneva_trip_sampling.csv')
    context.config('airport_trips_folder', default='airports/trips')
    # These settings are shared with the airport producer stages.
    context.config('geneva_airport_pdf', default='airports/geneva/geneva_airport.pdf')
    context.config('geneva_airport_time_data', default=None)
    context.config('zurich_airport_folder', default='airports/zurich')


def execute(context):

    # Reuse existing municipality demand, raw census homes and located hotels.

    ods = {'GVA': context.stage('data.airports.geneva'), 'ZRH': context.stage('data.airports.zurich')}
    model = context.stage('data.airports.model')
    homes, hotels = context.stage('data.airports.locations')
    settings = trip_settings(context)
    sources = source_paths(context)
    shares, evidence = read_mode_shares(sources)
    token = input_signature(ods, homes, hotels, model, settings, sources)
    folder = resolve(Path(context.config('data_path')), context.config('airport_trips_folder'))
    cached = load_trips(folder, token)
    if cached is not None:
        return cached

    # Sample Swiss-endpoint movements, retaining unavailable accommodation for audit.

    trips, unresolved, summary = generate_trips(ods, homes, hotels, model, shares, settings, sources)
    summary.update(signature=token, mode_evidence=evidence, allocation_model_signature=model.signature)
    summary['location_inputs'] = {'home_locations': len(homes), 'hotel_establishments': len(hotels),
                                  'homes': homes.attrs, 'hotels': hotels.attrs}
    trips.attrs['summary'] = summary
    trips.attrs['unresolved_demand_file'] = str(folder / 'unresolved_demand.csv')

    # Save reproducible trips, rejected demand and the assumptions used to sample them.

    folder.mkdir(parents=True, exist_ok=True)
    atomic_write(folder / 'trips.parquet', lambda p: trips.to_parquet(p, index=False))
    atomic_write(folder / 'unresolved_demand.csv', lambda p: unresolved.to_csv(p, index=False))
    atomic_write(folder / 'summary.json', lambda p: p.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'))
    manifest = {'signature': token, 'artifacts': {name: file_digest(folder / name)
                for name in ('trips.parquet', 'unresolved_demand.csv', 'summary.json')}}
    atomic_write(folder / 'signature.json', lambda p: p.write_text(json.dumps(manifest) + '\n', encoding='utf-8'))
    logger.info('Airport trips: %d located passenger trips; %d unresolved', len(trips), int(unresolved.generated_trips.sum()))
    return trips


def validate(context):
    sources = source_paths(context)
    # Synpp tracks all census, spatial and OD dependencies separately.
    payload = {'settings': trip_settings(context), 'sources': {k: file_digest(v) for k, v in sources.items()},
               'code': [file_digest(Path(__file__)), file_digest(Path(common.__file__)),
                        file_digest(Path(geneva_times.__file__))]}
    folder = resolve(Path(context.config('data_path')), context.config('airport_trips_folder'))
    payload['exports'] = export_signature(output_paths(folder))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def output_paths(folder):
    return {name: folder / name for name in ('trips.parquet', 'unresolved_demand.csv', 'summary.json', 'signature.json')}


def trip_settings(context):
    seed = context.config('random_seed')
    rate = context.config('airport_trips_sample_rate')
    detour = context.config('airport_trips_detour_factor')
    speeds = context.config('airport_trips_speeds_kmh')
    fallback = context.config('airport_trips_missing_hotel')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('random_seed must be a non-negative integer')
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('airport_trips_sample_rate must be between zero and one')
    if isinstance(detour, bool) or not isinstance(detour, (int, float)) or not np.isfinite(detour) or detour < 1:
        raise ValueError('airport_trips_detour_factor must be finite and at least one')
    if not isinstance(speeds, dict) or set(speeds) != set(MODES) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                       not np.isfinite(v) or v <= 0 for v in speeds.values()):
        raise ValueError(f'airport_trips_speeds_kmh must give a positive speed for each of {MODES}')
    if fallback != 'unresolved':
        raise ValueError('airport_trips_missing_hotel must be unresolved: visitors use hotels only')
    return {'seed': seed, 'day': geneva_times.day_label(context.config('airport_trips_day')),
            'sample_rate': float(rate), 'detour_factor': float(detour),
            'speeds_kmh': dict(speeds), 'missing_hotel': fallback, 'version': VERSION}


def source_paths(context):
    root = Path(context.config('data_path'))
    modes = resolve(root, context.config('airport_trips_mode_data'))
    return {'modes': modes, 'mode_metadata': modes.with_suffix('.json'),
            'pdf': resolve(root, context.config('geneva_airport_pdf')),
            'GVA': geneva.time_data_path(context),
            'ZRH': zurich.input_folder(context) / 'data_from_websites/zurich_flight_times.json'}


def read_mode_shares(sources):
    metadata = json.loads(sources['mode_metadata'].read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1 or metadata['pdf_sha256'] != file_digest(sources['pdf']):
        raise ValueError('Review Geneva trip mode evidence after the airport PDF changes')
    if metadata['csv_sha256'] != file_digest(sources['modes']):
        raise ValueError('Review changed Geneva trip mode shares and update their metadata')
    table = pd.read_csv(sources['modes'])
    if not {'country', 'mode', 'percent', 'pdf_page'}.issubset(table.columns):
        raise ValueError('Mode evidence requires country, mode, percent and pdf_page columns')
    swiss = table.loc[table.country.eq('CH')]
    if swiss['mode'].duplicated().any() or set(swiss['mode']) != set(MODES):
        raise ValueError('Swiss Geneva mode shares require one observation per mode')
    shares = swiss.set_index('mode').loc[list(MODES), 'percent'].to_numpy(float)
    if not np.isfinite(shares).all() or (shares < 0).any() or (shares > 100).any():
        raise ValueError('Mode shares must be percentages between zero and 100')
    return normalized(shares), metadata


def hash_frame(frame):
    return pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()


def input_signature(ods, homes, hotels, model, settings, sources):
    payload = {'settings': settings, 'allocation_model': model.signature, 'airports': model.airports,
               'sources': {name: file_digest(path) for name, path in sorted(sources.items())}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    for frame in [ods['GVA'], ods['ZRH'], homes, hotels]:
        digest.update(hash_frame(frame))
    for path in [Path(__file__), Path(common.__file__), Path(geneva_times.__file__)]:
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def expand_demand(ods, model, settings, rng):
    output = []
    for airport in ('GVA', 'ZRH'):
        frame = ods[airport].copy().reset_index(drop=True)
        # Filter before expansion or rounding; foreign access is outside model scope.
        if 'country' not in frame:
            raise ValueError(f'{airport} ODs missing country column')
        frame['od_id'] = [f'{airport}:{i}' for i in range(len(frame))]
        frame = frame.loc[frame.country.eq('CH')].copy()
        validate_demand(frame, airport, model)
        frame['airport'] = airport
        if 'profile' not in frame:
            frame['profile'] = 'pooled'
        # Apply any explicit sampling rate while retaining the rounded airport total.
        counts = frame.daily_trips_integer.to_numpy(dtype=np.int64)
        if settings['sample_rate'] != 1:
            counts = apportion(counts * settings['sample_rate'])
        selected = np.repeat(np.arange(len(frame)), counts)
        frame = frame.iloc[selected].reset_index(drop=True)
        frame['trip_id'] = [f'airport_{airport}_{i}' for i in range(len(frame))]
        frame['traveller_type'] = frame.profile.map({'outgoing': 'resident', 'incoming_other': 'visitor', 'pooled': 'unknown'})
        frame['visitor_probability'] = frame.profile.map({'outgoing': 0., 'incoming_other': 1.})
        frame['traveller_type_source'] = 'Geneva observed survey profile'
        if frame.traveller_type.isna().any():
            raise ValueError('Unknown airport passenger profile')
        if airport == 'ZRH':
            # Condition the transferred profile mixture on the municipality already allocated.
            # A global coin flip would erase the calibrated resident/visitor geography.
            pooled = model.probabilities(airport)
            probability = np.divide(model.swiss_visitor_share * model.probabilities(airport, 'incoming_other'),
                                    pooled, out=np.zeros_like(pooled), where=pooled > 0)
            lookup = pd.Series(probability, index=model.municipalities.municipality_id)
            swiss = frame.country.eq('CH') & frame.municipality_id.notna()
            frame.loc[swiss, 'visitor_probability'] = frame.loc[swiss, 'municipality_id'].map(lookup)
            visitor = rng.random(int(swiss.sum())) < frame.loc[swiss, 'visitor_probability'].to_numpy(float)
            frame.loc[swiss, 'traveller_type'] = np.where(visitor, 'visitor', 'resident')
            frame['traveller_type_source'] = 'Geneva profile mixture transferred to Zurich, conditional on municipality'
        # Visitor tag describes the journey/accommodation profile, not citizenship.
        frame['is_visitor'] = frame.traveller_type.map({'resident': False, 'visitor': True}).astype('boolean')
        output.append(frame)
    columns = ['trip_id', 'od_id', 'airport', 'direction', 'region_id', 'country', 'municipality_id',
               'profile', 'traveller_type', 'origin_region', 'destination_region',
               'is_visitor', 'visitor_probability', 'traveller_type_source', 'traffic_year', 'month']
    result = pd.concat(output, ignore_index=True)
    return result[[c for c in columns if c in result]].copy()


def validate_demand(frame, airport, model):
    required = {'country', 'direction', 'municipality_id', 'region_id', 'origin_region',
                'destination_region', 'daily_trips_integer', 'daily_trips'}
    if not required.issubset(frame):
        raise ValueError(f'{airport} ODs missing columns: {sorted(required - set(frame.columns))}')
    if frame[list(required - {'municipality_id'})].isna().any().any():
        raise ValueError(f'{airport} ODs contain missing demand fields')
    counts = frame[['daily_trips', 'daily_trips_integer']].to_numpy(dtype=float)
    if (not np.isfinite(counts).all() or (counts < 0).any() or
            not np.equal(counts[:, 1], np.floor(counts[:, 1])).all()):
        raise ValueError(f'{airport} OD demand must be non-negative with integer trip counts')
    if not frame.direction.isin(['to_airport', 'from_airport']).all():
        raise ValueError(f'{airport} ODs contain unknown ground directions')
    if airport == 'GVA' and ('profile' not in frame or not frame.profile.isin(['outgoing', 'incoming_other']).all()):
        raise ValueError('Geneva ODs require observed outgoing/incoming_other profiles')
    swiss = frame.country.eq('CH') & frame.municipality_id.notna()
    if not frame.loc[swiss, 'municipality_id'].isin(model.municipalities.municipality_id).all():
        raise ValueError(f'{airport} Swiss OD municipality is absent from the allocation model')


def sample_endpoints(frame, homes, hotels, rng):
    frame = frame.copy()
    frame['endpoint_x'] = np.nan
    frame['endpoint_y'] = np.nan
    frame['location_source'] = pd.Series('', index=frame.index, dtype='str')
    frame['unresolved_reason'] = np.where(frame.country.ne('CH'), 'foreign_region_without_municipality', '')
    home_groups = {int(k): g for k, g in homes.groupby('municipality_id', sort=False)}
    hotel_groups = {int(k): g for k, g in hotels.groupby('municipality_id', sort=False)}
    swiss = frame.country.eq('CH')
    frame.loc[swiss & frame.municipality_id.isna(), 'unresolved_reason'] = 'missing_municipality'
    for (municipality, kind), rows in frame.loc[swiss & frame.municipality_id.notna()].groupby(['municipality_id', 'traveller_type'], sort=True):
        source = 'hotel' if kind == 'visitor' else 'home'
        pool = (hotel_groups if kind == 'visitor' else home_groups).get(int(municipality))
        if pool is None or pool.empty:
            frame.loc[rows.index, 'unresolved_reason'] = 'no_hotel_in_municipality' if kind == 'visitor' else 'no_valid_home_in_municipality'
            continue
        choices = rng.choice(len(pool), len(rows), p=normalized(pool.weight))
        frame.loc[rows.index, ['endpoint_x', 'endpoint_y']] = pool.iloc[choices][['x', 'y']].to_numpy()
        frame.loc[rows.index, 'location_source'] = source
    return frame


def generate_trips(ods, homes, hotels, model, shares, settings, sources):
    coverage = {}
    for airport in ('GVA', 'ZRH'):
        snapshot = geneva_times.load_snapshot(sources[airport])
        coverage[airport] = {direction: geneva_times.timing_coverage(snapshot, settings['day'], direction)
                             for direction in ('to_airport', 'from_airport')}
    streams = np.random.SeedSequence(settings['seed']).spawn(4)
    frame = expand_demand(ods, model, settings, np.random.default_rng(streams[0]))
    frame = sample_endpoints(frame, homes, hotels, np.random.default_rng(streams[1]))
    rejected = frame.loc[frame.unresolved_reason.ne('')].copy()
    keys = ['airport', 'direction', 'region_id', 'country', 'municipality_id', 'profile', 'traveller_type', 'is_visitor', 'unresolved_reason']
    unresolved = rejected.groupby(keys, dropna=False).size().rename('generated_trips').reset_index()
    frame = frame.loc[frame.unresolved_reason.eq('')].drop(columns='unresolved_reason').reset_index(drop=True)
    frame['access_mode'] = np.random.default_rng(streams[2]).choice(MODES, len(frame), p=shares)
    frame['mode'] = frame.access_mode.map({m: ('pt' if m in ('rail', 'bus_tram_aerobus') else 'other' if m == 'other' else 'car') for m in MODES})
    frame['mode_share_source'] = 'Geneva Swiss-endpoint survey; see summary.mode_evidence'
    airport_xy = {a: Transformer.from_crs(4326, 2056, always_xy=True).transform(c['longitude'], c['latitude']) for a, c in model.airports.items()}
    ax = frame.airport.map({a: xy[0] for a, xy in airport_xy.items()}).to_numpy()
    ay = frame.airport.map({a: xy[1] for a, xy in airport_xy.items()}).to_numpy()
    to = frame.direction.eq('to_airport')
    frame['origin_x'] = np.where(to, frame.endpoint_x, ax)
    frame['origin_y'] = np.where(to, frame.endpoint_y, ay)
    frame['destination_x'] = np.where(to, ax, frame.endpoint_x)
    frame['destination_y'] = np.where(to, ay, frame.endpoint_y)
    frame['origin_municipality_id'] = frame.municipality_id.where(to).astype('Int64')
    frame['destination_municipality_id'] = frame.municipality_id.where(~to).astype('Int64')
    frame['euclidean_distance_m'] = np.hypot(frame.destination_x - frame.origin_x, frame.destination_y - frame.origin_y)
    frame['distance_m'] = frame.euclidean_distance_m * settings['detour_factor']
    frame['speed_kmh'] = frame.access_mode.map(settings['speeds_kmh'])
    frame['travel_time_seconds'] = frame.distance_m / (frame.speed_kmh / 3.6)
    timing = []
    seeds = streams[3].spawn(4)
    for i, (airport, direction) in enumerate((a, d) for a in ('GVA', 'ZRH') for d in ('to_airport', 'from_airport')):
        rows = frame.loc[frame.airport.eq(airport) & frame.direction.eq(direction)]
        # Validate requested-day coverage even for a zero-demand airport/direction.
        sampled = geneva_times.sample_departure_times(direction, len(rows), seed=seeds[i], day=settings['day'],
                    travel_time_seconds=rows.travel_time_seconds.to_numpy(), snapshot_path=sources[airport])
        sampled.index = rows.index
        timing.append(sampled.drop(columns='direction'))
    times = pd.concat(timing).reindex(frame.index)
    frame = pd.concat([frame, times], axis=1)
    frame['departure_time'] = frame.departure_seconds
    frame['arrival_time'] = frame.departure_seconds + frame.travel_time_seconds
    frame['sample_weight'] = 1. / settings['sample_rate'] if settings['sample_rate'] else 0.
    frame = frame.drop(columns=['endpoint_x', 'endpoint_y']).convert_dtypes()
    summary = {'settings': settings, 'crs': 'EPSG:2056', 'time_unit': 'local wall-clock seconds relative to flight day; no midnight wrapping',
               'scope': 'Swiss ground endpoints only; foreign ground endpoints excluded before trip generation',
               'is_visitor_definition': 'Journey profile, not passport nationality: true=visitor/hotel, false=resident/household',
               'generated_trips': len(frame), 'unresolved_trips': len(rejected),
               'by_airport': {}, 'assumptions': [
                   'Swiss/foreign endpoint country is separate from resident/visitor profile; nationality is unavailable.',
                   'Geneva outgoing profiles use homes; incoming_other profiles use hotels where recorded.',
                   'Zurich resident/visitor probabilities transfer Geneva profiles, conditional on each allocated municipality.',
                   'Both airports use Geneva Swiss-endpoint mode shares (page 21); modes are independent of direction, profile and municipality.',
                   'Home coordinates are sampled proportional to permanent main-residence population at each coordinate; hotels are uniform establishments, without room counts.',
                   'Residents use homes and visitors use hotels only; missing accommodation remains unresolved in the same municipality.',
                   'Speeds and detour factor are configurable assumptions; no network routing or PT timetable is used.',
                   'Timing uses archived operating flights, equal weights per calendar date and assumed check-in/clearance buffers.',
                   'Selected day changes timing only; demand remains the airport stage annual/monthly average day.',
                   'Foreign ground endpoints are out of scope; only their excluded aggregate counts appear in the summary.',
                   'Passenger movements only: no escort return legs, vehicle occupancy conversion, employees or linked round trips.']}
    for airport in ('GVA', 'ZRH'):
        rows = frame.loc[frame.airport.eq(airport)]
        source = ods[airport]
        domestic = source.loc[source.country.eq('CH')]
        foreign = source.loc[source.country.ne('CH')]
        if airport == 'GVA':
            total = domestic.daily_trips.sum()
            visitor_share = (float(domestic.loc[domestic.profile.eq('incoming_other'), 'daily_trips'].sum() / total)
                             if total > 0 else None)
            share_source = 'Geneva supplied 2025 survey pp. 4-12: raked region/profile margins, conditioned on Swiss ground endpoints'
        else:
            visitor_share = float(model.swiss_visitor_share)
            share_source = 'Saved Geneva Swiss resident/visitor mixture transferred to Zurich; not a Zurich observation'
        summary['by_airport'][airport] = {'input_daily_trips': float(ods[airport].daily_trips.sum()),
            'input_daily_trips_integer': int(ods[airport].daily_trips_integer.sum()),
            'in_scope_daily_trips': float(domestic.daily_trips.sum()),
            'in_scope_daily_trips_integer': int(domestic.daily_trips_integer.sum()),
            'excluded_foreign_daily_trips': float(foreign.daily_trips.sum()),
            'excluded_foreign_daily_trips_integer': int(foreign.daily_trips_integer.sum()),
            'resident_profile_share': None if visitor_share is None else 1 - visitor_share,
            'visitor_profile_share': visitor_share, 'profile_share_source': share_source,
            'generated_trips': len(rows), 'unresolved_trips': int(rejected.airport.eq(airport).sum()),
            'traveller_types': {k: int(v) for k, v in rows.traveller_type.value_counts().items()},
            'access_modes': {k: int(v) for k, v in rows.access_mode.value_counts().items()},
            'location_sources': {k: int(v) for k, v in rows.location_source.value_counts().items()},
            'timing_source': str(sources[airport]), 'timing_coverage': coverage[airport]}
        expected = int(np.floor(domestic.daily_trips_integer.sum() * settings['sample_rate'] + .5))
        generated = len(rows) + int(rejected.airport.eq(airport).sum())
        if generated != expected:
            raise ValueError(f'{airport} trip generation changed the source passenger total')
        summary['by_airport'][airport]['sampled_trips_total'] = generated
        summary['by_airport'][airport]['unresolved_reasons'] = {
            k: int(v) for k, v in rejected.loc[rejected.airport.eq(airport), 'unresolved_reason'].value_counts().items()}
    return frame, unresolved, summary


def load_trips(folder, token):
    try:
        metadata = json.loads((folder / 'signature.json').read_text(encoding='utf-8'))
        if metadata['signature'] != token:
            return None
        if any(file_digest(folder / name) != metadata['artifacts'][name]
               for name in ('trips.parquet', 'unresolved_demand.csv', 'summary.json')):
            return None
        frame = pd.read_parquet(folder / 'trips.parquet')
        frame.attrs['summary'] = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        frame.attrs['unresolved_demand_file'] = str(folder / 'unresolved_demand.csv')
        return frame
    except (OSError, ValueError, KeyError, TypeError):
        return None
