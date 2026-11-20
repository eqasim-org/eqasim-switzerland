"""Sample daily Geneva/Zurich passenger access trips from municipality ODs."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from data.airports._shared import common
from synthesis.airports import modes as mode_allocation
from data.airports._shared import access
from synthesis.airports import residents
from data.airports.geneva import ods as geneva
from data.airports.geneva import times as geneva_times
from data.airports import ods as airport_ods
from data.airports.zurich import ods as zurich
from data.airports._shared.common import apportion, atomic_write, export_signature, file_digest, normalized
from data.airports.geneva.model import resolve

logger = logging.getLogger('synpp')
MODES = ('dropoff', 'private_car', 'rail', 'bus_tram_aerobus', 'taxi', 'ridehail', 'other')
# Preserve the observed access mode; collapse it only for the simulation output.
# The report does not break down "other": passenger travel is a modelling fallback.
# Drop-off, taxi and ride-hail travellers take a car and generate road traffic.
# With no separate taxi/ride-hail simulation modes, represent these trips as car.
MODE_MAP = {
    'dropoff': 'car',
    'private_car': 'car',
    'rail': 'pt',
    'bus_tram_aerobus': 'pt',
    'taxi': 'car',
    'ridehail': 'car',
    'other': 'car_passenger',
}
# All road categories use the same effective speed, including car passengers.
SPEEDS = dict(zip(MODES, (60., 60., 70., 30., 60., 60., 60.)))
VERSION = 12


def configure(context):
    context.stage('data.airports.ods')
    context.stage('data.airports.geneva.model')
    context.stage('synthesis.airports.locations')
    context.config('data_path')
    context.config('random_seed')
    context.config('airport_trips_day', default='all')
    context.config('airport_trips_sample_rate', default=None)
    context.config('airport_trips_speeds_kmh', default=SPEEDS)
    context.config('airport_trips_detour_factor', default=1.35)
    context.config('airport_trips_missing_hotel', default='unresolved')
    context.config('airport_trips_mode_data', default='airports/geneva/geneva_trip_sampling.csv')
    context.config('airport_trips_regional_mode_data', default='airports/geneva/geneva_regional_mode_shares.csv')
    context.config('airport_trips_folder', default='airports/trips')
    context.config('airport_trips_access_points', default=access.DEFAULT_FILES)
    # These settings are shared with the airport producer stages.
    context.config('geneva_airport_pdf', default='airports/geneva/geneva_airport.pdf')
    context.config('geneva_airport_time_data', default=None)
    context.config('zurich_airport_folder', default='airports/zurich')


def execute(context):

    # Reuse already scaled municipality demand and the checked location pools.

    ods = airport_ods.split_airports(context.stage('data.airports.ods'))
    model = context.stage('data.airports.geneva.model')
    homes, hotels = context.stage('synthesis.airports.locations')
    settings = trip_settings(context)
    sources = source_paths(context)
    shares, evidence = read_mode_shares(sources)
    token = input_signature(ods, homes, hotels, model, settings, sources)
    folder = resolve(Path(context.config('data_path')), context.config('airport_trips_folder'))
    cached = load_trips(folder, token)
    if cached is not None:
        return cached

    # Sample Swiss-endpoint movements, retaining unavailable accommodation for audit.

    distance_model = mode_allocation.read_distance_model(sources, model, ods['GVA'], MODES)
    trips, unresolved, summary = generate_trips(ods, homes, hotels, model, shares, settings, sources, distance_model)
    summary.update(signature=token, mode_evidence=evidence, allocation_model_signature=model.signature)
    kinds = hotels.get('accommodation_type', pd.Series('hotel', index=hotels.index))
    summary['location_inputs'] = {'resident_candidates': len(homes), 'distinct_resident_homes': len(homes[['x','y']].drop_duplicates()), 'accommodation_establishments': len(hotels),
                                  'hotel_establishments': int(kinds.eq('hotel').sum()),
                                  'dormitory_establishments': int(kinds.eq('dormitory').sum()),
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
                        file_digest(Path(geneva_times.__file__)), file_digest(Path(mode_allocation.__file__)),
                        file_digest(Path(access.__file__)), file_digest(Path(residents.__file__))]}
    folder = resolve(Path(context.config('data_path')), context.config('airport_trips_folder'))
    payload['exports'] = export_signature(output_paths(folder))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def output_paths(folder):
    return {name: folder / name for name in ('trips.parquet', 'unresolved_demand.csv', 'summary.json', 'signature.json')}


def trip_settings(context):
    seed = context.config('random_seed')
    detour = context.config('airport_trips_detour_factor')
    speeds = common.config_mapping(context, 'airport_trips_speeds_kmh')
    fallback = context.config('airport_trips_missing_hotel')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('random_seed must be a non-negative integer')
    if context.config('airport_trips_sample_rate') is not None:
        raise ValueError('Remove airport_trips_sample_rate; data.airports.ods now uses input_downsampling')
    if isinstance(detour, bool) or not isinstance(detour, (int, float)) or not np.isfinite(detour) or detour < 1:
        raise ValueError('airport_trips_detour_factor must be finite and at least one')
    if not isinstance(speeds, dict) or set(speeds) != set(MODES) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                                       not np.isfinite(v) or v <= 0 for v in speeds.values()):
        raise ValueError(f'airport_trips_speeds_kmh must give a positive speed for each of {MODES}')
    if fallback != 'unresolved':
        raise ValueError('airport_trips_missing_hotel must be unresolved: visitors use accommodation only')
    return {'seed': seed, 'day': geneva_times.day_label(context.config('airport_trips_day')),
            'detour_factor': float(detour),
            'speeds_kmh': dict(speeds), 'missing_hotel': fallback, 'version': VERSION}


def source_paths(context):
    root = Path(context.config('data_path'))
    modes = resolve(root, context.config('airport_trips_mode_data'))
    regional = resolve(root, context.config('airport_trips_regional_mode_data'))
    access_files = common.config_mapping(context, 'airport_trips_access_points')
    if not isinstance(access_files, dict) or set(access_files) != {'GVA', 'ZRH'}:
        raise ValueError('airport_trips_access_points must give GVA and ZRH JSON paths')
    return {'modes': modes, 'mode_metadata': modes.with_suffix('.json'),
            **{f'{a}_access': resolve(root, path) for a, path in access_files.items()},
            'regional_modes': regional, 'regional_mode_metadata': regional.with_suffix('.json'),
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
               'scaled_signatures': {a: ods[a].attrs.get('signature') for a in ('GVA', 'ZRH')},
               'resident_inputs': dict(homes.attrs),
               'sources': {name: file_digest(path) for name, path in sorted(sources.items())}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    for frame in [ods['GVA'], ods['ZRH'], homes, hotels]:
        digest.update(hash_frame(frame))
    for path in [Path(__file__), Path(common.__file__), Path(geneva_times.__file__), Path(mode_allocation.__file__), Path(access.__file__), Path(residents.__file__)]:
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def expand_demand(ods, model, settings, rng, hotels):
    output = []
    assignments = {}
    for airport in ('GVA', 'ZRH'):
        frame = ods[airport].copy().reset_index(drop=True)
        validate_demand(frame, airport, model)
        frame['airport'] = airport
        if 'profile' not in frame:
            frame['profile'] = 'pooled'
        # Expand each count exactly once: the upstream stage has done all scaling.
        counts = frame.daily_trips_integer.to_numpy(dtype=np.int64)
        selected = np.repeat(np.arange(len(frame)), counts)
        frame = frame.iloc[selected].reset_index(drop=True)
        frame['trip_id'] = [f'airport_{airport}_{i}' for i in range(len(frame))]
        frame['source_profile'] = frame.profile
        # Keep modelled municipal demand fixed, then assign the visitor population.
        source = ods[airport]
        total = float(source.daily_trips.sum())
        share = (float(source.loc[source.profile.eq('incoming_other'), 'daily_trips'].sum()) / total
                 if airport == 'GVA' and total > 0 else model.swiss_visitor_share)
        frame, assignments[airport] = assign_visitors(frame, hotels, share, rng)
        output.append(frame)
    columns = ['trip_id', 'od_id', 'airport', 'direction', 'region_id', 'country', 'municipality_id',
               'profile', 'traveller_type', 'origin_region', 'destination_region',
               'is_visitor', 'visitor_probability', 'traveller_type_source', 'traffic_year', 'month',
               'sampling_rate', 'sample_weight', 'source_profile', 'municipality_hotels', 'municipality_accommodations']
    result = pd.concat(output, ignore_index=True)
    result = result[[c for c in columns if c in result]].copy()
    result.attrs['visitor_assignment'] = assignments
    return result


def assign_visitors(frame, hotels, share, rng):
    """Allocate visitors by hotel+dormitory count, bounded by municipal trip demand."""
    frame = frame.copy()
    hotel_counts = hotels.groupby('municipality_id').size()
    frame['municipality_hotels'] = frame.municipality_id.map(hotel_counts).fillna(0).astype(int)
    frame['municipality_accommodations'] = frame['municipality_hotels']
    frame['is_visitor'] = False
    frame['visitor_probability'] = 0.
    targets, assigned = 0, 0
    directions = list(frame.groupby('direction', sort=True))
    direction_targets = apportion(np.array([len(rows) for _, rows in directions]) * share)
    for (_, rows), target in zip(directions, direction_targets):
        groups = list(rows.groupby('municipality_id', sort=True).groups.values())
        capacity = np.array([len(indices) for indices in groups], dtype=int)
        weights = np.array([int(frame.loc[indices[0], 'municipality_hotels']) for indices in groups], dtype=float)
        quotas = hotel_quotas(capacity, weights, target)
        for indices, count in zip(groups, quotas):
            frame.loc[indices, 'visitor_probability'] = float(count / len(indices))
            if count:
                frame.loc[rng.choice(np.asarray(indices), count, replace=False), 'is_visitor'] = True
        targets += int(target)
        assigned += int(quotas.sum())
    frame['traveller_type'] = np.where(frame.is_visitor, 'visitor', 'resident')
    frame['profile'] = np.where(frame.is_visitor, 'incoming_other', 'outgoing')
    frame['traveller_type_source'] = 'Airport visitor share allocated by municipal hotel+dormitory establishment counts; no accommodation means resident'
    return frame, {'method': 'accommodation_counts_with_municipal_trip_capacity',
                   'target_visitor_share': float(share), 'target_visitor_trips': targets,
                   'assigned_visitor_trips': assigned, 'unfilled_visitor_target': targets - assigned}


def hotel_quotas(capacity, hotels, target):
    """Redistribute hotel-weighted quotas when a municipality reaches its OD capacity."""
    capacity, hotels = np.asarray(capacity, dtype=int), np.asarray(hotels, dtype=float)
    quota = np.zeros(len(capacity), dtype=float)
    active = np.asarray(hotels) > 0
    remaining = min(int(target), int(np.asarray(capacity)[active].sum()))
    while remaining > 0 and active.any():
        indices = np.flatnonzero(active)
        proposed = remaining * hotels[indices] / hotels[indices].sum()
        capped = proposed >= capacity[indices]
        if not capped.any():
            quota[indices] = proposed
            break
        filled = indices[capped]
        quota[filled] = capacity[filled]
        remaining -= int(capacity[filled].sum())
        active[filled] = False
    return apportion(np.minimum(quota, capacity))


def validate_demand(frame, airport, model):
    required = {'country', 'direction', 'municipality_id', 'region_id', 'origin_region',
                'destination_region', 'daily_trips_integer', 'daily_trips',
                'od_id', 'sampling_rate', 'sample_weight', 'full_population_daily_trips'}
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
    if not frame.country.eq('CH').all():
        raise ValueError('Trip generation requires Swiss-endpoint ODs from data.airports.ods')
    if airport == 'GVA' and ('profile' not in frame or not frame.profile.isin(['outgoing', 'incoming_other']).all()):
        raise ValueError('Geneva ODs require observed outgoing/incoming_other profiles')
    swiss = frame.country.eq('CH') & frame.municipality_id.notna()
    if not frame.loc[swiss, 'municipality_id'].isin(model.municipalities.municipality_id).all():
        raise ValueError(f'{airport} Swiss OD municipality is absent from the allocation model')


def sample_endpoints(frame, homes, hotels, rng):
    """Retain each selected resident's home and sample only visitor accommodation."""
    frame = frame.copy()
    if not {'person_id','endpoint_x','endpoint_y','endpoint_ovgk','unresolved_reason'}.issubset(frame):
        raise ValueError('Select synthetic residents before sampling visitor endpoints')
    hotel_groups = {int(k): g for k,g in hotels.groupby('municipality_id',sort=False)}
    visitors = frame.loc[frame.is_visitor]
    for (municipality, mode), rows in visitors.groupby(['municipality_id','mode'],sort=True):
        pool = hotel_groups.get(int(municipality))
        if pool is None or pool.empty:
            frame.loc[rows.index,'unresolved_reason'] = 'no_accommodation_in_municipality'
            continue
        if mode == 'pt':
            tier = rows.pt_access_tier.iloc[0]
            pool = pool.loc[pool.ovgk.eq('D') if tier == 'D' else pool.ovgk.isin(['A','B','C'])]
            if pool.empty:
                frame.loc[rows.index,'unresolved_reason'] = 'no_pt_accessible_accommodation_in_municipality'
                continue
        selected = pool.iloc[rng.choice(len(pool),len(rows),p=normalized(pool.weight))]
        frame.loc[rows.index,['endpoint_x','endpoint_y']] = selected[['x','y']].to_numpy()
        frame.loc[rows.index,'endpoint_ovgk'] = selected.ovgk.astype(str).to_numpy()
        frame.loc[rows.index,'location_source'] = (selected.accommodation_type.to_numpy()
            if 'accommodation_type' in selected else 'hotel')
    return frame


def generate_trips(ods, homes, hotels, model, shares, settings, sources, distance_model=None):
    access_points, access_evidence = access.read_access_points(sources)
    coverage = {}
    for airport in ('GVA', 'ZRH'):
        snapshot = geneva_times.load_snapshot(sources[airport])
        coverage[airport] = {direction: geneva_times.timing_coverage(snapshot, settings['day'], direction)
                             for direction in ('to_airport', 'from_airport')}
    streams = np.random.SeedSequence(settings['seed']).spawn(4)
    frame = expand_demand(ods, model, settings, np.random.default_rng(streams[0]), hotels)
    visitor_assignment = frame.attrs['visitor_assignment']
    # Select unique resident agents using pooled survey demographic preferences.
    location_streams = streams[1].spawn(2)
    frame, resident_summary = residents.select_residents(frame, homes, np.random.default_rng(location_streams[0]))

    # Allocate mode totals using each selected resident's actual home accessibility.
    frame, mode_diagnostics = mode_allocation.assign_modes(frame, homes, hotels, model, shares,
        MODES, MODE_MAP, np.random.default_rng(streams[2]), distance_model)
    frame = sample_endpoints(frame, homes, hotels, np.random.default_rng(location_streams[1]))
    rejected = frame.loc[frame.unresolved_reason.ne('')].copy()
    keys = ['airport', 'direction', 'region_id', 'country', 'municipality_id', 'profile', 'traveller_type', 'is_visitor', 'access_mode', 'mode', 'unresolved_reason']
    unresolved = rejected.groupby(keys, dropna=False).size().rename('generated_trips').reset_index()
    frame = frame.loc[frame.unresolved_reason.eq('')].drop(columns='unresolved_reason').reset_index(drop=True)
    frame['mode_share_source'] = 'Geneva p21 Swiss mode targets + p23 distance preferences; see summary.mode_evidence and mode_distance_model'
    airport_xy = {f'{a}:{kind}': Transformer.from_crs(4326, 2056, always_xy=True).transform(p['longitude'], p['latitude'])
                  for a, points in access_points.items() for kind, p in points.items()}
    frame['airport_access_point'] = np.where(frame['mode'].eq('pt'), 'pt', 'road')
    key = frame.airport + ':' + frame.airport_access_point
    frame['airport_x'] = key.map({k: xy[0] for k, xy in airport_xy.items()})
    frame['airport_y'] = key.map({k: xy[1] for k, xy in airport_xy.items()})
    # Use the facilities' whole-metre convention before deriving either endpoint,
    # distances or travel times. Both XML writers then receive identical points.
    coordinates = ['airport_x', 'airport_y', 'endpoint_x', 'endpoint_y']
    frame[coordinates] = np.trunc(frame[coordinates].to_numpy(dtype=float)).astype(np.int64)
    ax, ay = frame.airport_x.to_numpy(), frame.airport_y.to_numpy()
    to = frame.direction.eq('to_airport')
    # Visitors stay at temporary accommodation rather than resident households.
    lodging = np.where(frame.is_visitor, 'home_secondary', 'home')
    frame['origin_purpose'] = np.where(to, lodging, 'airport')
    frame['destination_purpose'] = np.where(to, 'airport', lodging)
    frame['activity_chain'] = frame.origin_purpose + '-' + frame.destination_purpose
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
    frame = frame.drop(columns=['endpoint_x', 'endpoint_y']).convert_dtypes()
    summary = {'settings': settings, 'crs': 'EPSG:2056', 'time_unit': 'local wall-clock seconds relative to flight day; no midnight wrapping',
               'scope': 'Swiss ground endpoints only; foreign ground endpoints excluded before trip generation',
               'is_visitor_definition': 'Journey profile, not passport nationality: true=visitor/hotel or dormitory, false=resident/household',
               'mode_map': dict(MODE_MAP),
               'airport_access_points': access_evidence,
               'mode_distance_model': distance_model,
               'resident_sampling': resident_summary,
               'generated_trips': len(frame), 'unresolved_trips': len(rejected),
               'by_airport': {}, 'assumptions': [
                   'Swiss/foreign endpoint country is separate from resident/visitor profile; nationality is unavailable.',
                   'Municipal OD counts are fixed before visitor assignment; visitors are weighted by hotel+dormitory counts and capped by municipal trip capacity.',
                   'Geneva supplies the airport visitor share, transferred to Zurich; municipalities without usable hotels contain residents only.',
                   'Both airports target Geneva Swiss-endpoint mode totals (page 21), allocated using linear distance preferences fitted to three Geneva regional groups (page 23).',
                   'Only direct home/airport chains for residents and home_secondary/airport chains for visitors. No intermediate activities or later outings.',
                   'Dropoff, taxi and ridehail use car to represent their road traffic because separate taxi/ridehail simulation modes are unavailable; other uses car_passenger as an explicit fallback.',
                   'Residents are selected without replacement from synthesis.population.sampled in their fixed municipality, using pooled age/sex/employment preferences from the 2015/2021 airport microcensus. Their actual home coordinates and person IDs are retained.',
                   'Visitors use uniformly weighted NOGA 551 hotels and 559 accommodation/dormitories, including student residences; visitor person IDs stay null.',
                   'Resident demographic selection precedes mode allocation; PT support is evaluated at selected agents homes. Missing resident capacity remains unresolved rather than reusing or relocating agents.',
                   'Residents use homes and visitors use hotels/dormitories; missing accommodation remains unresolved in the same municipality.',
                   'PT quotas use A/B/C first and D only if necessary. Remaining PT shortfalls transfer to car/car-passenger in proportion to their shares; all municipal counts stay fixed.',
                   'Physical airport endpoints depend on mode: station/interchange centroid for PT, parking/drop-off point for car/car-passenger. Gravity reference points remain separate.',
                   'Speeds and detour factor are configurable assumptions; no network routing or PT timetable is used.',
                   'Timing uses archived operating flights, equal weights per calendar date and assumed check-in/clearance buffers.',
                   'Selected day changes timing only; demand remains the airport stage annual/monthly average day.',
                   'Foreign ground endpoints are out of scope; only their excluded aggregate counts appear in the summary.',
                   'Passenger movements only: no escort return legs, vehicle occupancy conversion, employees or linked round trips.']}
    for airport in ('GVA', 'ZRH'):
        rows = frame.loc[frame.airport.eq(airport)]
        source = ods[airport]
        domestic = source.loc[source.country.eq('CH')]
        scaling = source.attrs.get('summary', {})
        source_totals = scaling.get('by_airport', {}).get(airport, {})
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
            'full_population_daily_trips': float(source.full_population_daily_trips.sum()),
            'sampling_rate': scaling.get('sampling_rate'),
            'excluded_foreign_daily_trips': source_totals.get('excluded_foreign_daily_trips', 0.),
            'excluded_foreign_daily_trips_integer': source_totals.get('excluded_foreign_daily_trips_integer', 0),
            'resident_profile_share': None if visitor_share is None else 1 - visitor_share,
            'visitor_profile_share': visitor_share, 'profile_share_source': share_source,
            'visitor_assignment': visitor_assignment[airport],
            'generated_trips': len(rows), 'unresolved_trips': int(rejected.airport.eq(airport).sum()),
            'traveller_types': {k: int(v) for k, v in rows.traveller_type.value_counts().items()},
            'access_modes': {k: int(v) for k, v in rows.access_mode.value_counts().items()},
            'simulation_modes': {k: int(v) for k, v in rows['mode'].value_counts().items()},
            'mode_assignment': mode_diagnostics[airport],
            'location_sources': {k: int(v) for k, v in rows.location_source.value_counts().items()},
            'timing_source': str(sources[airport]), 'timing_coverage': coverage[airport]}
        expected = int(domestic.daily_trips_integer.sum())
        generated = len(rows) + int(rejected.airport.eq(airport).sum())
        if generated != expected:
            raise ValueError(f'{airport} trip generation changed the source passenger total')
        summary['by_airport'][airport]['sampled_trips_total'] = generated
        summary['by_airport'][airport]['unresolved_reasons'] = {
            k: int(v) for k, v in rejected.loc[rejected.airport.eq(airport), 'unresolved_reason'].value_counts().items()}
    return frame, unresolved, summary


def inject_airport_trips_in_population(population, trips, *, reserved_vehicle_ids=()):
    """Replace selected residents' plans and add visitors as a minimal population.

    Return a new ``synthesis.prepared.tables.Population``; all inputs stay read-only.
    Each ground movement is one agent with a two-activity plan, as for freight/LCV.
    Residents retain their sampled person ID. Visitors receive deterministic IDs
    checked against every prepared person, freight/LCV agent and airport resident.
    """
    from dataclasses import replace

    # Validate the small contract consumed by the population and resource writers.

    columns = ['trip_id', 'person_id', 'is_visitor', 'airport', 'direction', 'mode',
               'origin_x', 'origin_y', 'destination_x', 'destination_y',
               'departure_time', 'arrival_time']
    missing = set(columns) - set(trips.columns)
    if missing:
        raise ValueError(f'Airport injection is missing columns: {sorted(missing)}')
    if population.airport_trips is not None:
        raise ValueError('Airport trips have already been injected')
    frame = trips[columns].copy().reset_index(drop=True)
    if frame.trip_id.isna().any() or frame.trip_id.astype(str).duplicated().any():
        raise ValueError('Airport trip IDs must be unique and non-missing')
    if frame.is_visitor.isna().any() or not frame.is_visitor.isin([True, False]).all():
        raise ValueError('Airport is_visitor must contain booleans')
    frame['is_visitor'] = frame.is_visitor.astype(bool)
    if not frame.direction.isin(['to_airport', 'from_airport']).all():
        raise ValueError('Unknown airport trip direction')
    if not frame['mode'].isin(['car', 'car_passenger', 'pt', 'walk', 'bike']).all():
        raise ValueError('Airport modes must use the five simulation modes')
    coordinates = ['origin_x', 'origin_y', 'destination_x', 'destination_y']
    numeric = frame[coordinates + ['departure_time', 'arrival_time']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (frame.arrival_time < frame.departure_time).any():
        raise ValueError('Airport coordinates/times must be finite and arrival cannot precede departure')
    # Normalize older/imported trip tables too, before population and facilities
    # share this prepared table; never alter the upstream input in place.
    frame[coordinates] = np.trunc(frame[coordinates].to_numpy(dtype=float)).astype(np.int64)
    residents = frame.loc[~frame.is_visitor, 'person_id']
    if residents.isna().any() or residents.astype(str).duplicated().any():
        raise ValueError('Airport residents must have unique, non-missing person IDs')
    resident_ids = set(residents.astype(str))
    # The Swiss Java configurator counts adults in existing households. Retain
    # resident age for that calculation; visitor agents have no household record.
    ages = population.persons.set_index(population.persons.person_id.astype(str))['age']
    frame['age'] = frame.person_id.astype(str).map(ages)
    if 'age' in trips:
        frame['age'] = frame.age.fillna(trips.reset_index(drop=True).age)
    if frame.loc[~frame.is_visitor, 'age'].isna().any():
        raise ValueError('Resident airport agents require age for household preparation')

    # Remove ordinary resident plans and resources, preserving their person IDs.

    persons = population.persons.loc[~population.persons.person_id.astype(str).isin(resident_ids)].copy()
    activities = population.activities.loc[~population.activities.person_id.astype(str).isin(resident_ids)].copy()
    vehicles = population.vehicles.loc[~population.vehicles.owner_id.astype(str).isin(resident_ids)].copy()
    missing_residents = resident_ids - set(population.persons.person_id.astype(str))
    if missing_residents:
        logger.info('Restoring %d sampled airport residents absent from ordinary prepared plans', len(missing_residents))

    # Reserve all exported IDs, including freight and LCV and filtered residents.

    occupied = set(population.persons.person_id.astype(str)) | resident_ids
    for prefix, demand in [('freight_', population.freight_trips), ('lcv_', population.lcv_trips)]:
        if demand is not None:
            ids = set(prefix + demand.agent_id.astype(str))
            if resident_ids & ids:
                raise ValueError('Airport resident IDs overlap freight/LCV IDs')
            occupied.update(ids)
    agent_ids = frame.person_id.astype('object').copy()
    for index in frame.index[frame.is_visitor]:
        agent_ids.loc[index] = _reserve_airport_id('airport_visitor_' + str(frame.at[index, 'trip_id']), occupied)
    frame['agent_id'] = agent_ids.astype(str)
    to = frame.direction.eq('to_airport')
    lodging = np.where(frame.is_visitor, 'home_secondary', 'home')
    frame['origin_purpose'] = np.where(to, lodging, 'airport')
    frame['destination_purpose'] = np.where(to, 'airport', lodging)
    # Explicit facilities keep the airport/station and accommodation coordinates
    # intact during Java preparation, which requires a facility for each activity.
    for endpoint in ['origin', 'destination']:
        frame[endpoint + '_facility_id'] = 'airport:' + frame.agent_id + ':' + endpoint

    # Supply only the mode resource used by each plan, with unique vehicle IDs.

    vehicle_ids = set(map(str, reserved_vehicle_ids)) | set(population.vehicles.vehicle_id.astype(str))
    for fleet in [population.freight_vehicles, population.lcv_vehicles]:
        if fleet is not None:
            vehicle_ids.update(fleet.vehicle_id.astype(str))
    fleet_rows = []
    for trip in frame.itertuples(index=False):
        if trip.mode in ['car', 'car_passenger', 'bike']:
            vehicle_id = _reserve_airport_id(f'airport:{trip.agent_id}:{trip.mode}', vehicle_ids)
            fleet_rows.append((trip.mode, vehicle_id, trip.agent_id, 'default_' + trip.mode, 0, '6'))
    fleet = pd.DataFrame(fleet_rows, columns=['mode', 'vehicle_id', 'owner_id', 'type_id', 'age', 'euro'])
    return replace(population, persons=persons, activities=activities, vehicles=vehicles,
                   airport_trips=frame.sort_values('agent_id').reset_index(drop=True),
                   airport_vehicles=fleet.sort_values('owner_id').reset_index(drop=True))


def _reserve_airport_id(base, occupied):
    """Resolve collisions deterministically, including collisions with suffixes."""
    candidate, suffix = base, 0
    while candidate in occupied:
        suffix += 1
        candidate = f'{base}_{suffix}'
    occupied.add(candidate)
    return candidate


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
