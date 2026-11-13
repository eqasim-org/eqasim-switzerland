"""Zurich airport passenger ground demand from archived, reviewed public data."""

import calendar
import hashlib
import json
import logging
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from data.airports import common, geneva_times
from data.airports.common import apportion, atomic_write, export_signature, file_digest

MODEL_VERSION = 2
logger = logging.getLogger('synpp')


def configure(context):
    context.config('data_path')
    context.stage('data.airports.model')
    context.config('override_zurich_ods', default=False, volatile=True)
    context.config('zurich_airport_folder', default='airports/zurich')
    context.config('zurich_airport_month', default=None)
    context.config('zurich_airport_demand_factor', default=1.0)


def execute(context):

    # Validate archived evidence and check for unchanged processed outputs.

    root = input_folder(context)
    settings = demand_settings(context)
    inputs = load_inputs(root)
    allocation_model = context.stage('data.airports.model')
    token = input_signature(root, settings, inputs['metadata'], allocation_model.signature)
    trips = None if context.config('override_zurich_ods') else load_processed_data(root, token)
    if trips is not None:
        logger.info('Zurich airport: reusing %s', root / 'zurich_ods.parquet')
        return trips

    # Build the coarsest supported regional demand without inferring passenger modes.

    trips = allocation_model.allocate(build_trips(inputs['monthly'], inputs['regions'], **settings), 'ZRH')
    timing = geneva_times.build_time_distribution(inputs['timing'])
    logger.warning('Zurich catchment uses a historical 88/12 Switzerland/abroad benchmark; no current canton OD is available.')

    # Save counts, timing coverage and source limitations for downstream generation.

    summary = build_summary(trips, inputs, settings)
    summary['municipal_allocation'] = {'model_signature': allocation_model.signature,
        'parameters': allocation_model.parameters, 'swiss_resolution': 'municipality',
        'foreign_resolution': 'unresolved outside-Switzerland group',
        'visitor_share_transferred_from_geneva': allocation_model.swiss_visitor_share}
    save_outputs(root, token, trips, timing, summary, inputs['monthly'])
    logger.info('Zurich airport: %.1f daily local passenger ground trips', trips.daily_trips.sum())
    return trips


def validate(context):
    root = input_folder(context)
    metadata = json.loads((root / 'data_from_websites/zurich_airport.json').read_text(encoding='utf-8'))
    token = input_signature(root, demand_settings(context), metadata)
    token += export_signature({**output_paths(root), 'manifest': root / 'zurich_ods.signature.json'})
    return token if not context.config('override_zurich_ods') else token + str(uuid.uuid4())


def input_folder(context):
    path = Path(context.config('zurich_airport_folder'))
    return path if path.is_absolute() else Path(context.config('data_path')) / path


def demand_settings(context):
    month = context.config('zurich_airport_month')
    factor = context.config('zurich_airport_demand_factor')
    if month is not None and (isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12):
        raise ValueError('zurich_airport_month must be null or an integer from 1 to 12')
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not np.isfinite(factor) or factor < 0:
        raise ValueError('zurich_airport_demand_factor must be finite and non-negative')
    return {'month': month, 'demand_factor': float(factor)}


def source_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Zurich source files must be inside the airport input folder')
    return path


def input_signature(root, settings, metadata, model_signature=None):
    paths = [root / 'data_from_websites/zurich_airport.json', source_path(root, metadata['monthly_file']),
             source_path(root, metadata['regions_file']), root / 'data_from_websites/zurich_flight_times.json',
             Path(__file__), Path(common.__file__), Path(geneva_times.__file__)]
    paths.extend(source_path(root, source['local_file']) for source in metadata['sources'])
    payload = {'version': MODEL_VERSION, 'settings': settings, 'digests': [file_digest(p) for p in paths], 'allocation_model': model_signature}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_inputs(root):
    metadata = json.loads((root / 'data_from_websites/zurich_airport.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported Zurich metadata schema')
    # Reviewed inputs are tied to the archived evidence. A changed source requires
    # review instead of quietly reusing old extracted numbers from a new PDF.
    for source in metadata['sources']:
        if not source.get('url', '').startswith('https://') or file_digest(source_path(root, source['local_file'])) != source['sha256']:
            raise ValueError(f'Review changed Zurich source and update its metadata: {source.get("id")}')
    for name, digest in metadata['reviewed_inputs'].items():
        if file_digest(source_path(root, name)) != digest:
            raise ValueError(f'Review changed Zurich table and update its metadata: {name}')
    monthly = pd.read_csv(source_path(root, metadata['monthly_file'])).sort_values('month').reset_index(drop=True)
    numeric = ['year', 'month', 'local_passengers', 'transfer_movements', 'transit_passengers',
               'terminal_arrivals', 'terminal_departures']
    if not set(numeric).issubset(monthly.columns):
        raise ValueError('Missing required Zurich monthly columns')
    if len(monthly) != 12 or monthly.month.tolist() != list(range(1, 13)):
        raise ValueError('Zurich monthly source must contain each month exactly once')
    values = monthly[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError('Zurich monthly counts must be finite non-negative integers')
    if not monthly.year.eq(metadata['traffic_year']).all():
        raise ValueError('Mixed Zurich traffic years')
    if monthly.local_passengers.sum() != metadata['annual_local_passengers']:
        raise ValueError('Zurich local months do not reconcile to the annual report')
    if monthly.transfer_movements.sum() != metadata['annual_transfer_movements']:
        raise ValueError('Zurich transfer months do not reconcile to the annual report')
    if monthly.transit_passengers.sum() != metadata['annual_transit_passengers']:
        raise ValueError('Zurich transit months do not reconcile to the annual report')
    local = monthly.terminal_arrivals + monthly.terminal_departures - monthly.transfer_movements
    if not local.eq(monthly.local_passengers).all():
        raise ValueError('Zurich local and terminal/transfer counts do not reconcile')
    if ((monthly.terminal_arrivals - monthly.transfer_movements / 2 < 0)
            | (monthly.terminal_departures - monthly.transfer_movements / 2 < 0)).any():
        raise ValueError('Zurich transfers exceed terminal traffic')
    total = monthly.local_passengers.sum() + monthly.transfer_movements.sum() + monthly.transit_passengers.sum() + metadata['annual_general_aviation_and_other']
    if total != metadata['annual_total_passengers']:
        raise ValueError('Zurich accounting categories do not reconcile')
    regions = pd.read_csv(source_path(root, metadata['regions_file']), keep_default_na=False)
    required = {'region_id', 'region_name', 'country', 'region_type', 'share', 'source_id', 'evidence'}
    if not required.issubset(regions.columns) or regions.empty or regions.region_id.duplicated().any():
        raise ValueError('Invalid or duplicate Zurich regions')
    shares = pd.to_numeric(regions.share).to_numpy(float)
    if not np.isfinite(shares).all() or (shares < 0).any() or not np.isclose(shares.sum(), 1., atol=1e-10, rtol=0):
        raise ValueError('Zurich regional shares must be non-negative and sum to one')
    if not set(regions.source_id).issubset(s['id'] for s in metadata['sources']):
        raise ValueError('Zurich regional source reference is missing')
    timing = geneva_times.load_snapshot(root / 'data_from_websites/zurich_flight_times.json')
    return {'metadata': metadata, 'monthly': monthly, 'regions': regions, 'timing': timing}


def build_trips(monthly, regions, month=None, demand_factor=1.0):
    year = int(monthly.year.iloc[0])
    year_days = 366 if calendar.isleap(year) else 365
    period = monthly if month is None else monthly.loc[monthly.month.eq(month)]
    days = year_days if month is None else calendar.monthrange(year, month)[1]
    output = []
    for direction, column in [('to_airport', 'terminal_departures'), ('from_airport', 'terminal_arrivals')]:
        annual = (monthly[column] - monthly.transfer_movements / 2).sum() * demand_factor
        count = (period[column] - period.transfer_movements / 2).sum() * demand_factor
        frame = regions.copy()
        frame['direction'] = direction
        frame['origin_region'] = frame.region_id if direction == 'to_airport' else 'ZRH'
        frame['destination_region'] = 'ZRH' if direction == 'to_airport' else frame.region_id
        frame['annual_trips'] = annual * frame.share
        frame['period_trips'] = count * frame.share
        frame['daily_trips'] = frame.period_trips / days
        frame['traffic_year'] = year
        frame['month'] = 0 if month is None else month
        frame['period_days'] = days
        frame['demand_factor'] = demand_factor
        frame['regional_evidence'] = 'historical_access_origin_proxy'
        output.append(frame)
    trips = pd.concat(output, ignore_index=True)
    for column in ['annual_trips', 'period_trips', 'daily_trips']:
        trips[column + '_integer'] = apportion(trips[column])
    return trips


def build_summary(trips, inputs, settings):
    timing = inputs['timing']
    coverage = geneva_times.flight_frame(timing)
    return {'unit': 'local scheduled/charter passenger ground-access movements',
            'airport': 'ZRH', 'settings': settings, 'annual_trips': float(trips.annual_trips.sum()),
            'daily_trips': float(trips.daily_trips.sum()), 'sources': inputs['metadata']['sources'],
            'assumptions': inputs['metadata']['assumptions'], 'catchment': inputs['metadata']['catchment'],
            'missing': inputs['metadata']['missing'],
            'context_only_modal_share': inputs['metadata']['context_only_modal_share'],
            'timing': {k: v for k, v in timing.items() if k != 'flights'},
            'covered_days': [geneva_times.DAY_NAMES[d] for d in sorted(coverage.weekday.unique())],
            'retained_flight_records': len(coverage),
            'notes': ['Weekday timing averages only available weekdays; missing named days raise an error.',
                      'Historical catchment is applied in both directions; no current canton or resident/visitor split inferred.',
                      'No passenger mode assignments, employee trips, greeters or freight generated.',
                      'Monthly local seasonality is observed, not inferred from total passengers.']}


def output_paths(root):
    return {'trips': root / 'zurich_ods.parquet', 'csv': root / 'zurich_ods.csv',
            'regional': root / 'zurich_ods.regional_od.csv', 'summary': root / 'zurich_ods.summary.json',
            'times': root / 'zurich_ods.time_distribution.csv', 'monthly': root / 'zurich_demand_monthly.csv', 'municipal': root / 'zurich_ods.municipality_od.csv'}


def load_processed_data(root, token):
    try:
        metadata = json.loads((root / 'zurich_ods.signature.json').read_text(encoding='utf-8'))
        if metadata['signature'] != token:
            return None
        paths = output_paths(root)
        if any(file_digest(path) != metadata['artifacts'][key] for key, path in paths.items()):
            return None
        return pd.read_parquet(paths['trips'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_outputs(root, token, trips, timing, summary, monthly):
    paths = output_paths(root)
    atomic_write(paths['trips'], lambda p: trips.to_parquet(p, index=False))
    atomic_write(paths['csv'], lambda p: trips.to_csv(p, index=False))
    coarse = trips.copy()
    if 'spatial_resolution' in coarse:
        coarse['origin_region'] = coarse.region_id.where(coarse.direction.eq('to_airport'), 'ZRH')
        coarse['destination_region'] = coarse.region_id.where(coarse.direction.eq('from_airport'), 'ZRH')
    counts = ['annual_trips', 'period_trips', 'daily_trips', 'annual_trips_integer', 'period_trips_integer', 'daily_trips_integer']
    coarse_keys = ['origin_region', 'destination_region', 'region_id', 'region_name', 'country', 'region_type', 'direction', 'traffic_year', 'month', 'period_days']
    regional = coarse.groupby(coarse_keys, as_index=False, sort=False)[counts].sum()
    atomic_write(paths['regional'], lambda p: regional.to_csv(p, index=False))
    atomic_write(paths['municipal'], lambda p: trips.to_csv(p, index=False))
    atomic_write(paths['times'], lambda p: timing.to_csv(p, index=False))
    demand = monthly.copy()
    demand['local_arrivals'] = demand.terminal_arrivals - demand.transfer_movements / 2
    demand['local_departures'] = demand.terminal_departures - demand.transfer_movements / 2
    demand['calendar_days'] = [calendar.monthrange(int(y), int(m))[1] for y, m in zip(demand.year, demand.month)]
    demand['average_daily_local_trips'] = demand.local_passengers / demand.calendar_days
    atomic_write(paths['monthly'], lambda p: demand.to_csv(p, index=False))
    atomic_write(paths['summary'], lambda p: p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'))
    manifest = {'signature': token, 'artifacts': {key: file_digest(path) for key, path in paths.items()}}
    atomic_write(root / 'zurich_ods.signature.json', lambda p: p.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8'))
