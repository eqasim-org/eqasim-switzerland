"""Combine Geneva/Zurich Swiss-endpoint ODs and apply the population sample rate."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.airports._shared import common
from data.airports._shared.common import apportion, atomic_write, export_signature, file_digest
from data.airports.geneva.model import resolve

VERSION = 1
COUNTS = ('annual_trips', 'period_trips', 'daily_trips')


def configure(context):
    context.stage('data.airports.geneva.ods')
    context.stage('data.airports.zurich.ods')
    context.config('data_path')
    context.config('input_downsampling')
    context.config('airport_ods_folder', default='airports/ods')


def execute(context):

    # Read full-population ODs and the same fraction used to sample the population.

    sources = {'GVA': context.stage('data.airports.geneva.ods'),
               'ZRH': context.stage('data.airports.zurich.ods')}
    rate = sampling_rate(context.config('input_downsampling'))
    folder = output_folder(context)
    token = input_signature(sources, rate)
    cached = load_ods(folder, token)
    if cached is not None:
        return cached

    # Exclude foreign endpoints and scale counts before expanding passenger rows.

    result = scale_ods(sources, rate)
    result.attrs['signature'] = token
    summary = result.attrs['summary']

    # Persist auditable source counts, sampled ODs and output checksums.

    atomic_write(folder / 'ods.parquet', lambda p: result.to_parquet(p, index=False))
    atomic_write(folder / 'ods.csv', lambda p: result.to_csv(p, index=False))
    atomic_write(folder / 'summary.json', lambda p: p.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8'))
    manifest = {'signature': token, 'artifacts': {name: file_digest(folder / name)
                for name in ('ods.parquet', 'ods.csv', 'summary.json')}}
    atomic_write(folder / 'signature.json', lambda p: p.write_text(json.dumps(manifest) + '\n', encoding='utf-8'))
    return result


def validate(context):
    payload = {'sampling_rate': sampling_rate(context.config('input_downsampling')),
               'code': [file_digest(Path(__file__)), file_digest(Path(common.__file__))],
               'exports': export_signature(output_paths(output_folder(context)))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sampling_rate(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or
            not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('input_downsampling must be a finite sampling fraction between zero and one')
    return float(value)


def output_folder(context):
    return resolve(Path(context.config('data_path')), context.config('airport_ods_folder'))


def output_paths(folder):
    return {name: folder / name for name in ('ods.parquet', 'ods.csv', 'summary.json', 'signature.json')}


def input_signature(sources, rate):
    digest = hashlib.sha256(json.dumps({'version': VERSION, 'sampling_rate': rate}, sort_keys=True).encode())
    for airport in ('GVA', 'ZRH'):
        digest.update(airport.encode())
        digest.update(pd.util.hash_pandas_object(sources[airport], index=False).to_numpy().tobytes())
    for path in (Path(__file__), Path(common.__file__)):
        digest.update(file_digest(path).encode())
    return digest.hexdigest()


def scale_ods(sources, rate):
    """Return combined Swiss ODs with scaled counts and retained source values."""
    rate = sampling_rate(rate)
    output = []
    summary = {'sampling_rate': rate, 'scope': 'Swiss ground endpoints only',
               'method': 'Scale floating counts once; hierarchical largest-remainder rounding within each airport',
               'by_airport': {}}
    for airport in ('GVA', 'ZRH'):
        source = sources[airport]
        required = {'country', 'direction', 'region_id', 'daily_trips', 'daily_trips_integer'}
        if not required.issubset(source.columns):
            raise ValueError(f'{airport} airport ODs missing columns: {sorted(required - set(source.columns))}')
        if 'sampling_rate' in source or any(c.startswith('full_population_') for c in source.columns):
            raise ValueError(f'{airport} airport ODs have already been scaled')
        frame = source.copy().reset_index(drop=True)
        frame['od_id'] = [f'{airport}:{i}' for i in range(len(frame))]
        foreign = frame.loc[frame.country.ne('CH')]
        frame = frame.loc[frame.country.eq('CH')].copy().reset_index(drop=True)
        if not frame.direction.isin(['to_airport', 'from_airport']).all() or frame.region_id.isna().any():
            raise ValueError(f'{airport} ODs require known directions and regions')
        if 'profile' not in frame:
            frame['profile'] = 'pooled'
        if frame.profile.isna().any():
            raise ValueError(f'{airport} ODs contain missing traveller profiles')
        frame['airport'] = airport
        frame['sampling_rate'] = rate
        frame['sample_weight'] = 1. / rate if rate else 0.
        for column in COUNTS:
            if column not in frame:
                continue
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'{airport} {column} must be finite and non-negative')
            integer = column + '_integer'
            frame['full_population_' + column] = values
            if integer in frame:
                original = frame[integer].to_numpy(dtype=float)
                if not np.isfinite(original).all() or (original < 0).any() or not np.equal(original, np.floor(original)).all():
                    raise ValueError(f'{airport} {integer} must contain non-negative integers')
                frame['full_population_' + integer] = original.astype(np.int64)
            frame[column] = values * rate
            frame[integer] = (frame['full_population_' + integer].to_numpy(dtype=np.int64)
                              if rate == 1 and 'full_population_' + integer in frame
                              else rounded_counts(frame, column))
        summary['by_airport'][airport] = {
            'full_population_daily_trips': float(frame.full_population_daily_trips.sum()),
            'full_population_daily_trips_integer': int(frame.full_population_daily_trips_integer.sum()),
            'scaled_daily_trips': float(frame.daily_trips.sum()),
            'scaled_daily_trips_integer': int(frame.daily_trips_integer.sum()),
            'excluded_foreign_daily_trips': float(foreign.daily_trips.sum()),
            'excluded_foreign_daily_trips_integer': int(foreign.daily_trips_integer.sum())}
        output.append(frame)
    result = pd.concat(output, ignore_index=True).convert_dtypes()
    result.attrs['summary'] = summary
    return result


def rounded_counts(frame, column):
    """Round airport totals first, then regional/profile groups and municipalities."""
    counts = np.zeros(len(frame), dtype=np.int64)
    groups = list(frame.groupby(['direction', 'region_id', 'profile'], sort=False, dropna=False).indices.values())
    values = frame[column].to_numpy(dtype=float)
    totals = np.array([values[indices].sum() for indices in groups])
    for indices, target in zip(groups, apportion(totals)):
        mass = values[indices].sum()
        if target > 0 and mass > 0:
            counts[indices] = apportion(values[indices] / mass * target)
    return counts


def split_airports(frame):
    """Keep stage metadata when selecting each airport for trip generation."""
    if not {'airport', 'sampling_rate', 'sample_weight', 'od_id'}.issubset(frame.columns):
        raise ValueError('Trip generation requires ODs from data.airports.ods')
    if not frame.country.eq('CH').all():
        raise ValueError('Scaled airport ODs must have Swiss ground endpoints only')
    return {airport: frame.loc[frame.airport.eq(airport)].copy() for airport in ('GVA', 'ZRH')}


def load_ods(folder, token):
    try:
        manifest = json.loads((folder / 'signature.json').read_text(encoding='utf-8'))
        if manifest['signature'] != token:
            return None
        if any(file_digest(folder / name) != manifest['artifacts'][name]
               for name in ('ods.parquet', 'ods.csv', 'summary.json')):
            return None
        result = pd.read_parquet(folder / 'ods.parquet')
        result.attrs['summary'] = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        result.attrs['signature'] = token
        return result
    except (OSError, KeyError, TypeError, ValueError):
        return None
