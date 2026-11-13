"""Stable observed PT requests, with verified original survey keys for Google joins."""
from pathlib import Path
import json
import polars as pl
from pyproj import Transformer

FIELDS = ['origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']


def configure(context):
    context.stage('data.microcensus.trips')
    context.config('data_path')
    context.config('random_seed')
    context.config('pt_microcensus_sample_size', None)


def execute(context):

    # Extract the trip table from the survey stage's tuple and retain observed PT.

    result = context.stage('data.microcensus.trips')
    trips = result[0] if isinstance(result, tuple) else result
    trips = trips if isinstance(trips, pl.DataFrame) else pl.from_pandas(trips)
    trips = prepare_trips(trips)
    valid = pl.all_horizontal([pl.col(c).is_finite() for c in FIELDS]).fill_null(False) & (pl.col('departure_time') >= 0)
    trips.filter(~valid).write_parquet(Path(context.path()) / 'excluded_invalid_inputs.parquet')
    total, invalid = trips.height, trips.filter(~valid).height
    trips = trips.filter(valid)
    size = context.config('pt_microcensus_sample_size')
    if size is not None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError('pt_microcensus_sample_size must be null (all trips) or a positive integer')
        trips = trips.sample(n=min(size, trips.height), seed=context.config('random_seed')).sort('person_id', 'survey_trip_id')
    if trips.is_empty():
        raise ValueError('No observed PT trips have valid coordinates and departure times')

    # Recover original WEGNR using departure time and endpoint checks, not renumbered IDs.

    source = Path(context.config('data_path')) / 'microcensus' / 'wege.csv'
    raw = pl.read_csv(source, encoding='windows-1252', columns=[
        'HHNR', 'WEGNR', 'f51100', 'S_X_CH1903', 'S_Y_CH1903', 'Z_X_CH1903', 'Z_Y_CH1903'])
    trips = attach_original_keys(trips, raw)
    trips.write_parquet(Path(context.path()) / 'requests.parquet')
    (Path(context.path()) / 'selection.json').write_text(json.dumps({
        'observed_pt_trips': total, 'invalid_inputs': invalid, 'routed_requests': trips.height,
        'unverified_original_keys': trips['google_trip_id'].null_count(),
        'sample_size': size, 'random_seed': context.config('random_seed')}, indent=2) + '\n')
    return trips


def prepare_trips(trips):
    trips = trips.filter(pl.col('mode') == 'pt').rename({'trip_id': 'survey_trip_id'}).with_columns(
        pl.col('person_id', 'survey_trip_id').cast(pl.Int64), pl.col(FIELDS).cast(pl.Float64))
    if trips['person_id'].null_count() or trips['survey_trip_id'].null_count():
        raise ValueError('Microcensus person/trip identifiers must be non-null')
    if trips.select('person_id', 'survey_trip_id').is_duplicated().any():
        raise ValueError('Microcensus person/trip identifiers must be unique')
    return trips.with_columns(pl.concat_str('person_id', 'survey_trip_id', separator=':').alias('trip_id'),
                             ((pl.col('arrival_time') - pl.col('departure_time')) / 60).alias('observed_time_min')).sort('person_id', 'survey_trip_id')


def attach_original_keys(trips, raw):
    transform = Transformer.from_crs(21781, 2056, always_xy=True)
    trips = trips.with_columns(pl.col('departure_time').cast(pl.Float64))
    raw = raw.rename({'HHNR': 'person_id', 'WEGNR': 'google_trip_id'}).with_columns(
        pl.col('person_id', 'google_trip_id').cast(pl.Int64), (pl.col('f51100').cast(pl.Float64) * 60).alias('departure_time'))
    for prefix, stem in [('S', 'origin'), ('Z', 'destination')]:
        x, y = transform.transform(raw[prefix + '_X_CH1903'].to_numpy(), raw[prefix + '_Y_CH1903'].to_numpy())
        raw = raw.with_columns(pl.Series('raw_' + stem + '_x', x), pl.Series('raw_' + stem + '_y', y))
    candidates = trips.select('trip_id', 'person_id', *FIELDS).join(raw, on=['person_id', 'departure_time'], how='inner')
    same = pl.all_horizontal([
        (((pl.col(stem + '_x') - pl.col('raw_' + stem + '_x')) ** 2 +
          (pl.col(stem + '_y') - pl.col('raw_' + stem + '_y')) ** 2).sqrt() <= 1.) for stem in ('origin', 'destination')])
    matches = candidates.filter(same.fill_null(False)).select('trip_id', 'google_trip_id')
    matches = matches.filter(~pl.col('trip_id').is_duplicated())
    # A split survey trip must not reuse the same Google journey more than once.
    matches = matches.join(trips.select('trip_id', 'person_id'), on='trip_id', validate='1:1')
    matches = matches.filter(~pl.struct('person_id', 'google_trip_id').is_duplicated()).drop('person_id')
    return trips.join(matches, on='trip_id', how='left', validate='1:1', maintain_order='left')


def validate(context):
    path = Path(context.config('data_path')) / 'microcensus' / 'wege.csv'
    return [str(path), path.stat().st_mtime_ns, path.stat().st_size]
