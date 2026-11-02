"""Reproducible GTFS benchmark: genuine FLASH-TB, unpruned TB and Java RAPTOR.

Run python -m mode_choice.routing.flash.benchmark --help. Generated files stay in --output.
"""
import argparse
import json
import hashlib
from pathlib import Path
import re
import subprocess
import time

import numpy as np
import pandas as pd
import polars as pl
from pyproj import Transformer

from mode_choice.routing.flash.router import Router, read_stops
from mode_choice.routing.flash.runtime import build
from mode_choice.routing.flash.timetable import prepare


def fixture(directory):
    directory.mkdir(parents=True, exist_ok=True)
    def write(name, rows): pd.DataFrame(rows).to_csv(directory / (name + '.txt'), index=False)
    write('agency', [dict(agency_id='x', agency_name='fixture', agency_url='http://example.org', agency_timezone='Europe/Zurich')])
    write('stops', [dict(stop_id=str(i), stop_name=str(i), stop_lat=46.2, stop_lon=6.1 + .01 * i) for i in range(4)])
    write('routes', [dict(route_id=str(i), agency_id='x', route_short_name=str(i), route_long_name=str(i), route_type=3) for i in range(3)])
    write('calendar', [dict(service_id='daily', monday=1, tuesday=1, wednesday=1, thursday=1, friday=1, saturday=1, sunday=1, start_date=20260301, end_date=20260331)])
    for name, header in [('calendar_dates', 'service_id,date,exception_type'), ('frequencies', 'trip_id,start_time,end_time,headway_secs'), ('transfers', 'from_stop_id,to_stop_id,min_transfer_time,transfer_type')]:
        (directory / (name + '.txt')).write_text(header + '\n')
    trips, times = [], []
    for hour in range(6, 23):
        for route, sequence in [(0, [(0, 0), (1, 5), (2, 10)]), (1, [(2, 12), (3, 17)]), (2, [(0, 4), (3, 24)])]:
            trip = f'{hour}_{route}'
            trips.append(dict(route_id=str(route), service_id='daily', trip_id=trip, trip_headsign=trip, trip_short_name=trip))
            for index, (stop, minute) in enumerate(sequence):
                clock = f'{hour:02d}:{minute:02d}:00'
                times.append(dict(trip_id=trip, arrival_time=clock, departure_time=clock, stop_id=str(stop), stop_sequence=index + 1, pickup_type=0, drop_off_type=0))
    write('trips', trips)
    write('stop_times', times)
    return directory


def java_inputs(feed, timetable, directory):
    native = read_stops(timetable['stops'])
    x, y = Transformer.from_crs(4326, 2056, always_xy=True).transform(native['lon'].to_numpy(), native['lat'].to_numpy())
    stops = native.with_columns(pl.Series('x', x), pl.Series('y', y))
    stops.select('stop', 'x', 'y').write_csv(directory / 'java_stops.tsv', separator='\t')
    mapping = stops.select(pl.col('gtfs_stop_id').alias('stop_id'), 'stop')
    events = pl.read_csv(feed / 'stop_times.txt', schema_overrides={'trip_id': pl.String, 'stop_id': pl.String})
    # External feeds must already be filtered to the benchmark day; fixture is single-service.
    def seconds(column):
        pieces = pl.col(column).str.split_exact(':', 2)
        return sum(pieces.struct.field(f'field_{i}').cast(pl.Int64) * factor for i, factor in enumerate([3600, 60, 1]))
    events = events.join(mapping, on='stop_id', validate='m:1').sort('trip_id', 'stop_sequence').select(
        pl.col('trip_id').alias('trip'), 'stop', seconds('arrival_time').alias('arrival'), seconds('departure_time').alias('departure'))
    events.write_csv(directory / 'java_events.tsv', separator='\t')
    return stops


def run(args):
    if args.queries <= 0: raise ValueError('queries must be positive')
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    binary = Path(args.binary).resolve() if args.binary else directory / 'flash_tb_router'
    with (directory / 'build.log').open('w') as log:
        if not args.binary: build(args.source, binary, args.cxx, args.numa_include, log)
    feed = Path(args.gtfs).resolve() if args.gtfs else fixture(directory / 'gtfs')
    if args.gtfs:
        # Copy and date-filter before either router sees the feed: avoid a mismatched Java timetable.
        from data.gtfs.utils import read_feed, filter_feed_to_date, write_feed
        import shutil
        import zipfile
        archive = directory / 'source.zip'
        with zipfile.ZipFile(archive, 'w') as zipfile_out:
            for path in feed.glob('*.txt'): zipfile_out.write(path, path.name)
        filtered = filter_feed_to_date(read_feed(str(archive)), args.date)
        feed = directory / 'gtfs'
        write_feed(filtered, str(feed))
    began = time.perf_counter()
    timetable = prepare(binary, feed, directory / 'prepared', args.date, 1, args.cells)
    preprocessing = time.perf_counter() - began
    stops = java_inputs(feed, timetable, directory)
    rng = np.random.default_rng(args.seed)
    # Coordinate requests exactly at stops isolate the transit algorithms.
    if not args.gtfs:
        origins = np.zeros(args.queries, dtype=int)
        targets = np.full(args.queries, stops.height - 1, dtype=int)
        departures = rng.integers(7 * 3600, 20 * 3600, args.queries)
    else:
        origins = rng.integers(0, stops.height, args.queries)
        targets = rng.integers(0, stops.height, args.queries)
        departures = rng.integers(7 * 3600, 20 * 3600, args.queries)
    trips = pl.DataFrame(dict(trip_id=[str(i) for i in range(args.queries)], origin_x=stops['x'].to_numpy()[origins],
        origin_y=stops['y'].to_numpy()[origins], destination_x=stops['x'].to_numpy()[targets], destination_y=stops['y'].to_numpy()[targets], departure_time=departures.astype(float)))
    trips.write_parquet(directory / 'requests.parquet')
    np.savetxt(directory / 'java_requests.tsv', np.column_stack([np.arange(args.queries), origins, targets, departures]), fmt='%d', delimiter='\t')
    initialization_start = time.perf_counter()
    router = Router(timetable, access_stops=1, max_access_distance_m=1.)
    initialization_seconds = time.perf_counter() - initialization_start
    timings, results = {}, {}
    for algorithm in ('flash', 'tb'):
        results[algorithm] = router.route(trips, directory, algorithm)
        timings[algorithm] = dict(router.timings)
        results[algorithm].write_parquet(directory / f'{algorithm}.parquet')
    router.close()
    jar = str(Path(args.jar).resolve())
    classes = directory / 'classes'
    classes.mkdir(exist_ok=True)
    java_source = Path(__file__).parent / 'native/JavaBenchmark.java'
    with (directory / 'javac.log').open('w') as log:
        subprocess.run(['javac', '-cp', jar, '-d', str(classes), str(java_source)], check=True, stdout=log, stderr=log)
    began = time.perf_counter()
    with (directory / 'java.log').open('w') as log:
        subprocess.run(['java', '-Xmx2G', '-Dmatsim.useLocalDtds=true', '-cp', str(classes) + ':' + jar, 'JavaBenchmark',
                        str(directory / 'java_stops.tsv'), str(directory / 'java_events.tsv'), str(directory / 'java_requests.tsv'), str(directory / 'java.csv'), timetable['timetable'] + '.transfers.tsv'], check=True, stdout=log, stderr=log)
    timings['java'] = {'process_seconds': time.perf_counter() - began}
    match = re.search(r'QUERY_SECONDS=([0-9.eE+-]+)', (directory / 'java.log').read_text())
    if match: timings['java']['loop_seconds'] = float(match[1])
    java = pl.read_csv(directory / 'java.csv', schema_overrides={'trip_id': pl.String, 'total_time_min': pl.Float64, 'distance_km': pl.Float64, 'in_vehicle_time_min': pl.Float64})
    comparison = results['flash'].join(java, on='trip_id', validate='1:1')
    comparison = comparison.with_columns(
        (pl.col('pt_total_travel_time_min') - pl.col('total_time_min')).alias('time_difference_min'),
        (pl.col('pt_in_vehicle_distance_km') - pl.col('distance_km')).alias('distance_difference_km'),
        (pl.col('pt_transfers') - pl.col('transfers')).alias('transfer_difference'))
    comparison.write_csv(directory / 'comparison.csv')
    reachable = comparison.filter((pl.col('pt_total_travel_time_min') < 1e9) & pl.col('total_time_min').is_finite())
    if reachable.is_empty(): raise RuntimeError('No jointly reachable requests: inspect Java logs before comparing timings')
    report = {'python_initialization_seconds': initialization_seconds, 'date': args.date, 'seed': args.seed, 'dataset': 'GTFS sample' if args.gtfs else 'controlled GTFS',
              'native_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
              'flash_reachable': int((results['flash']['pt_total_travel_time_min'] < 1e9).sum()),
              'java_reachable': int(java['total_time_min'].is_finite().sum()),
              'queries': args.queries, 'stops': stops.height, 'preprocessing_seconds': preprocessing, 'timings': timings,
              'matched_reachable': reachable.height, 'mean_absolute_time_difference_min': reachable['time_difference_min'].abs().mean(),
              'mean_absolute_distance_difference_km': reachable['distance_difference_km'].abs().mean(),
              'transfer_agreement': (reachable['transfer_difference'] == 0).mean(),
              'flash_vs_tb_max_time_difference_min': (results['flash']['pt_total_travel_time_min'] - results['tb']['pt_total_travel_time_min']).abs().max(),
              'scope': 'single-thread, stop-to-stop, matched timetable, matched transitive transfers and earliest-arrival cost; Java warm loop vs FLASH native loop both include result writing; not production utility/fare comparison'}
    (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--binary', help='Reuse an explicitly supplied compiled adapter')
    parser.add_argument('--jar', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--numa-include', default='')
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--gtfs', help='Optional extracted GTFS; a private date-filtered copy is benchmarked')
    parser.add_argument('--date', default='20260318')
    parser.add_argument('--queries', type=int, default=1000)
    parser.add_argument('--cells', type=int, default=2)
    parser.add_argument('--seed', type=int, default=518745)
    run(parser.parse_args())
