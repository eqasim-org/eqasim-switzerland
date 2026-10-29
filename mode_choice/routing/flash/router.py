"""Coordinate access/egress, native Pareto queries and explicit journey metrics."""
from pathlib import Path
import subprocess
import tempfile
import time
import re
import json

import numpy as np
import polars as pl
from pyproj import Transformer
from scipy.spatial import cKDTree

from mode_choice.defaults import UNROUTABLE_TIME_MIN
from mode_choice.routing.flash.cache import reader

COLUMNS = ('pt_total_travel_time_min', 'pt_total_distance_km', 'pt_in_vehicle_time_min',
           'pt_in_vehicle_distance_km', 'pt_access_egress_time_min', 'pt_transfer_time_min', 'pt_transfers')


class Router:
    def __init__(self, timetable, access_stops=4, max_access_distance_m=1500., walk_speed=1.3, detour=1.3, keep_process=False,
                 save_selected_legs=False, threads=1):
        if isinstance(access_stops, bool) or not isinstance(access_stops, int) or access_stops < 1 or not all(np.isfinite(v) and v > 0 for v in (max_access_distance_m, walk_speed, detour)):
            raise ValueError('Access stop count, distance, walking speed and detour must be positive')
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError('FLASH query threads must be a positive integer')
        self.threads = threads
        self.timetable = timetable
        self.process = None
        self.log = None
        self._cache_reader = reader(timetable)
        self._cache_reader.__enter__()
        try:
            self.stops = read_stops(timetable['stops'])
            transform = Transformer.from_crs(4326, 2056, always_xy=True)
            x, y = transform.transform(self.stops['lon'].to_numpy(), self.stops['lat'].to_numpy())
            self.xy = np.column_stack([x, y])
            self.tree = cKDTree(self.xy)
            self.k = min(access_stops, len(self.stops))
            self.radius, self.speed, self.detour = max_access_distance_m, walk_speed, detour
            self.timings = {}
            self.keep_process = keep_process
            self.save_selected_legs = save_selected_legs
            self.journeys = pl.DataFrame()
            self.selected_legs = pl.DataFrame()
        except BaseException:
            self.close()
            raise

    def route(self, trips, directory=None, algorithm='flash'):
        if self._cache_reader is None:
            raise RuntimeError('The FLASH router has been closed')
        if algorithm not in ('flash', 'tb'):
            raise ValueError('algorithm must be flash or tb')
        required = ['trip_id', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time']
        trips = trips.select(required)
        if trips['trip_id'].null_count() or trips['trip_id'].n_unique() != trips.height:
            raise ValueError('Trip IDs must be unique and non-null')
        if trips.filter(~pl.all_horizontal(pl.col(required[1:]).is_finite()).fill_null(False)
                        | (pl.col('departure_time') < 0)).height:
            raise ValueError('Invalid request coordinates or departure times')
        with tempfile.TemporaryDirectory(dir=directory, prefix='flash-query-') as scratch:
            return self._route(trips, Path(scratch), algorithm)

    def _route(self, trips, directory, algorithm):
        started = time.perf_counter()
        self.timings = {}
        self.journeys = pl.DataFrame()
        self.selected_legs = pl.DataFrame()
        queries, metadata = [], []
        origins = self.tree.query(trips.select('origin_x', 'origin_y').to_numpy(), k=list(range(1, self.k + 1)))
        targets = self.tree.query(trips.select('destination_x', 'destination_y').to_numpy(), k=list(range(1, self.k + 1)))
        for row in range(trips.height):
            for da, a in zip(origins[0][row], origins[1][row]):
                if da > self.radius: continue
                for db, b in zip(targets[0][row], targets[1][row]):
                    if db > self.radius or a == b: continue
                    access, egress = np.ceil(da * self.detour / self.speed), np.ceil(db * self.detour / self.speed)
                    queries.append((len(queries), a, b, int(np.ceil(trips['departure_time'][row] + access))))
                    metadata.append((row, access, egress, da * self.detour, db * self.detour))
        results = []
        if queries:
            source, target = directory / 'requests.tsv', directory / 'legs.tsv'
            np.savetxt(source, np.asarray(queries, dtype=np.int64), fmt='%d', delimiter='\t')
            native_start = time.perf_counter()
            if self.keep_process and algorithm == 'flash':
                if self.process is None:
                    self.log = tempfile.TemporaryFile(mode='w+t')
                    self.process = subprocess.Popen([self.timetable['binary'], 'serve', self.timetable['timetable'], str(self.threads)],
                                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True)
                self.process.stdin.write(json.dumps(str(source), ensure_ascii=False) + ' ' + json.dumps(str(target), ensure_ascii=False) + '\n')
                self.process.stdin.flush()
                while True:
                    line = self.process.stdout.readline()
                    if not line:
                        self.log.seek(0)
                        raise RuntimeError('FLASH-TB process failed: ' + self.log.read()[-4000:])
                    if line.strip() == 'FLASH_DONE': break
                self.log.seek(0, 2)
                end = self.log.tell()
                self.log.seek(max(0, end - 2048))
                diagnostics = self.log.read()
                self.log.seek(0, 2)
            else:
                with (directory / 'native.log').open('w') as log:
                    subprocess.run([self.timetable['binary'], 'query', self.timetable['timetable'], str(source), str(target), algorithm, str(self.threads)],
                                   check=True, stdout=log, stderr=log)
                diagnostics = (directory / 'native.log').read_text()
            self.timings['native_process_seconds'] = time.perf_counter() - native_start
            for label, name, cast in [('QUERY_SECONDS', 'native_loop_seconds', float),
                                      ('QUERY_THREADS', 'native_query_threads', int),
                                      ('QUERY_COUNT', 'native_queries_reported', int)]:
                matches = re.findall(label + r'=([0-9.eE+-]+)', diagnostics)
                if matches: self.timings[name] = cast(matches[-1])
            if self.timings.get('native_queries_reported', len(queries)) != len(queries):
                raise RuntimeError('FLASH did not process all stop-pair requests')
            legs = pl.read_csv(target, separator='\t', schema_overrides={
                **{c: pl.Int64 for c in ('query', 'journey', 'leg', 'departure', 'arrival', 'vehicle')},
                'from': pl.String, 'to': pl.String, 'distance_m': pl.Float64}).with_columns(
                    pl.col('from', 'to').str.replace_all(',', '', literal=True).cast(pl.Int64))
            self.journeys = legs
            meta = pl.DataFrame([(*entry, queries[i][3], float(trips['departure_time'][entry[0]])) for i, entry in enumerate(metadata)],
                                schema=['row', 'access', 'egress', 'access_m', 'egress_m', 'native_departure', 'original_departure'], orient='row').with_row_index('query').with_columns(pl.col('query').cast(pl.Int64))
            group = ['query', 'journey']
            legs = legs.join(meta, on='query', validate='m:1').sort('query', 'journey', 'leg').with_columns(
                pl.col('arrival').shift(1).over(group).fill_null(pl.col('native_departure')).alias('previous'),
                pl.col('leg').max().over(group).alias('last_leg'),
                pl.col('vehicle').cum_sum().over(group).alias('boardings'))
            vehicle = pl.col('vehicle') == 1
            first = (pl.col('leg') == 0) & ~vehicle
            last = (pl.col('leg') == pl.col('last_leg')) & ~vehicle
            duration = pl.col('arrival') - pl.col('departure')
            initial_walk = pl.col('departure') - pl.col('native_departure') if algorithm == 'flash' else duration
            initial_wait = duration if algorithm == 'flash' else pl.lit(0.)
            wait = (pl.col('departure') - pl.col('previous')).clip(lower_bound=0)
            metrics = legs.group_by(group).agg(
                pl.col('row').first(),
                ((pl.col('arrival').max() + pl.col('egress').first() - pl.col('original_departure').first()) / 60).alias('pt_total_travel_time_min'),
                ((pl.col('distance_m').sum() + pl.col('access_m').first() + pl.col('egress_m').first()) / 1000).alias('pt_total_distance_km'),
                duration.filter(vehicle).sum().truediv(60).alias('pt_in_vehicle_time_min'),
                pl.col('distance_m').filter(vehicle).sum().truediv(1000).alias('pt_in_vehicle_distance_km'),
                ((pl.col('access').first() + pl.col('egress').first() +
                  initial_walk.filter(first).sum() + duration.filter(last & ~first).sum()) / 60).alias('pt_access_egress_time_min'),
                ((duration.filter(~vehicle & ~first & ~last).sum() + wait.filter(vehicle & (pl.col('boardings') > 1)).sum()) / 60).alias('pt_transfer_time_min'),
                (pl.col('vehicle').sum() - 1).cast(pl.Float64).alias('pt_transfers'),
                ((initial_wait.filter(first).sum() + wait.filter(vehicle & (pl.col('boardings') == 1)).sum()) / 60).alias('pt_initial_waiting_time_min'),
            ).filter(pl.col('pt_transfers') >= 0)
            results = metrics.select('row', 'query', 'journey', *COLUMNS, 'pt_initial_waiting_time_min')
        # Earliest arrival, then fewer transfers. Do not pretend this is the Java utility objective.
        schema = {'row': pl.Int64, 'query': pl.Int64, 'journey': pl.Int64, **{c: pl.Float64 for c in (*COLUMNS, 'pt_initial_waiting_time_min')}}
        # Query and journey indices break exact ties independently of Polars' group order.
        candidates = (results if isinstance(results, pl.DataFrame) else pl.DataFrame(results, schema=schema)).sort(
            ['row', 'pt_total_travel_time_min', 'pt_transfers', 'query', 'journey'])
        best = candidates.unique('row', keep='first', maintain_order=True)
        if self.save_selected_legs and not self.journeys.is_empty():
            keys = trips.select('trip_id').with_row_index('row').with_columns(pl.col('row').cast(pl.Int64))
            selected = best.select('row', 'query', 'journey').join(keys, on='row', how='left', validate='m:1')
            self.selected_legs = self.journeys.join(selected.select('trip_id', 'query', 'journey'), on=['query', 'journey'], how='inner', validate='m:1')
        best = best.drop('query', 'journey')
        output = trips.select('trip_id').with_row_index('row').with_columns(pl.col('row').cast(pl.Int64)).join(best, on='row', how='left', validate='1:1', maintain_order='left').drop('row')
        output = output.with_columns([pl.col(c).fill_null(UNROUTABLE_TIME_MIN if 'time' in c else 0.) for c in (*COLUMNS, 'pt_initial_waiting_time_min')])
        self.timings.update(total_seconds=time.perf_counter() - started, native_queries=len(queries), input_trips=trips.height)
        return output

    def close(self):
        try:
            if self.process is not None:
                try: self.process.stdin.close()
                except BrokenPipeError: pass
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process.stdout.close()
                self.log.close()
                self.process = None
        finally:
            if self._cache_reader is not None:
                self._cache_reader.__exit__(None, None, None)
                self._cache_reader = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_stops(path):
    """Read current and legacy TSVs without inferring GTFS identifiers as numbers."""
    stops = pl.read_csv(path, separator='\t', schema_overrides={
        'stop': pl.String, 'lat': pl.Float64, 'lon': pl.Float64, 'gtfs_stop_id': pl.String,
    }).with_columns(pl.col('stop').str.replace_all(',', '', literal=True).cast(pl.Int64))
    # The spatial tree's row indices are sent to the native router as stop IDs.
    if stops.is_empty() or not np.array_equal(stops['stop'].to_numpy(), np.arange(stops.height)):
        raise ValueError('FLASH stops must have contiguous IDs in row order, starting at zero')
    return stops
