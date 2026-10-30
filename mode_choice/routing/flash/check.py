"""Small executable regression checks; requires a compiled native adapter."""
import argparse
from pathlib import Path
import tempfile
import numpy as np
import polars as pl

from mode_choice.routing.flash.benchmark import fixture
from mode_choice.routing.flash.router import Router
from mode_choice.routing.flash.timetable import prepare


def check(binary):
    with tempfile.TemporaryDirectory(prefix='flash-tests-') as temporary:
        root = Path(temporary)
        feed = fixture(root / 'gtfs')
        timetable = prepare(binary, feed, root / 'prepared', '20260318', 1, 2)
        with Router(timetable, access_stops=1, max_access_distance_m=1, keep_process=True) as router:
            xy = router.xy
            trips = pl.DataFrame({'trip_id': ['9007199254740993', 'direct', 'unreachable'],
                                  'origin_x': [xy[0, 0], xy[0, 0], xy[3, 0]], 'origin_y': [xy[0, 1], xy[0, 1], xy[3, 1]],
                                  'destination_x': [xy[3, 0], xy[3, 0], xy[0, 0]], 'destination_y': [xy[3, 1], xy[3, 1], xy[0, 1]],
                                  'departure_time': [28740., 29040., 28740.]})
            result = router.route(trips)
            assert result['trip_id'].to_list() == trips['trip_id'].to_list()
            assert result['pt_total_travel_time_min'].to_list() == [18., 20., 1e9]
            assert result['pt_in_vehicle_time_min'].to_list()[:2] == [15., 20.]
            assert result['pt_transfers'].to_list()[:2] == [1., 0.]
            assert result['pt_initial_waiting_time_min'].to_list()[:2] == [1., 0.]
            assert result['pt_transfer_time_min'].to_list()[:2] == [2., 0.]
            assert 2.3 < result['pt_in_vehicle_distance_km'][0] < 2.32
            process = router.process.pid
            assert router.route(trips).equals(result)
            assert router.process.pid == process
            assert router.route(trips.head(0)).is_empty()
            for invalid in [trips.with_columns(pl.lit(float('nan')).alias('origin_x')),
                            trips.with_columns(pl.lit('same').alias('trip_id'))]:
                try: router.route(invalid)
                except ValueError: pass
                else: raise AssertionError('Invalid input accepted')
            plain = router.route(trips, algorithm='tb')
            assert np.array_equal(plain['pt_total_travel_time_min'].to_numpy(), result['pt_total_travel_time_min'].to_numpy())
        assert router.process is None
        for threads, cells in [(0, 2), (1, 1)]:
            try: prepare(binary, feed, root / 'invalid', '20260318', threads, cells)
            except ValueError: pass
            else: raise AssertionError('Invalid preprocessing options accepted')
        import subprocess
        try: prepare(binary, feed, root / 'no-service', '20270401', 1, 2)
        except subprocess.CalledProcessError: pass
        else: raise AssertionError('Inactive service day accepted')
        print('Passed routing regression checks: direct/transfer/unreachable routing, metrics, large IDs, TB equivalence, process reuse/cleanup, empty/invalid inputs and preprocessing options.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    check(parser.parse_args().binary)
