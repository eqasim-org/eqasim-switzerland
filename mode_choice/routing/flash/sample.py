"""Extract a small real GTFS region for tests without editing the source feed."""
from pathlib import Path
import argparse
import shutil
import polars as pl


def extract(feed, output, bounds, route_limit=2):
    feed, output = Path(feed).resolve(), Path(output).resolve()
    if feed == output or feed in output.parents:
        raise ValueError('Use a separate output directory; upstream feeds are read-only')
    output.mkdir(parents=True, exist_ok=True)
    def scan(name): return pl.scan_csv(feed / (name + '.txt'), infer_schema_length=0)
    west, south, east, north = bounds
    stops = scan('stops').filter(pl.col('stop_lon').cast(pl.Float64).is_between(west, east)
                                & pl.col('stop_lat').cast(pl.Float64).is_between(south, north)).collect()
    visiting = scan('stop_times').filter(pl.col('stop_id').is_in(stops['stop_id'].implode())).select('trip_id').unique().collect()
    trips = scan('trips').collect()
    routes = (trips.join(visiting, on='trip_id', how='semi').group_by('route_id').len()
              .sort(['len', 'route_id'], descending=[True, False]).head(route_limit)['route_id'])
    trips = trips.filter(pl.col('route_id').is_in(routes.implode()))
    events = scan('stop_times').filter(pl.col('trip_id').is_in(trips['trip_id'].implode())).collect()
    used = events['stop_id'].unique()
    for name, frame in [('trips', trips), ('stop_times', events),
                        ('stops', scan('stops').filter(pl.col('stop_id').is_in(used.implode())).collect()),
                        ('routes', scan('routes').filter(pl.col('route_id').is_in(routes.implode())).collect())]:
        frame.write_csv(output / (name + '.txt'))
    for name in ('agency', 'calendar', 'calendar_dates'):
        if (feed / (name + '.txt')).exists(): shutil.copyfile(feed / (name + '.txt'), output / (name + '.txt'))
    if (feed / 'transfers.txt').exists():
        scan('transfers').filter(pl.col('from_stop_id').is_in(used.implode()) & pl.col('to_stop_id').is_in(used.implode())).collect().write_csv(output / 'transfers.txt')
    else: (output / 'transfers.txt').write_text('from_stop_id,to_stop_id,min_transfer_time,transfer_type\n')
    if (feed / 'frequencies.txt').exists():
        scan('frequencies').filter(pl.col('trip_id').is_in(trips['trip_id'].implode())).collect().write_csv(output / 'frequencies.txt')
    else: (output / 'frequencies.txt').write_text('trip_id,start_time,end_time,headway_secs\n')
    print({'routes': routes.to_list(), 'trips_before_date_filter': trips.height, 'stops': len(used), 'output': str(output)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feed', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bounds', type=float, nargs=4, default=[6.1, 46.18, 6.18, 46.24])
    parser.add_argument('--routes', type=int, default=2)
    args = parser.parse_args()
    extract(args.feed, args.output, args.bounds, args.routes)
