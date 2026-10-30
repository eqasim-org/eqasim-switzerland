"""Trip-aligned PT diagnostics; report only quantities actually exported by each router."""
from pathlib import Path
import csv
import datetime
import gzip
import html
import json
from urllib.parse import unquote, urlparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from lxml import etree

from mode_choice.defaults import UNROUTABLE_TIME_MIN
from mode_choice.routing.flash.router import COLUMNS


def reachable(frame):
    times = frame['pt_total_travel_time_min'].to_numpy()
    return np.isfinite(times) & (times >= 0) & (times < UNROUTABLE_TIME_MIN)


def save_figure(figure, output, name):
    figure.tight_layout()
    figure.savefig(Path(output) / name, dpi=150)
    plt.close(figure)


def single_overview(routes, output, backend):
    valid = routes.filter(pl.Series(reachable(routes)))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, column, label in zip(axes, COLUMNS[:2] + ('pt_transfers',),
                                   ['Total travel time (min)', 'Total distance (km)', 'Transfers']):
        axis.hist(valid[column].to_numpy(), bins=35)
        axis.set(xlabel=label, ylabel='Trips')
    figure.suptitle(f'{backend.upper()}: {valid.height:,}/{routes.height:,} reachable trips')
    save_figure(figure, output, backend + '_overview.png')


def compare(java, flash, output):
    for name, frame in [('java', java), ('flash', flash)]:
        if frame['trip_id'].null_count() or frame['trip_id'].n_unique() != frame.height:
            raise ValueError(f'{name}: duplicate or missing trip IDs')
    if java.height != flash.height or java.select('trip_id').join(flash.select('trip_id'), on='trip_id', how='anti').height:
        raise ValueError('The PT backends did not return the same requested trip IDs')
    joined = java.with_columns(pl.Series('java_reachable', reachable(java))).join(
        flash.with_columns(pl.Series('flash_reachable', reachable(flash))),
        on='trip_id', suffix='_flash', validate='1:1', maintain_order='left')
    joined.write_parquet(Path(output) / 'comparison.parquet')
    joined.write_csv(Path(output) / 'comparison.csv')
    both = joined.filter(pl.col('java_reachable') & pl.col('flash_reachable'))
    reachability = {
        'both': both.height,
        'java_only': joined.filter(pl.col('java_reachable') & ~pl.col('flash_reachable')).height,
        'flash_only': joined.filter(~pl.col('java_reachable') & pl.col('flash_reachable')).height,
        'neither': joined.filter(~pl.col('java_reachable') & ~pl.col('flash_reachable')).height,
    }
    rows = []
    for column in COLUMNS:
        a, b = both[column].to_numpy(), both[column + '_flash'].to_numpy()
        finite = np.isfinite(a) & np.isfinite(b)
        a, b = a[finite], b[finite]
        delta = b - a
        rows.append({'variable': column, 'matched_trips': len(a),
                     'java_mean': float(a.mean()) if len(a) else None,
                     'flash_mean': float(b.mean()) if len(a) else None,
                     'flash_minus_java_mean': float(delta.mean()) if len(a) else None,
                     'median_absolute_difference': float(np.median(abs(delta))) if len(a) else None,
                     'p90_absolute_difference': float(np.quantile(abs(delta), .9)) if len(a) else None})
    pl.DataFrame(rows).write_csv(Path(output) / 'metrics.csv')
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    for column, label, axis in zip([COLUMNS[0], COLUMNS[1], 'pt_transfers'],
                                   ['Total time (min)', 'Total distance (km)', 'Transfers'], axes[0]):
        a, b = both[column].to_numpy(), both[column + '_flash'].to_numpy()
        axis.scatter(a, b, s=7, alpha=.3)
        limit = max(float(np.max(a, initial=0)), float(np.max(b, initial=0)), 1)
        axis.plot([0, limit], [0, limit], color='grey', linestyle='--')
        axis.set(xlabel='Java: ' + label, ylabel='FLASH: ' + label)
    for column, label, axis in zip([COLUMNS[0], COLUMNS[1]], ['Time difference (min)', 'Distance difference (km)'], axes[1, :2]):
        axis.hist((both[column + '_flash'] - both[column]).to_numpy(), bins=40)
        axis.axvline(0, color='grey', linestyle='--')
        axis.set(xlabel='FLASH − Java: ' + label, ylabel='Matched trips')
    axes[1, 2].bar(list(reachability), list(reachability.values()))
    axes[1, 2].set(ylabel='Trips', title='Reachability (all requests)')
    save_figure(figure, output, 'route_comparison.png')
    figure, axis = plt.subplots(figsize=(8, 4))
    for name, frame in [('Java', java), ('FLASH', flash)]:
        values = np.sort(frame.filter(pl.Series(reachable(frame)))[COLUMNS[0]].to_numpy())
        if len(values):
            axis.plot(values, np.arange(1, len(values) + 1) / len(values), label=name)
    axis.set(xlabel='Total travel time (min)', ylabel='Cumulative fraction of reachable trips')
    axis.legend()
    save_figure(figure, output, 'travel_time_distribution.png')
    return {'requests': java.height, 'reachability': reachability, 'metrics': rows}


def timing_report(java, flash, output):
    rows = []
    for backend, data in [('java', java), ('flash', flash)]:
        for index, seconds in enumerate(data['routing_stage_seconds']):
            rows.append({'backend': backend, 'repeat': index + 1, 'seconds': seconds,
                         'trips_per_second': data['trips'] / seconds, 'scope': data['timing_scope'],
                         'workers': data.get('routing_workers'),
                         'phase': 'cold' if backend == 'flash' and index == 0 else 'warm'})
    for index, seconds in enumerate(flash.get('serial_routing_stage_seconds', [])):
        rows.append({'backend': 'flash_1_worker', 'repeat': index + 1, 'seconds': seconds,
                     'trips_per_second': flash['trips'] / seconds, 'scope': flash['timing_scope'],
                     'workers': 1, 'phase': 'cold' if index == 0 else 'warm'})
    pl.DataFrame(rows).write_csv(Path(output) / 'routing_times.csv')
    (Path(output) / 'timings.json').write_text(json.dumps({'java': java, 'flash': flash}, indent=2) + '\n')
    figure, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].bar(['Java initialization', 'FLASH preprocessing'],
                [java['startup_seconds'], flash['preprocessing']['preprocessing_seconds']])
    axes[0].set(ylabel='Seconds (log scale)', title='Recorded setup (different operations)', yscale='log')
    cold = [row for row in rows if row['phase'] == 'cold']
    axes[1].bar([f"{row['backend']}\n{row['workers']} worker(s)" for row in cold], [row['seconds'] for row in cold])
    axes[1].set(ylabel='Seconds', title='First FLASH call, including worker initialization')
    warm = [row for row in rows if row['phase'] == 'warm']
    backends = list(dict.fromkeys(row['backend'] for row in warm))
    means, deviations, labels = [], [], []
    for backend in backends:
        values = [row['seconds'] for row in warm if row['backend'] == backend]
        means.append(float(np.mean(values)))
        deviations.append(float(np.std(values)))
        workers = next(row['workers'] for row in warm if row['backend'] == backend)
        labels.append(f'{backend}\n{workers} worker(s)' if workers is not None else backend)
    axes[2].bar(labels, means, yerr=deviations, capsize=5)
    axes[2].set(ylabel='Seconds', title='Warm complete stage: mean ± SD')
    save_figure(figure, output, 'runtime.png')


def vehicle_mode(value):
    value = str(value).lower()
    if value in ('rail', 'train', 's', 'r', 're', 'ic', 'ir'): return 'rail'
    if value in ('bus', 'coach', 'postbus'): return 'bus'
    if value == 'tram': return 'tram'
    if value in ('subway', 'metro'): return 'subway'
    if value in ('ferry', 'ship', 'boat'): return 'ferry'
    return 'other'


def gtfs_mode(value):
    value = int(value)
    if value == 0 or 900 <= value < 1000: return 'tram'
    if value == 1 or 400 <= value < 500: return 'subway'
    if value == 2 or 100 <= value < 200: return 'rail'
    if value == 3 or 700 <= value < 900 or 200 <= value < 300: return 'bus'
    if value == 4 or 1000 <= value < 1100: return 'ferry'
    return 'other'


def schedule_modes(config_path, wanted):
    root = etree.parse(str(config_path))
    values = root.xpath('//module[@name="transit"]/param[@name="transitScheduleFile"]/@value')
    if not values:
        return {}
    value = unquote(urlparse(values[0]).path) if values[0].startswith('file:') else values[0]
    path = Path(value)
    if not path.is_absolute(): path = Path(config_path).parent / path
    modes, line = {}, None
    with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as stream:
        for event, element in etree.iterparse(stream, events=('start', 'end'), tag=('transitLine', 'transitRoute')):
            if event == 'start' and element.tag == 'transitLine':
                line = element.get('id')
            if event == 'end' and element.tag == 'transitRoute':
                key = (line, element.get('id'))
                if key in wanted: modes[key] = vehicle_mode(element.findtext('transportMode', 'other'))
                element.clear()
                while element.getprevious() is not None: del element.getparent()[0]
            if event == 'end' and element.tag == 'transitLine': element.clear()
    return modes


def journey_modes(java, flash, output):
    rows, pending = [], []
    with open(java['itineraries'], newline='') as stream:
        for record in csv.DictReader(stream):
            itinerary = json.loads(record.get('route') or '{}')
            for part in itinerary.get('parts', []):
                if part.get('kind') != 'leg' or part.get('mode') != 'pt': continue
                route = part.get('route') or {}
                description = route.get('description') or '{}'
                description = json.loads(description) if isinstance(description, str) else description
                key = (str(description.get('transitLineId')), str(description.get('transitRouteId')))
                pending.append((key, {'backend': 'java', 'trip_id': record['identifier'],
                                     'distance_km': float(route.get('distance') or 0) / 1000}))
    modes = schedule_modes(java['config_path'], {key for key, _ in pending})
    rows.extend({**row, 'mode': modes.get(key, 'other')} for key, row in pending)
    for path in flash['legs']:
        legs = pl.read_parquet(path).filter(pl.col('vehicle') == 1)
        for row in legs.select('trip_id', 'distance_m', 'gtfs_route_type').iter_rows(named=True):
            rows.append({'backend': 'flash', 'trip_id': row['trip_id'], 'distance_km': row['distance_m'] / 1000,
                         'mode': gtfs_mode(row['gtfs_route_type'])})
    if not rows: return
    legs = pl.DataFrame(rows)
    legs.write_parquet(Path(output) / 'selected_vehicle_legs.parquet')
    totals = legs.group_by('backend', 'mode').agg(pl.len().alias('boardings'), pl.col('distance_km').sum(),
                                                pl.col('trip_id').n_unique().alias('trips')).sort('mode', 'backend')
    totals.write_csv(Path(output) / 'vehicle_modes.csv')
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = sorted(totals['mode'].unique().to_list())
    for offset, backend in [(-.2, 'java'), (.2, 'flash')]:
        lookup = {row['mode']: row for row in totals.filter(pl.col('backend') == backend).iter_rows(named=True)}
        for axis, column in zip(axes, ['boardings', 'distance_km']):
            axis.bar(np.arange(len(labels)) + offset, [lookup.get(label, {}).get(column, 0) for label in labels], .4, label=backend)
            axis.set_xticks(np.arange(len(labels)), labels)
            axis.set(ylabel=column.replace('_', ' '), title='Selected journeys in sampled requests')
            axis.legend()
    save_figure(figure, output, 'vehicle_modes.png')


def service_departures(feed, date, output):
    """Count scheduled vehicle trips by first departure, not stop calls or sampled boardings."""
    feed, date = Path(feed), str(date).replace('-', '')
    active = set()
    calendar = feed / 'calendar.txt'
    if calendar.exists():
        frame = pl.read_csv(calendar, infer_schema=False)
        weekday = datetime.datetime.strptime(date, '%Y%m%d').strftime('%A').lower()
        active.update(frame.filter((pl.col('start_date') <= date) & (pl.col('end_date') >= date)
                                  & (pl.col(weekday) == '1'))['service_id'].to_list())
    exceptions = feed / 'calendar_dates.txt'
    if exceptions.exists():
        for row in pl.read_csv(exceptions, infer_schema=False).filter(pl.col('date') == date).iter_rows(named=True):
            if row['exception_type'] == '1': active.add(row['service_id'])
            else: active.discard(row['service_id'])
    trips = pl.read_csv(feed / 'trips.txt', infer_schema=False).filter(pl.col('service_id').is_in(sorted(active)))
    routes = pl.read_csv(feed / 'routes.txt', infer_schema=False).select('route_id', 'route_type')
    first = pl.scan_csv(feed / 'stop_times.txt', infer_schema=False).select('trip_id', 'stop_sequence', 'departure_time').with_columns(
        pl.col('stop_sequence').cast(pl.Int64)).sort('trip_id', 'stop_sequence').group_by('trip_id').agg(
        pl.col('departure_time').first()).collect(engine='streaming')
    departures = trips.select('trip_id', 'route_id').join(first, on='trip_id', validate='1:1').join(routes, on='route_id', validate='m:1')
    frequency_file = feed / 'frequencies.txt'
    if frequency_file.exists() and frequency_file.stat().st_size:
        frequencies = pl.read_csv(frequency_file, infer_schema=False)
        if frequencies.height:
            departures = departures.join(frequencies.select('trip_id').unique(), on='trip_id', how='anti')
            # Headway services are expected departures; exact_times=0 is not a fixed timetable.
            expanded = []
            lookup = {row['trip_id']: row for row in trips.join(routes, on='route_id').iter_rows(named=True)}
            for row in frequencies.iter_rows(named=True):
                if row['trip_id'] not in lookup: continue
                start = sum(int(x) * unit for x, unit in zip(row['start_time'].split(':'), [3600, 60, 1]))
                end = sum(int(x) * unit for x, unit in zip(row['end_time'].split(':'), [3600, 60, 1]))
                headway = int(row['headway_secs'])
                if headway <= 0: raise ValueError('GTFS frequency headway must be positive')
                expanded.extend((gtfs_mode(lookup[row['trip_id']]['route_type']), second // 3600) for second in range(start, end, headway))
        else: expanded = []
    else: expanded = []
    records = [(gtfs_mode(row['route_type']), int(row['departure_time'].split(':')[0]))
               for row in departures.iter_rows(named=True) if row['departure_time']]
    records.extend(expanded)
    totals = pl.DataFrame(records, schema=['mode', 'hour'], orient='row').group_by('mode', 'hour').len().sort('mode', 'hour')
    totals.write_csv(Path(output) / 'scheduled_departures.csv')
    figure, axis = plt.subplots(figsize=(10, 4))
    for mode in sorted(totals['mode'].unique().to_list()):
        values = totals.filter(pl.col('mode') == mode)
        axis.plot(values['hour'], values['len'], label=mode, marker='.')
    axis.set(xlabel='Hour of service day (hours beyond 24 retained)', ylabel='Vehicle trips starting',
             title=f'Full GTFS scheduled/expected departures on {date}')
    if not totals.is_empty(): axis.legend()
    save_figure(figure, output, 'scheduled_departures.png')


def write_index(output, summary, warnings):
    output = Path(output)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    notes = [
        'Both backends use the exact same seeded population requests and Python fare model.',
        'Total journey time includes initial waiting. DMC components exclude initial waiting.',
        'Java uses its configured routing disutility; FLASH selects earliest arrival, then fewer transfers.',
        'FLASH distances sum straight-line stop-to-stop segments; Java uses its transit network. Distance differences are not necessarily routing errors.',
        'Mode kilometres are passenger-leg kilometres in this request sample, not fleet vehicle kilometres.',
        'GTFS departures describe the full service day, not router-specific frequencies. Headway services are expected departures.',
        'FLASH repeats retain one native worker pool: first call includes native initialization, subsequent calls measure warm routing. Java server initialization is measured separately.',
        'FLASH routes independent stop-pair requests in parallel using flash_tb_query_threads (defaults to threads), sharing one timetable and keeping private search state per worker.',
        'Unreachable routes are excluded from paired utility-variable statistics and reported separately.',
    ]
    pictures = ''.join(f'<h2>{html.escape(path.stem.replace("_", " "))}</h2><img src="{path.name}" style="max-width:100%">'
                       for path in sorted(output.glob('*.png')) if not path.name.startswith('flash_only_'))
    links = ''.join(f'<li><a href="{path.name}">{path.name}</a></li>' for path in sorted(output.iterdir())
                    if path.suffix in ('.csv', '.json', '.parquet'))
    page = '<!doctype html><meta charset="utf-8"><title>PT router comparison</title><body style="max-width:1200px;margin:2rem auto;font-family:sans-serif">'
    page += '<h1>Java / FLASH PT router comparison</h1><ul>' + ''.join(f'<li>{html.escape(note)}</li>' for note in notes) + '</ul>'
    if warnings: page += '<h2>Unavailable diagnostics</h2><ul>' + ''.join(f'<li>{html.escape(w)}</li>' for w in warnings) + '</ul>'
    page += pictures + '<h2>Saved data</h2><ul>' + links + '</ul>'
    if 'flash_only_diagnostics' in summary:
        from analysis.pt_routers.unmatched import render
        page += render(output, summary['flash_only_diagnostics'])
    if 'java_raptor_diagnosis' in summary:
        from analysis.pt_routers.raptor_diagnosis import render as render_diagnosis
        page += render_diagnosis(output, summary['java_raptor_diagnosis'])
    page += '</body>'
    (output / 'index.html').write_text(page)
