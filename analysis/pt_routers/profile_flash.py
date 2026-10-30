"""Audit day filtering and separate native FLASH work from Python and startup costs."""
from pathlib import Path
import cProfile
import datetime
import html
import io
import json
import logging
import os
import platform
import pstats
import re
import time

import polars as pl

from analysis.pt_routers import java
from analysis.pt_routers.context import ConfigureView, ExecuteView
from mode_choice.routing.flash import pt as flash_pt
from mode_choice.routing.flash.router import Router


def configure(context):
    context.stage('analysis.pt_routers.inputs')
    context.stage('analysis.pt_routers.timetable')
    context.stage('analysis.pt_routers.java')
    flash_pt.configure(ConfigureView(context, {'mode_choice.data.prepare': 'analysis.pt_routers.inputs',
                                             'mode_choice.routing.flash.timetable': 'analysis.pt_routers.timetable'}))
    java.configure(context)
    context.config('pt_router_analysis_source_gtfs', '')
    if not context.config('pt_router_analysis_source_gtfs'):
        context.stage('data.gtfs.cleaned')
    context.config('pt_router_profile_include_tb', True)
    context.config('flash_tb_preprocessing_threads', 1)
    context.config('gtfs_date', 'dayWithMostServices')
    context.config('flash_tb_date', '')


def execute(context):

    # Remeasure Java on this allocation, without overwriting the original comparison.

    directory = Path(context.path())
    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    result = {'host': platform.node(), 'allocated_cpus': len(os.sched_getaffinity(0)),
              'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK'), 'polars_threads': pl.thread_pool_size(),
              'native_query_threads': 1, 'preprocessing_threads': context.config('flash_tb_preprocessing_threads'),
              'java_routing_workers': context.config('threads'),
              'scope': 'FLASH raw routing under cProfile; native loop includes journey reconstruction, distance and TSV export; Java uses the complete production stage on this same node'}
    measured = context.stage('analysis.pt_routers.java')['timing']
    same_job = os.environ.get('SLURM_JOB_ID') is not None and measured.get('slurm_job_id') == os.environ['SLURM_JOB_ID']
    if same_job and measured.get('host') == result['host']:
        result['java'] = measured
    else:
        result['java'] = java.execute(ExecuteView(context, {}, directory=directory / 'java'))['timing']

    # Time first and subsequent calls to the same persistent native process.

    trips = context.stage('analysis.pt_routers.inputs')
    table = context.stage('analysis.pt_routers.timetable')
    records = []
    for access_stops in sorted({1, context.config('flash_tb_access_stops')}, reverse=True):
        started = time.perf_counter()
        with Router(table, access_stops, context.config('flash_tb_max_access_distance_m'),
                    context.config('flash_tb_walk_speed_m_per_s'), context.config('flash_tb_walk_detour_factor'),
                    keep_process=True, save_selected_legs=True) as router:
            setup = time.perf_counter() - started
            for repeat in range(context.config('pt_router_analysis_repeats')):
                profile = cProfile.Profile()
                profile.enable()
                routes = router.route(trips, directory)
                profile.disable()
                router.log.seek(0)
                native_log = router.log.read()
                router.log.seek(0, 2)
                timings = [float(value) for value in re.findall(r'QUERY_SECONDS=([0-9.eE+-]+)', native_log)]
                record = {'algorithm': 'flash', 'access_stops': access_stops, 'repeat': repeat + 1,
                          'python_router_setup_seconds': setup, **router.timings,
                          'native_loop_and_export_seconds': timings[-1] if timings else None,
                          'journey_legs_exported': router.journeys.height,
                          'selected_legs': router.selected_legs.height,
                          'reachable': routes.filter(pl.col('pt_total_travel_time_min') < 1e9).height}
                record['python_seconds'] = record['total_seconds'] - record.get('native_process_seconds', 0.)
                record['native_startup_and_ipc_seconds'] = (record.get('native_process_seconds', 0.) - timings[-1]) if timings else None
                records.append(record)
                stem = f'flash_k{access_stops}_repeat{repeat + 1}'
                routes.write_parquet(output / (stem + '.parquet'))
                profile.dump_stats(str(output / (stem + '.pstats')))
                stream = io.StringIO()
                pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(30)
                (output / (stem + '_python_profile.txt')).write_text(stream.getvalue())
                logging.getLogger('synpp').info('FLASH profile: %s', json.dumps(record))

            # Use the same queries and timetable to compare arc flags with unpruned TB.

            if context.config('pt_router_profile_include_tb') and access_stops == context.config('flash_tb_access_stops'):
                router.route(trips, directory, algorithm='tb')
                records.append({'algorithm': 'tb', 'access_stops': access_stops,
                                'repeat': 1, **router.timings})
    result['flash_cases'] = records
    (output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')

    # Independently count the source GTFS services and events for the requested day.

    feed = context.config('pt_router_analysis_source_gtfs') or str(Path(context.path('data.gtfs.cleaned')) / 'output')
    date = str(context.config('flash_tb_date') or context.config('gtfs_date')).replace('-', '')
    result['gtfs'] = audit_day(feed, date)
    (output / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    write_report(result, output)
    print(f'PT performance audit: {output / "profile.json"}')
    return str(output / 'profile.json')


def audit_day(feed, date):
    feed = Path(feed)
    active = set()
    if (feed / 'calendar.txt').exists():
        weekday = datetime.datetime.strptime(date, '%Y%m%d').strftime('%A').lower()
        calendars = pl.read_csv(feed / 'calendar.txt', infer_schema=False)
        active.update(calendars.filter((pl.col('start_date') <= date) & (pl.col('end_date') >= date)
                                       & (pl.col(weekday) == '1'))['service_id'].to_list())
    if (feed / 'calendar_dates.txt').exists():
        exceptions = pl.scan_csv(feed / 'calendar_dates.txt', infer_schema=False).filter(pl.col('date') == date).collect(engine='streaming')
        for row in exceptions.iter_rows(named=True):
            if row['exception_type'] == '1': active.add(row['service_id'])
            else: active.discard(row['service_id'])
    trips = pl.scan_csv(feed / 'trips.txt', infer_schema=False).select('trip_id', 'service_id')
    all_trips = trips.select(pl.len()).collect(engine='streaming').item()
    selected = trips.filter(pl.col('service_id').is_in(sorted(active))).select('trip_id').collect(engine='streaming')
    events = pl.scan_csv(feed / 'stop_times.txt', infer_schema=False).select('trip_id')
    all_events = events.select(pl.len()).collect(engine='streaming').item()
    day_events = events.join(selected.lazy(), on='trip_id', how='semi').select(pl.len()).collect(engine='streaming').item()
    return {'date': date, 'active_services': len(active), 'source_trips_all_dates': all_trips,
            'trips_on_requested_day': selected.height, 'source_stop_events_all_dates': all_events,
            'stop_events_on_requested_day_before_native_deduplication': day_events}


def write_report(result, output):
    """Keep raw-routing and complete-stage timings explicitly distinguished."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    flash = [row for row in result['flash_cases'] if row['algorithm'] == 'flash']
    access_stops = max(row['access_stops'] for row in flash)
    cases = [row for row in flash if row['access_stops'] == access_stops]
    figure, axis = plt.subplots(figsize=(10, 5))
    labels = [f"FLASH #{row['repeat']}\n{'First native call' if row['repeat'] == 1 else 'Native process reused'}" for row in cases]
    bottoms = [0.] * len(cases)
    for column, label in [('native_startup_and_ipc_seconds', 'Native startup / IPC'),
                          ('native_loop_and_export_seconds', 'Native search, reconstruction and TSV'),
                          ('python_seconds', 'Python request preparation and aggregation')]:
        values = [row[column] for row in cases]
        axis.bar(labels, values, bottom=bottoms, label=label)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    for index, total in enumerate(bottoms): axis.text(index, total + .1, f'{total:.2f}s', ha='center')
    axis.set(ylabel=f"Seconds for {cases[0]['input_trips']:,} requests", title=f'FLASH raw routing: {access_stops} candidate stops per endpoint')
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / 'flash_cost_breakdown.png', dpi=150)
    plt.close(figure)

    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
        f"FLASH, {row['access_stops']} access/egress candidates", row['repeat'],
        f"{row['total_seconds']:.3f}", row['native_queries'], row['reachable'])) + '</tr>' for row in flash)
    gtfs = ''.join(f'<li>{html.escape(key)}: {html.escape(str(value))}</li>' for key, value in result['gtfs'].items())
    date = html.escape(result['gtfs']['date'])
    java_workers = result.get('java_routing_workers', result['allocated_cpus'])
    java_times = ', '.join(f'{value:.3f}s' for value in result['java']['routing_stage_seconds'])
    page = f'''<!doctype html><meta charset="utf-8"><title>PT routing performance audit</title>
<body style="max-width:1100px;margin:2rem auto;font-family:sans-serif">
<h1>PT routing performance audit</h1>
<p>Java and FLASH were measured on {html.escape(result['host'])}, with {result['allocated_cpus']} allocated CPUs.
Java uses {java_workers} routing workers; FLASH uses {result['native_query_threads']} query worker and
{result['preprocessing_threads']} preprocessing workers.</p>
<p>Java complete production-stage times, excluding server initialization: {java_times}.
FLASH numbers below are raw routing under cProfile, excluding fares and the selected-leg parquet export.
The native loop includes journey reconstruction, distances and TSV export; it is not the search kernel alone.</p>
<img src="flash_cost_breakdown.png" style="max-width:100%">
<table border="1" cellpadding="8"><tr><th>Case</th><th>Call</th><th>Raw routing seconds</th><th>Native stop-pair queries</th><th>Reachable requests</th></tr>{rows}</table>
<p>The first FLASH call includes binary deserialization and native query-index construction. Subsequent calls reuse that process.
Reducing access candidates changes reachability, so the one-stop case is diagnostic, not an equivalent faster configuration.</p>
<h2>Independent GTFS service-day check</h2><ul>{gtfs}</ul>
<p>The native preprocessing uses the same inclusive start/end date, {date}, applying days of operation.
Source files contain many service dates; the routed timetable contains only the selected day's service.
Services continuing after midnight retain their GTFS times above 24:00.</p>
<p><a href="profile.json">Detailed timings and counts</a></p></body>'''
    (output / 'index.html').write_text(page)
