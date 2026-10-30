"""Run production Java PT routing first, then release its memory before FLASH preprocessing."""
from pathlib import Path
import json
import shutil
import time
import pickle
import os
import platform

from mode_choice.routing import options
from mode_choice.routing.java import pt as java_pt, requests
from mode_choice.routing.java.client import RoutingServer
from mode_choice.routing.python import prices
from mode_choice.routing_variables import pt
from analysis.pt_routers.context import ConfigureView, ExecuteView, private_server
from matsim.runtime.java import get_memories


def configure(context):
    context.stage('analysis.pt_routers.inputs')
    context.stage('analysis.pt_routers.runtime')
    options.configure_pt(context)
    context.config('pt_router_analysis_source_server', '')
    ignored = ['mode_choice.routing.java.requests']
    if context.config('pt_router_analysis_source_server'):
        ignored.append('mode_choice.routing.java.server')
    java_pt.configure(ConfigureView(context, {'mode_choice.data.prepare': 'analysis.pt_routers.inputs'}, ignored=ignored))
    context.config('threads', volatile=True)
    context.config('java_memory', '50G', volatile=True)
    context.config('mode_choice_server_request_timeout', 7200.0)
    prices.configure(context)
    context.config('output_path')
    context.config('pt_router_analysis_output_folder', 'pt_routers')
    context.config('pt_router_analysis_repeats', 2)


def execute(context):

    # Own a private server launch and sample-specific request CSV.

    directory = Path(context.path())
    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    trips = context.stage('analysis.pt_routers.inputs')
    source = context.config('pt_router_analysis_source_server')
    if source:
        with open(source, 'rb') as stream:
            cached_settings = pickle.load(stream)
    else:
        cached_settings = context.stage('mode_choice.routing.java.server')
    settings = private_server(cached_settings, directory / 'server')
    settings['request_timeout'] = context.config('mode_choice_server_request_timeout')
    settings['command'][settings['command'].index('--threads') + 1] = str(context.config('threads'))
    xmx, xms = get_memories(context, {}, context.config('java_memory'))
    settings['command'] = [('-Xmx' + xmx if value.startswith('-Xmx') else '-Xms' + xms if value.startswith('-Xms') else value) for value in settings['command']]
    jar = Path(context.path('analysis.pt_routers.runtime')) / context.stage('analysis.pt_routers.runtime')
    settings['command'][settings['command'].index('-cp') + 1] = str(jar.resolve())
    view = ExecuteView(context, {'mode_choice.data.prepare': trips, 'mode_choice.routing.java.server': settings},
                       {'mode_choice_pt_routing_backend': 'java', 'mode_choice_pt_price_backend': 'python'})
    view.stages['mode_choice.routing.java.requests'] = requests.execute(view)
    trips.write_parquet(output / 'requests.parquet')
    repeats = context.config('pt_router_analysis_repeats')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('pt_router_analysis_repeats must be a positive integer')
    timings = []
    RoutingServer.stop_server()
    try:

        # Measure initialization separately from complete routing-stage invocations.

        started = time.perf_counter()
        RoutingServer.run_server_if_not_running(settings)
        RoutingServer.wait_until_ready('pt')
        startup = time.perf_counter() - started
        for repeat in range(repeats):
            started = time.perf_counter()
            result = pt.execute(view)
            timings.append(time.perf_counter() - started)
            result.write_parquet(directory / f'routes_repeat_{repeat}.parquet')
            if repeat == 0:
                result.write_parquet(output / 'java_routes.parquet')
                shutil.copyfile(directory / 'cost.parquet', output / 'java_cost.parquet')
                shutil.copyfile(directory / 'pt_routes.csv', output / 'java_itineraries.csv')

        # Save useful partial results even when later FLASH preprocessing reaches the job limit.

        report = {'startup_seconds': startup, 'routing_stage_seconds': timings, 'trips': trips.height,
                  'pricing': 'python', 'host': platform.node(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                  'routing_workers': context.config('threads'),
                  'timing_scope': 'production PT stage including CSV parsing and Python prices; server startup excluded'}
        from analysis.pt_routers.report import single_overview
        single_overview(result, output, 'java')
        (output / 'java_timing.json').write_text(json.dumps(report, indent=2) + '\n')
        return {'routes': str(output / 'java_routes.parquet'), 'timing': report,
                'itineraries': str(output / 'java_itineraries.csv'), 'config_path': settings['config_path']}
    finally:
        RoutingServer.stop_server()


def validate(context):
    source = context.config('pt_router_analysis_source_server')
    if source:
        stat = Path(source).stat()
        return [str(source), stat.st_mtime_ns, stat.st_size]
