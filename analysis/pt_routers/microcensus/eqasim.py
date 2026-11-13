"""Shared experiment runner for eqasim Rust PT and SwissRailRaptor."""
from pathlib import Path
import json
import shutil
import time
import polars as pl

from analysis.pt_routers.context import ExecuteView
from matsim.runtime.java import get_memories
from mode_choice.defaults import ROUTER_SERVER_CLASS
from mode_choice.routing.eqasim import pt
from mode_choice.routing.eqasim.client import RoutingServer
from mode_choice.routing.eqasim.runtime import validate_runtime


def configure(context):
    context.stage('analysis.pt_routers.microcensus.inputs')
    context.stage('analysis.pt_routers.microcensus.scenario')
    context.stage('analysis.pt_routers.runtime')
    context.stage('matsim.runtime.java')
    context.config('threads', volatile=True)
    context.config('java_memory', '50G', volatile=True)
    context.config('pt_microcensus_repeats', 2)
    context.config('pt_microcensus_batch_size', 5000)
    context.config('pt_microcensus_timeout', 7200.)


def execute(context, rust):

    # Use one scenario and identical requests; fares and road assignment are excluded.

    trips = context.stage('analysis.pt_routers.microcensus.inputs')
    scenario = context.stage('analysis.pt_routers.microcensus.scenario')
    directory = Path(context.path())
    source = directory / 'requests.csv'
    trips.select('trip_id', 'origin_x', 'origin_y', 'destination_x', 'destination_y', 'departure_time').rename(
        {'trip_id': 'identifier'}).write_csv(source)
    settings = server_settings(context, scenario, directory, rust)
    view = ExecuteView(context, {'mode_choice.data.prepare': trips, 'mode_choice.routing.eqasim.server': settings,
                                'mode_choice.routing.eqasim.requests': str(source)},
                       {'route_population_once': False, 'use_marouter_in_mode_choice': False,
                        'mode_choice_pt_routing_backend': 'java', 'mode_choice_pt_price_backend': 'python'})
    repeats = context.config('pt_microcensus_repeats')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('pt_microcensus_repeats must be a positive integer')
    timings = []
    RoutingServer.stop_server()
    try:
        started = time.perf_counter()
        RoutingServer.run_server_if_not_running(settings)
        RoutingServer.wait_until_ready('pt')
        startup = time.perf_counter() - started
        for repeat in range(repeats):
            started = time.perf_counter()
            result = pt.execute(view)
            timings.append(time.perf_counter() - started)
            if repeat == 0:
                raw = pl.read_csv(directory / 'pt_routes.csv', columns=['identifier'], schema_overrides={'identifier': pl.String})
                result = result.join(raw.rename({'identifier': 'trip_id'}).with_columns(pl.lit(True).alias('route_row_present')),
                                     on='trip_id', how='left', validate='1:1', maintain_order='left')
                result.write_parquet(directory / 'routes.parquet')
                shutil.copy2(directory / 'pt_routes.csv', directory / 'raw_routes.csv')
        timing = {'startup_seconds': startup, 'routing_seconds': timings, 'requests': trips.height,
                  'threads': context.config('threads'), 'rust_pt_router': rust,
                  'scope': 'production eqasim PT adapter including CSV and components; no fares',
                  'initialization_log': RoutingServer.log_path}
        (directory / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
        return {'routes': str(directory / 'routes.parquet'), 'timing': timing,
                'raw_routes': str(directory / 'raw_routes.csv')}
    finally:
        RoutingServer.stop_server()


def server_settings(context, scenario, directory, rust):
    jar = Path(context.path('analysis.pt_routers.runtime')) / context.stage('analysis.pt_routers.runtime')
    validate_runtime(jar)
    runtime = context.stage('matsim.runtime.java')
    java = shutil.which(runtime['binary'])
    if java is None:
        raise FileNotFoundError('Java executable is unavailable')
    xmx, xms = get_memories(context, runtime, context.config('java_memory'))
    temporary = directory / 'java_tmp'
    temporary.mkdir(exist_ok=True)
    command = [java, '-cp', str(jar), '-Xmx' + xmx, '-Xms' + xms, '-XX:+UseParallelGC',
               '-Djava.io.tmpdir=' + str(temporary), '-Dmatsim.useLocalDtds=true', '-Djava.awt.headless=true',
               ROUTER_SERVER_CLASS, '--config-path', scenario['config'], '--network-path', scenario['network'],
               '--compute-cost', 'false', '--marouter', 'false', '--marouter-iterations', '3',
               '--use-rust-pt-router', str(rust).lower(), '--export-routes', 'false',
               '--threads', str(context.config('threads')), '--batch-size', str(context.config('pt_microcensus_batch_size')),
               '--chunk-size', str(context.config('pt_microcensus_batch_size')), '--host', '127.0.0.1', '--port', '0']
    return {'command': command, 'cwd': str(directory), 'config_path': scenario['config'],
            'startup_timeout': context.config('pt_microcensus_timeout'),
            'request_timeout': context.config('pt_microcensus_timeout'), 'shutdown_timeout': 30., 'poll_interval': 1.}


def validation_token():
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
