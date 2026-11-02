"""Run production FLASH routing on the same sample, with selected-journey diagnostics."""
from pathlib import Path
import json
import hashlib
import shutil
import time
from mode_choice.routing import options
from mode_choice.routing.flash import pt as flash_pt
from mode_choice.routing.flash import router
from mode_choice.routing.python import prices
from mode_choice.routing_variables import pt
from analysis.pt_routers.context import ConfigureView, ExecuteView


def configure(context):
    context.stage('analysis.pt_routers.inputs')
    options.configure_pt(context)
    flash_pt.configure(ConfigureView(context, {'mode_choice.data.prepare': 'analysis.pt_routers.inputs',
                                             'mode_choice.routing.flash.timetable': 'analysis.pt_routers.timetable'}))
    prices.configure(context)
    context.config('output_path')
    context.config('pt_router_analysis_output_folder', 'pt_routers')
    context.config('pt_router_analysis_repeats', 2)
    context.config('pt_router_analysis_verify_serial', True)


def execute(context):

    # Bind production routing to the identical sampled requests and full GTFS timetable.

    trips = context.stage('analysis.pt_routers.inputs')
    timetable = context.stage('analysis.pt_routers.timetable')
    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    view = ExecuteView(context, {'mode_choice.data.prepare': trips, 'mode_choice.routing.flash.timetable': timetable},
                       {'mode_choice_pt_routing_backend': 'flash', 'mode_choice_pt_price_backend': 'python',
                        'flash_tb_save_journeys': True})
    # Retain one native worker pool across cold and warm production-stage calls.

    workers = flash_pt.query_threads(view)
    repeats = context.config('pt_router_analysis_repeats')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('pt_router_analysis_repeats must be a positive integer')
    for path in output.glob('flash_selected_legs_batch_*.parquet'):
        path.unlink()
    report, reference, saved_legs = run_repeats(view, output, workers, repeats, export=True)

    # Check full-population equality against an independent one-worker process.

    if context.config('pt_router_analysis_verify_serial'):
        serial_view = ExecuteView(view, {}, {'flash_tb_query_threads': 1})
        serial, _, _ = run_repeats(serial_view, output, 1, repeats, reference=reference, export=False)
        report['serial_routing_stage_seconds'] = serial['routing_stage_seconds']
        report['serial_native_batches'] = serial['native_batches']
        report['serial_consistency_verified'] = True
    report.update(trips=trips.height, pricing='python', preprocessing=timetable['preprocessing'],
                  routing_workers=workers, native_process_reused=True,
                  timing_scope='complete production routing and Python fares with selected-leg export; first call includes native initialization, subsequent calls reuse the worker pool')
    (output / 'flash_timing.json').write_text(json.dumps(report, indent=2) + '\n')
    return {'routes': str(output / 'flash_routes.parquet'), 'timing': report, 'legs': saved_legs}


def run_repeats(view, output, workers, repeats, reference=None, export=False):
    timings, native_batches, saved_legs = [], [], []
    started = time.perf_counter()
    with flash_pt.create_router(view) as engine:
        setup = time.perf_counter() - started
        for repeat in range(repeats):
            for path in Path(view.path()).glob('selected_legs_batch_*.parquet'):
                path.unlink()
            started = time.perf_counter()
            routes = flash_pt.route_trips(view, engine)
            result = pt.price_routes(view, routes)
            timings.append(time.perf_counter() - started)
            native_batches.append(engine.batch_timings)
            if any(batch.get('native_query_threads') != workers for batch in engine.batch_timings if batch['native_queries']):
                raise RuntimeError('FLASH used a different worker count than requested')
            if reference is not None and not reference.equals(result):
                raise RuntimeError(f'FLASH routing differs between repetitions or between 1 and {workers} workers')
            if reference is None:
                reference = result
            prefix = '' if export else 'serial_'
            result.write_parquet(Path(view.path()) / f'{prefix}routes_repeat_{repeat}.parquet')
            if export and repeat == 0:
                result.write_parquet(output / 'flash_routes.parquet')
                shutil.copyfile(Path(view.path()) / 'cost.parquet', output / 'flash_cost.parquet')
                for path in Path(view.path()).glob('selected_legs_batch_*.parquet'):
                    target = output / ('flash_' + path.name)
                    shutil.copyfile(path, target)
                    saved_legs.append(str(target))
    return {'routing_stage_seconds': timings, 'native_batches': native_batches,
            'python_router_setup_seconds': setup}, reference, saved_legs


def validate(context):
    # Synpp hashes this stage, so track the directly invoked Python helpers too.
    return [hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() for module in (router, flash_pt, pt, prices)]
