"""Survey PT reachability, journey metrics, runtime and paired Google comparisons."""
from pathlib import Path
import json
import shutil
import polars as pl
from matsim.runtime.eqasim import DEFAULT_EQASIM_COMMIT

from analysis.pt_routers.microcensus import report

BACKENDS = ('rust', 'raptor', 'flash', 'google')


def configure(context):
    context.stage('analysis.pt_routers.microcensus.inputs')
    for backend in BACKENDS:
        context.stage('analysis.pt_routers.microcensus.' + backend)
    context.config('output_path')
    context.config('pt_microcensus_output_folder', 'pt_microcensus')
    context.config('gtfs_date', 'dayWithMostServices')
    context.config('flash_tb_date', '')
    context.config('pt_router_analysis_eqasim_commit', DEFAULT_EQASIM_COMMIT)


def execute(context):

    # Export copies, preserving resumable Synpp results for every router.

    output = Path(context.config('output_path')) / context.config('pt_microcensus_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    trips = context.stage('analysis.pt_routers.microcensus.inputs')
    trips.write_parquet(output / 'requests.parquet')
    selection = Path(context.path('analysis.pt_routers.microcensus.inputs')) / 'selection.json'
    shutil.copy2(selection, output / 'selection.json')
    routes, timings = {}, {}
    for backend in BACKENDS:
        result = context.stage('analysis.pt_routers.microcensus.' + backend)
        shutil.copy2(result['routes'], output / (backend + '_raw_routes.parquet'))
        routes[backend] = report.classify(trips, pl.read_parquet(result['routes']), backend)
        routes[backend].write_parquet(output / (backend + '_routes.parquet'))
        timings[backend] = result['timing']
    metadata = {'service_date': context.config('flash_tb_date') or context.config('gtfs_date'),
                'eqasim_commit': context.config('pt_router_analysis_eqasim_commit'),
                'selection': json.loads(selection.read_text())}

    # Compare identical trips and expose missing-reference coverage separately from routing failure.

    return report.generate(trips, routes, timings, metadata, output)


def validate(context):
    import hashlib
    helper = hashlib.sha256(Path(report.__file__).read_bytes()).hexdigest()
    target = Path(context.config('output_path')) / context.config('pt_microcensus_output_folder') / 'index.html'
    return [helper, [target.stat().st_mtime_ns, target.stat().st_size] if target.is_file() else None]
