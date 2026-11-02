"""Compare the production Java and FLASH PT routers on identical synthetic trips."""
from pathlib import Path
import json
import polars as pl

from analysis.pt_routers import report, unmatched


def configure(context):
    context.stage('analysis.pt_routers.inputs')
    context.stage('analysis.pt_routers.java')
    context.stage('analysis.pt_routers.flash')
    context.config('output_path')
    context.config('pt_router_analysis_output_folder', 'pt_routers')
    context.config('pt_router_analysis_frequencies', True)
    context.config('pt_router_analysis_source_gtfs', '')
    if not context.config('pt_router_analysis_source_gtfs'):
        context.stage('data.gtfs.cleaned')
    context.config('gtfs_date', 'dayWithMostServices')
    context.config('flash_tb_date', '')
    context.config('random_seed')
    context.config('threads', volatile=True)
    context.config('flash_tb_walk_speed_m_per_s', 1.3)
    context.config('flash_tb_walk_detour_factor', 1.3)


def execute(context):

    # Compare complete trip-keyed outputs, including failed routes.

    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    java, flash = context.stage('analysis.pt_routers.java'), context.stage('analysis.pt_routers.flash')
    java_routes, flash_routes = pl.read_parquet(java['routes']), pl.read_parquet(flash['routes'])
    summary = report.compare(java_routes, flash_routes, output)
    summary.update(seed=context.config('random_seed'), threads=context.config('threads'), pricing='python',
                   date=str(context.config('flash_tb_date') or context.config('gtfs_date')))
    summary.update(flash_routing_workers=flash['timing'].get('routing_workers'),
                   java_routing_workers=java['timing'].get('routing_workers'),
                   serial_consistency_verified=flash['timing'].get('serial_consistency_verified', False))
    report.single_overview(flash_routes, output, 'flash')
    report.timing_report(java['timing'], flash['timing'], output)

    # Extract itinerary modes and scheduled departures when the source supports them.

    warnings = []
    diagnostics = [('vehicle modes', lambda: report.journey_modes(java, flash, output))]
    if context.config('pt_router_analysis_frequencies'):
        feed = context.config('pt_router_analysis_source_gtfs') or str(Path(context.path('data.gtfs.cleaned')) / 'output')
        diagnostics.append(('scheduled departures', lambda: report.service_departures(feed, summary['date'], output)))
    for name, diagnostic in diagnostics:
        try:
            diagnostic()
        except (ValueError, KeyError, OSError, TypeError, pl.exceptions.PolarsError) as error:
            warnings.append(f'{name}: {type(error).__name__}: {error}')
    summary['warnings'] = warnings

    # Diagnose FLASH-only results and preserve reproducible origin–destination examples.

    summary['flash_only_diagnostics'] = unmatched.analyse(
        output, context.stage('analysis.pt_routers.inputs'), context.config('random_seed'),
        java['itineraries'], flash['legs'], context.config('flash_tb_walk_speed_m_per_s'),
        context.config('flash_tb_walk_detour_factor'))

    # Save an HTML entry point alongside figures and machine-readable results.

    report.write_index(output, summary, warnings)
    (output / 'progress.json').write_text(json.dumps({'phase': 'complete', 'report': str(output / 'index.html')}) + '\n')
    print(f'PT router comparison: {output / "index.html"}')
    return str(output / 'index.html')
