"""Observed PT requests through the shared full-feed FLASH preprocessing."""
from pathlib import Path
import json
import time
import polars as pl
from analysis.pt_routers.context import ConfigureView, ExecuteView
from mode_choice.routing.flash import timetable, pt


def configure(context):
    context.stage('analysis.pt_routers.microcensus.raptor')
    context.stage('analysis.pt_routers.microcensus.inputs')
    context.config('pt_microcensus_source_gtfs', '')
    timetable.configure(ConfigureView(context, ignored=['data.gtfs.cleaned'] if context.config('pt_microcensus_source_gtfs') else []))
    pt.configure(ConfigureView(context, {'mode_choice.data.prepare': 'analysis.pt_routers.microcensus.inputs'},
                               ignored=['mode_choice.routing.flash.timetable']))
    context.config('pt_microcensus_repeats', 2)


def execute(context):

    # Reuse the same persistent preprocessing as population mode choice.

    feed = context.config('pt_microcensus_source_gtfs') or str(Path(context.path('data.gtfs.cleaned')) / 'output')
    prepared = timetable.prepare_shared(context, feed)
    trips = context.stage('analysis.pt_routers.microcensus.inputs')
    view = ExecuteView(context, {'mode_choice.data.prepare': trips, 'mode_choice.routing.flash.timetable': prepared})
    timings, batches = [], []
    started = time.perf_counter()
    with pt.create_router(view) as router:
        setup = time.perf_counter() - started
        for repeat in range(context.config('pt_microcensus_repeats')):
            started = time.perf_counter()
            result = pt.route_trips(view, router)
            timings.append(time.perf_counter() - started)
            batches.append(router.batch_timings)
            if repeat == 0:
                result.with_columns(pl.lit(True).alias('route_row_present')).write_parquet(Path(context.path()) / 'routes.parquet')
    timing = {'python_setup_seconds': setup, 'routing_seconds': timings, 'requests': trips.height,
              'threads': pt.query_threads(context), 'batch_timings': batches, 'preprocessing': prepared['preprocessing'],
              'scope': 'production FLASH adapter including native startup on first batch, TSV and component aggregation; no fares'}
    (Path(context.path()) / 'timing.json').write_text(json.dumps(timing, indent=2) + '\n')
    return {'routes': str(Path(context.path()) / 'routes.parquet'), 'timing': timing}


def validate(context):
    return timetable.validate(context)
