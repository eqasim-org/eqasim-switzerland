"""Use the shared FLASH timetable after Java has completed and released its memory."""
from pathlib import Path
import json

from mode_choice.routing.flash import timetable


def configure(context):
    context.stage('analysis.pt_routers.java')
    timetable.configure(context)
    context.config('output_path')
    context.config('pt_router_analysis_output_folder', 'pt_routers')


def execute(context):

    # Keep the full configured feed and share its preprocessing with production routing.

    output = Path(context.config('output_path')) / context.config('pt_router_analysis_output_folder')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'progress.json').write_text(json.dumps({'phase': 'flash_preprocessing',
                                                    'cache': str(timetable.cache_directory(context))}) + '\n')
    feed = Path(context.path('data.gtfs.cleaned')) / 'output'
    result = timetable.prepare_shared(context, feed)

    # Record original preprocessing time separately from this run's cache lookup.

    (output / 'flash_preprocessing.json').write_text(json.dumps(result['preprocessing'], indent=2) + '\n')
    (output / 'progress.json').write_text(json.dumps({'phase': 'flash_timetable_ready',
                                                    'reused': result['preprocessing']['reused']}) + '\n')
    return result


def validate(context):
    return timetable.validate(context)
