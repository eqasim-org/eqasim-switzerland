"""Experimental mode-choice integration; fares and itinerary limitations are explicit."""
from pathlib import Path
import polars as pl

from mode_choice.routing.flash.router import Router
from mode_choice.routing import options


def configure(context):
    options.configure(context)
    context.stage('mode_choice.data.prepare')
    context.stage('mode_choice.routing.flash.timetable')
    context.config('flash_tb_access_stops', 4)
    context.config('flash_tb_query_batch_size', 10000)
    context.config('flash_tb_save_journeys', False)
    context.config('threads', volatile=True)
    context.config('flash_tb_query_threads', None)
    query_threads(context)
    context.config('flash_tb_max_access_distance_m', 1500.)
    context.config('flash_tb_walk_speed_m_per_s', 1.3)
    context.config('flash_tb_walk_detour_factor', 1.3)
    if context.config('route_population_once') or context.config('prepare_sumo'):
        raise ValueError('FLASH-TB cannot yet export MATSim/SUMO itineraries; disable route_population_once and prepare_sumo')


def execute(context):

    # Route all alternatives with FLASH-TB and retain the native legs for inspection.

    with create_router(context) as router:
        return route_trips(context, router)


def create_router(context):
    return Router(context.stage('mode_choice.routing.flash.timetable'), context.config('flash_tb_access_stops'),
                  context.config('flash_tb_max_access_distance_m'), context.config('flash_tb_walk_speed_m_per_s'),
                  context.config('flash_tb_walk_detour_factor'), keep_process=True,
                  save_selected_legs=context.config('flash_tb_save_journeys'), threads=query_threads(context))


def route_trips(context, router):
    """Use a caller-owned warm engine, including across benchmark repetitions."""
    trips = context.stage('mode_choice.data.prepare')
    size = context.config('flash_tb_query_batch_size')
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError('flash_tb_query_batch_size must be a positive integer')
    outputs = []
    router.batch_timings = []
    for index, batch in enumerate(trips.iter_slices(size)):
        outputs.append(router.route(batch, context.path()))
        router.batch_timings.append(dict(router.timings))
        if context.config('flash_tb_save_journeys') and not router.selected_legs.is_empty():
            router.selected_legs.write_parquet(Path(context.path()) / f'selected_legs_batch_{index}.parquet')
    result = pl.concat(outputs) if outputs else router.route(trips, context.path())

    # Return routing variables; the PT stage selects pricing independently.

    return result.drop('pt_initial_waiting_time_min')


def query_threads(context):
    threads = context.config('flash_tb_query_threads')
    if threads is None: threads = context.config('threads')
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError('flash_tb_query_threads must be a positive integer or null (use threads)')
    return threads
