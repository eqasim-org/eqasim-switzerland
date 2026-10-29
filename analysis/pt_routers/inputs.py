"""One seeded, identically ordered sample for both production PT backends."""
from pathlib import Path
import polars as pl
import pickle
import shutil


def configure(context):
    context.config('pt_router_analysis_source_inputs', '')
    if not context.config('pt_router_analysis_source_inputs'):
        context.stage('mode_choice.data.prepare')
    context.config('random_seed')
    context.config('pt_router_analysis_sample_size', 2000)


def execute(context):

    # Take a stable sample from the actual synthetic population's eligible trips.

    source = context.config('pt_router_analysis_source_inputs')
    if source:
        snapshot = Path(context.path()) / 'source_inputs.p'
        shutil.copyfile(source, snapshot)
        with snapshot.open('rb') as stream:
            trips = pickle.load(stream)
    else:
        trips = context.stage('mode_choice.data.prepare')
    trips = trips if isinstance(trips, pl.DataFrame) else pl.from_pandas(trips)
    trips = trips.sort('person_id', 'trip_index')
    size = context.config('pt_router_analysis_sample_size')
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError('pt_router_analysis_sample_size must be a positive integer')
    if trips.is_empty():
        raise ValueError('No eligible trips for the PT router comparison')
    trips = trips.sample(n=min(size, trips.height), seed=context.config('random_seed')).sort('person_id', 'trip_index')

    # Persist exact coordinates, times and fare attributes for reproducibility.

    trips.write_parquet(Path(context.path()) / 'requests.parquet')
    return trips


def validate(context):
    source = context.config('pt_router_analysis_source_inputs')
    if source:
        stat = Path(source).stat()
        return [str(source), stat.st_mtime_ns, stat.st_size]
