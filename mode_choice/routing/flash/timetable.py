"""Shared date-specific FLASH preprocessing, independent of population and DMC fitting."""
from pathlib import Path
import datetime

from data.gtfs.sources import source_signatures
from mode_choice.routing.flash import cache


def configure(context):
    context.stage('mode_choice.routing.flash.runtime')
    context.stage('data.gtfs.cleaned')
    context.config('data_path')
    context.config('gtfs_folder', 'gtfs')
    context.config('gtfs_name', '')
    context.config('gtfs_tpg_file', '')
    context.config('gtfs_date', 'dayWithMostServices')
    context.config('flash_tb_date', '')
    context.config('flash_tb_preprocessing_threads', 1)
    context.config('flash_tb_partition_cells', 32)


def execute(context):

    # Resolve the extracted feed and the source archives that identify its contents.

    feed = Path(context.path('data.gtfs.cleaned')) / 'output'

    # Reuse preprocessing across stages, populations and working directories.

    return prepare_shared(context, feed)


def validate(context):
    return [source_signatures(context), cache.validation_token(cache_directory(context))]


def cache_directory(context):
    return Path(context.config('data_path')) / 'gtfs' / 'flash_tb'


def prepare_shared(context, feed):
    return prepare(context.stage('mode_choice.routing.flash.runtime'), feed, cache_directory(context),
                   context.config('flash_tb_date') or context.config('gtfs_date'),
                   context.config('flash_tb_preprocessing_threads'), context.config('flash_tb_partition_cells'),
                   sources=source_signatures(context))


def prepare(binary, feed, directory, date, threads=1, cells=32, *, sources=None):
    """Use the common cache implementation; standalone fixtures supply a private directory."""
    date = str(date).replace('-', '')
    if len(date) != 8 or not date.isdigit():
        raise ValueError('Set flash_tb_date to an explicit YYYYMMDD date when gtfs_date is automatic')
    datetime.datetime.strptime(date, '%Y%m%d')
    for name, value in [('threads', threads), ('cells', cells)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < (2 if name == 'cells' else 1):
            raise ValueError(f'Invalid FLASH-TB {name}: {value}')
    return cache.ensure(binary, feed, directory, date, threads, cells, sources)
