"""Resolve GTFS sources and identify changes without reading archive contents."""
from pathlib import Path
import logging
import os

logger = logging.getLogger("synpp")


def get_input_files(base_path):
    paths = [str(path) for path in Path(base_path).glob('*') if path.suffix.lower() == '.zip']
    if not paths:
        raise RuntimeError(f'Did not find any GTFS data (.zip) in {base_path}')
    return paths


def selected_inputs(context):
    """Match the cleaning stage's last-matching-name and optional TPG selection."""
    folder = Path(context.config('data_path')) / context.config('gtfs_folder')
    files = get_input_files(folder)
    name = context.config('gtfs_name')
    if name:
        matches = [path for path in files if name in path]
        if not matches:
            raise RuntimeError('Did not find any GTFS data matching the specified name.')
        files = matches[-1:]
    tpg = context.config('gtfs_tpg_file')
    if tpg:
        tpg = str(folder / tpg)
        # A feed selected by gtfs_name that is not the Swiss feed (no 'fp' in its name, e.g. an already merged feed
        # with the TPG lines) is used as it is: the TPG feed, whose file name is set by default, is not applied
        if name and not any(is_swiss_feed(path) for path in files):
            logger.info(f'gtfs_tpg_file not applied: {Path(files[0]).name} is not a Swiss feed (no "fp" in its name)')
            return files, ''
        if not Path(tpg).is_file():
            if not name:
                raise RuntimeError(f'gtfs_tpg_file not found: {tpg}')
            logger.warning(f'gtfs_tpg_file not found: {tpg}. The TPG lines of {Path(files[0]).name} are kept.')
            return files, ''
        files = [path for path in files if Path(path).resolve() != Path(tpg).resolve()]
    return files, tpg


def is_swiss_feed(path):
    """Whether a GTFS file is the Swiss feed, whose TPG lines gtfs_tpg_file replaces: 'fp' in its file name."""
    return 'fp' in os.path.basename(path)


def file_signatures(paths):
    """JSON-serializable identity, stable across interpreters and worker counts."""
    result = []
    for path in sorted({Path(os.path.abspath(path)) for path in paths}):
        stat = path.stat()
        result.append({'path': str(path), 'name': path.name, 'mtime_ns': stat.st_mtime_ns, 'size': stat.st_size})
    if not result:
        raise ValueError('No GTFS source files were found')
    return result


def source_signatures(context):
    files, tpg = selected_inputs(context)
    return file_signatures([*files, *([tpg] if tpg else [])])
