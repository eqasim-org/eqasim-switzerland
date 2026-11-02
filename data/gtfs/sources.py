"""Resolve GTFS sources and identify changes without reading archive contents."""
from pathlib import Path
import os


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
        if not Path(tpg).is_file():
            raise RuntimeError(f'gtfs_tpg_file not found: {tpg}')
        files = [path for path in files if Path(path).resolve() != Path(tpg).resolve()]
    return files, tpg


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
