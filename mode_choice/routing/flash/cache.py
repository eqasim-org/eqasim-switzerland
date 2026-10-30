"""One persistent FLASH timetable per data directory, with atomic replacement."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import logging
import shutil
import subprocess
import time

from data.gtfs.sources import file_signatures

FORMAT_VERSION = 1
TIMETABLE_SUFFIXES = ('', '.raptor', '.graph.aRCFlag', '.graph.beginOut', '.graph.coordinates',
                      '.graph.toVertex', '.graph.travelTime', '.raptor.graph.beginOut',
                      '.raptor.graph.coordinates', '.raptor.graph.toVertex', '.raptor.graph.travelTime',
                      '.stops.tsv', '.transfers.tsv')
ARTIFACTS = tuple('timetable.binary' + suffix for suffix in TIMETABLE_SUFFIXES)
logger = logging.getLogger('synpp')


def fingerprint(sources):
    payload = json.dumps(sorted(sources, key=lambda item: item['path']), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


@contextmanager
def lock(directory, exclusive=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.lock').open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_manifest(directory):
    try:
        return json.loads((Path(directory) / 'current' / 'manifest.json').read_text())
    except (FileNotFoundError, ValueError):
        return None


def complete(directory, manifest):
    if not isinstance(manifest, dict) or set(manifest.get('artifacts', {})) != set(ARTIFACTS):
        return False
    for name, size in manifest['artifacts'].items():
        path = Path(directory) / 'current' / name
        if not path.is_file() or path.stat().st_size != size or (not size and not name.endswith('.transfers.tsv')):
            return False
    return True


def validate_settings(manifest, date, cells):
    expected = {'format_version': FORMAT_VERSION, 'date': date, 'cells': cells}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('The shared FLASH timetable has different service-day, partition or format settings. '
                         'Use the same settings across consumers, or remove gtfs/flash_tb/current to explicitly rebuild it.')


def build_native(command, **kwargs):
    subprocess.run(command, **kwargs)


def result(binary, directory, manifest, reused, started):
    target = Path(directory) / 'current' / 'timetable.binary'
    return {'binary': str(binary), 'timetable': str(target), 'stops': str(target) + '.stops.tsv',
            'date': manifest['date'], 'cache_directory': str(directory), 'cache_key': manifest['cache_key'],
            'preprocessing': {'preprocessing_seconds': manifest['preprocessing_seconds'], 'date': manifest['date'],
                              'scope': 'full configured cleaned GTFS', 'reused': reused,
                              'cache_lookup_seconds': time.perf_counter() - started}}


def ensure(binary, feed, directory, date, threads, cells, sources=None, builder=None):
    """Reuse by GTFS identity; publish a complete new bundle before deleting the old one.

    Pipeline callers supply archive signatures. Standalone fixtures use extracted
    text-file signatures and their own cache directory. The query executable and
    thread counts are deliberately absent from the data identity.
    """
    started = time.perf_counter()
    feed, directory = Path(feed).resolve(), Path(directory).resolve()
    sources = sources if sources is not None else file_signatures(feed.glob('*.txt'))
    sources = sorted(sources, key=lambda item: item['path'])
    key = fingerprint(sources)

    # Fast readers can reuse the cache concurrently with active routing workers.

    with lock(directory):
        manifest = read_manifest(directory)
        if (not (directory / '.previous').exists() and not (directory / '.building').exists()
                and manifest and manifest.get('cache_key') == key and complete(directory, manifest)):
            validate_settings(manifest, date, cells)
            logger.info('Reusing shared FLASH preprocessing: %s', directory / 'current')
            return result(binary, directory, manifest, True, started)

    # Only one process may build or replace the shared data at a time.

    with lock(directory, exclusive=True):
        current, building, previous = (directory / name for name in ('current', '.building', '.previous'))
        if previous.exists() and not current.exists():
            previous.rename(current)
        if previous.exists():
            shutil.rmtree(previous)
        if building.exists():
            shutil.rmtree(building)
        manifest = read_manifest(directory)
        if manifest and manifest.get('cache_key') == key and complete(directory, manifest):
            validate_settings(manifest, date, cells)
            return result(binary, directory, manifest, True, started)
        logger.info('Preparing shared FLASH timetable: %s', current)
        building.mkdir()
        try:
            build_started = time.perf_counter()
            if builder is None:
                with (building / 'preprocessing.log').open('w') as log:
                    build_native([str(binary), 'prepare', str(feed), date, str(building / 'timetable.binary'),
                                    str(threads), str(cells)], check=True, stdout=log, stderr=log)
                seconds = time.perf_counter() - build_started
            else:
                seconds = builder(building)
            sizes = {}
            for name in ARTIFACTS:
                path = building / name
                if not path.is_file() or (not path.stat().st_size and not name.endswith('.transfers.tsv')):
                    raise RuntimeError(f'Missing or empty FLASH timetable artifact: {path}')
                sizes[name] = path.stat().st_size
            if file_signatures(item['path'] for item in sources) != sources:
                raise RuntimeError('GTFS sources changed during FLASH preprocessing; retry with the updated input')
            manifest = {'format_version': FORMAT_VERSION, 'cache_key': key, 'sources': sources,
                        'date': date, 'cells': cells, 'preprocessing_seconds': seconds, 'artifacts': sizes}
            (building / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            if current.exists():
                current.rename(previous)
            building.rename(current)
            if previous.exists():
                shutil.rmtree(previous)
        except BaseException:
            if not current.exists() and previous.exists():
                previous.rename(current)
            if building.exists():
                shutil.rmtree(building)
            raise
        return result(binary, directory, manifest, False, started)


@contextmanager
def reader(timetable):
    """Keep a shared bundle alive until its Python/native reader has closed."""
    if 'cache_directory' not in timetable:
        yield
        return
    with lock(timetable['cache_directory']):
        manifest = read_manifest(timetable['cache_directory'])
        if not manifest or manifest.get('cache_key') != timetable['cache_key'] or manifest.get('date') != timetable['date']:
            raise RuntimeError('The shared FLASH timetable was replaced since this stage ran; rerun its timetable stage')
        if not complete(timetable['cache_directory'], manifest):
            raise RuntimeError('The shared FLASH timetable is incomplete; rerun its timetable stage')
        yield


def validation_token(directory):
    """Invalidate Synpp's stage result if the shared bundle is removed or replaced."""
    directory = Path(directory)
    if not directory.exists():
        return None
    with lock(directory):
        manifest = read_manifest(directory)
        if not manifest or not complete(directory, manifest):
            return None
        stat = (directory / 'current' / 'manifest.json').stat()
        return [manifest['cache_key'], stat.st_mtime_ns, stat.st_size]


def import_prepared(binary, prepared, directory, sources, cells=32):
    """Explicit one-time migration of a verified legacy result; never a rebuild fallback.

    The caller must verify that the legacy data came from these exact GTFS
    sources. Original Synpp artifacts stay read-only; only copies are normalized.
    """
    def copy_bundle(destination):
        source = Path(prepared['timetable'])
        for suffix in TIMETABLE_SUFFIXES:
            origin = Path(prepared['stops']) if suffix == '.stops.tsv' else Path(str(source) + suffix)
            target = destination / ('timetable.binary' + suffix)
            if suffix in ('.stops.tsv', '.transfers.tsv'):
                columns = 1 if suffix == '.stops.tsv' else 2
                with origin.open() as incoming, target.open('w') as outgoing:
                    for line in incoming:
                        fields = line.rstrip('\n').split('\t')
                        fields[:columns] = [value.replace(',', '') for value in fields[:columns]]
                        outgoing.write('\t'.join(fields) + '\n')
            else:
                shutil.copy2(origin, target)
        log = source.parent / 'preprocessing.log'
        if log.exists():
            shutil.copy2(log, destination / log.name)
        return prepared['preprocessing']['preprocessing_seconds']
    return ensure(binary, '.', directory, prepared['date'], 1, cells, sources, copy_bundle)
