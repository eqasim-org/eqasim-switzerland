"""Standalone inspection of exported simulation files; no population stages run."""

import argparse
import hashlib
import json
import logging
import os
import pickle
import shlex
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen

from .build import build_network, build_people, index_person_features, index_user_groups, source_identity
from .diagnose import diagnose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    build = actions.add_parser('build', help='Stream exports and build compact reusable indexes')
    for field in ['events', 'plans', 'network', 'geometry', 'counts', 'swiss-border-cache', 'output']:
        build.add_argument('--' + field, required=True, type=Path)
    build.add_argument('--sample', required=True, type=float)
    build.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    build.add_argument('--sdk', help='Optional macOS SDK path; no system toolchain settings are modified')
    purposes = actions.add_parser('purposes', help='Add all-link activity-purpose shares to an existing map')
    purposes.add_argument('--output', required=True, type=Path)
    purposes.add_argument('--destination', type=Path, help='Optional separate directory for the new index')
    purposes.add_argument('--compiler', default=os.environ.get('CXX', 'c++'))
    purposes.add_argument('--sdk')
    for command, description in [('serve', 'Serve the map on localhost from existing indexes'),
                                 ('open', 'Start the local viewer if necessary and open its map')]:
        server = actions.add_parser(command, help=description)
        server.add_argument('--output', required=True, type=Path)
        server.add_argument('--sample', required=True, type=float)
        server.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    if args.action == 'purposes':
        from .purposes import build_purposes
        logging.basicConfig(level=logging.INFO)
        build_purposes(args.output, args.destination, args.compiler, args.sdk)
        return
    if not 0 < args.sample <= 1:
        parser.error('--sample must be in (0,1]')
    logging.basicConfig(level=logging.INFO)
    if args.action == 'open':
        open_viewer(args.output, args.sample, args.port)
        return
    if args.action == 'serve':
        from .serve import create_app
        create_app(args.output, args.sample).run(host='127.0.0.1', port=args.port, threaded=True, debug=False)
        return

    # Each build uses a new directory: existing indexes are never assumed to
    # describe a different export or overwritten after partial extraction.
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import pandas as pd
    counts = pd.read_csv(args.counts, sep=';', dtype={'linkId': str})
    if not counts.linkId.is_unique:
        raise ValueError('Duplicate daily-count link IDs')
    (args.output / 'link_ids.txt').write_text('\n'.join(counts.linkId) + '\n')
    np.save(args.output / 'reference_daily_counts.npy', counts.dailyCount.to_numpy())
    source = Path(__file__).parent / 'native/extract.cpp'
    binary = args.output / 'extract'
    command = shlex.split(args.compiler) + ['-O3', '-std=c++17']
    if args.sdk:
        command += ['-isysroot', args.sdk]
    subprocess.run(command + [str(source), '-lexpat', '-lz', '-o', str(binary)], check=True)
    subprocess.run([str(binary), 'plans', str(args.plans), str(args.output)], check=True)
    build_network(args.output, args.network, args.geometry)
    build_people(args.output)
    with args.swiss_border_cache.open('rb') as stream:
        border = pickle.load(stream).union_all()
    index_person_features(args.output, border)
    subprocess.run([str(binary), 'events', str(args.events), str(args.output), '86400'], check=True)
    index_user_groups(args.output)
    diagnose(args.output, args.sample)
    prepare_assets(args.output)
    manifest = {'sample': args.sample, 'window_seconds': [0, 86400],
                'sources': {name: source_identity(getattr(args, name)) for name in ['events', 'plans', 'network', 'geometry', 'counts']},
                'extractor_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def prepare_assets(output):
    """Pin small map dependencies locally; raw simulation data never leaves disk."""
    assets = Path(output) / 'assets'
    assets.mkdir(exist_ok=True)
    for name in ['leaflet.js', 'leaflet.css', 'images/layers.png', 'images/layers-2x.png']:
        path = assets / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen('https://unpkg.com/leaflet@1.9.4/dist/' + name, timeout=45) as response:
                path.write_bytes(response.read())


def open_viewer(output, sample, port):
    """A restart reuses disk indexes and does not reread the simulation XML."""
    output = Path(output).resolve()
    url = f'http://127.0.0.1:{port}/'
    try:
        with urlopen(url + 'api/meta', timeout=1):
            pass
    except OSError:
        repository = Path(__file__).resolve().parents[3]
        with (output / 'server.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-u', '-m', 'analysis.counts.road_users',
                'serve', '--output', str(output), '--sample', str(sample), '--port', str(port)],
                cwd=repository, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        (output / 'server.pid').write_text(str(process.pid))
        for _ in range(50):
            try:
                with urlopen(url + 'api/meta', timeout=1):
                    break
            except OSError:
                time.sleep(.2)
        else:
            raise RuntimeError('The map server did not start; inspect server.log')
    webbrowser.open(url)


if __name__ == '__main__':
    main()
