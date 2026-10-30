"""Add passage-weighted activity purposes to an existing road-user map."""

import hashlib
import json
import logging
import os
import shlex
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .build import GROUPS, link_statistics, source_identity


logger = logging.getLogger(__name__)


def build_purposes(output, destination=None, compiler=None, sdk=None):
    """Reuse plan indexes and stream events once; publish only validated results."""
    output = Path(output).resolve()
    destination = Path(destination or output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / 'manifest.json').read_text())
    events = Path(manifest['sources']['events']['path'])
    # Refuse to combine a replaced simulation export with the old map indexes.
    for name in ('events', 'plans'):
        expected = manifest['sources'][name]
        if source_identity(expected['path']) != expected:
            raise ValueError(f'{name} export has changed; rebuild the map before adding purposes')
    source = Path(__file__).parent / 'native/extract.cpp'
    with tempfile.TemporaryDirectory(prefix='purpose-build-', dir=destination) as directory:
        work = Path(directory)
        for name in ('persons.tsv', 'trips.tsv', 'link_ids.txt'):
            (work / name).symlink_to(output / name)
        with np.load(output / 'person_features.npz') as features:
            features['group'].astype('uint8').tofile(work / 'person_groups.bin')
        command = shlex.split(compiler or os.environ.get('CXX', 'c++')) + ['-O3', '-std=c++17']
        if sdk:
            command += ['-isysroot', sdk]
        subprocess.run(command + [str(source), '-lexpat', '-lz', '-o', str(work / 'extract')], check=True)
        subprocess.run([str(work / 'extract'), 'purposes', str(events), str(work), '86400'], check=True)

        # Store only observed link/group/purpose combinations, indexed by link.
        database = work / 'purposes.sqlite'
        stats = link_statistics(output)
        observed = np.zeros(len(stats), dtype=np.uint64)
        with sqlite3.connect(database) as connection:
            connection.execute('CREATE TABLE counts (link_idx INTEGER, group_id INTEGER, origin INTEGER, destination INTEGER, passages INTEGER)')
            for frame in pd.read_csv(work / 'purpose_counts.tsv', sep='\t', chunksize=100000):
                if not frame.group_id.between(0, len(GROUPS) - 1).all():
                    raise ValueError('Invalid traveller group')
                np.add.at(observed, frame.link_idx.to_numpy(), frame.passages.to_numpy(dtype=np.uint64))
                connection.executemany('INSERT INTO counts VALUES (?,?,?,?,?)', frame.itertuples(index=False, name=None))
            pd.read_csv(work / 'purpose_labels.tsv', sep='\t', keep_default_na=False).to_sql('purposes', connection, index=False)
            connection.execute('CREATE INDEX by_link ON counts(link_idx)')
            # Compare every link, including links with zero eligible passages.
            expected = stats['entries'].astype(np.uint64) - stats['transit'] - stats['bike']
            if not np.array_equal(observed, expected):
                raise ValueError(f"Purpose passage totals disagree on {np.count_nonzero(observed != expected)} links")
            unknown = connection.execute('SELECT coalesce(sum(passages),0) FROM counts WHERE origin=0 OR destination=0').fetchone()[0]
            report = dict(passages=int(observed.sum()), links_with_passages=int((observed > 0).sum()),
                          unknown_either_purpose=int(unknown), all_link_totals_match=True,
                          window_seconds=[0, 86400], sources=manifest['sources'],
                          extractor_sha256=hashlib.sha256(source.read_bytes() + source.with_name('purposes.h').read_bytes()).hexdigest())
        database.replace(destination / 'purposes.sqlite')
        (destination / 'purpose_validation.json').write_text(json.dumps(report, indent=2))
        logger.info('Purpose validation: %s passages on %s links; all link totals match', report['passages'], report['links_with_passages'])
    return report


def purpose_summary(connection, link_index):
    """Return within-group shares with Unknown retained in every denominator."""
    rows = connection.execute('''SELECT c.group_id,o.label,d.label,sum(c.passages)
        FROM counts c JOIN purposes o ON o.idx=c.origin JOIN purposes d ON d.idx=c.destination
        WHERE c.link_idx=? GROUP BY c.group_id,c.origin,c.destination''', (link_index,)).fetchall()
    summaries = []
    for group_id, label in [(-1, 'All road-user groups'), *enumerate(GROUPS)]:
        selected = [r for r in rows if group_id == -1 or r[0] == group_id]
        total = sum(r[3] for r in selected)
        item = dict(group_id=group_id, label=label, passages=total)
        for position, field in ((1, 'origin'), (2, 'destination')):
            counts = {}
            for row in selected:
                counts[row[position]] = counts.get(row[position], 0) + row[3]
            item[field] = [dict(label=name, count=count, share=count / total)
                           for name, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
        summaries.append(item)
    return dict(available=True, groups=summaries, unit='sample vehicle passages', window_seconds=[0, 86400])
