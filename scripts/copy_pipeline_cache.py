"""Copy reusable Synpp caches while excluding live simulation/export stages."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import yaml

EXCLUDED_STAGES = ('matsim.simulation.run', 'matsim.output', 'matsim.cutter.')


def copy_cache(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Source and target caches must be separate, non-nested directories')
    metadata_path = source / 'pipeline.json'
    metadata = json.loads(metadata_path.read_text())
    marker = target / 'baseline_snapshot.json'
    pending = target / '.baseline_copy_in_progress'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous.get('source') != str(source):
            raise ValueError('Target already contains a snapshot of another cache')
        print('Reusing completed baseline snapshot:', target, flush=True)
        return
    if target.exists() and any(target.iterdir()) and not pending.exists():
        raise FileExistsError('Refusing to overwrite an existing non-snapshot cache: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    pending.touch()
    command = ['rsync', '-a', '--stats', '--exclude=/pipeline.json*', '--exclude=/flowchart.json',
               '--exclude=*.lock', '--exclude=/baseline_snapshot.json', '--exclude=/.baseline_copy_in_progress']
    command += ['--exclude=/' + name + '*' for name in EXCLUDED_STAGES]
    print('Copying reusable cache:', source, '->', target, flush=True)
    subprocess.run(command + [str(source) + '/', str(target) + '/'], check=True)
    # Use the metadata read before copying, never a partially updated live file.
    reusable = {key: value for key, value in metadata.items() if not key.startswith(EXCLUDED_STAGES)}
    (target / 'pipeline.json').write_text(json.dumps(reusable, indent=2))
    marker.write_text(json.dumps(dict(source=str(source), copied_at_utc=datetime.now(timezone.utc).isoformat(),
                                     completed_stages=len(reusable), excluded_stage_prefixes=EXCLUDED_STAGES), indent=2))
    pending.unlink()
    print('Baseline snapshot complete:', target, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_config', type=Path)
    parser.add_argument('target_config', type=Path)
    args = parser.parse_args()
    source = yaml.safe_load(args.source_config.read_text())['working_directory']
    target = yaml.safe_load(args.target_config.read_text())['working_directory']
    copy_cache(source, target)
