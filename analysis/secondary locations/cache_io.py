"""Read completed Synpp artifacts without scheduling or modifying the pipeline."""
from pathlib import Path
from collections import deque
import hashlib
import json
import pickle
import logging
import yaml

LOG = logging.getLogger(__name__)
ROOT_STAGES = ('synthesis.population.spatial.locations',
               'synthesis.population.spatial.secondary.locations')


class RunCache:
    def __init__(self, config_path):
        self.path = Path(config_path).resolve()
        self.yaml = yaml.safe_load(self.path.read_text())
        self.directory = Path(self.yaml['working_directory'])
        self.meta = json.loads((self.directory / 'pipeline.json').read_text())
        roots = [key for stage in ROOT_STAGES for key in self.candidates(stage)]
        if not roots:
            raise FileNotFoundError(f'No completed spatial assignment in {self.directory}')
        self.root = max(roots, key=lambda k: self.meta[k]['updated'])
        self.closure, self.depth = set(), {}
        pending = deque([(self.root, 0)])
        while pending:
            key, depth = pending.popleft()
            if key in self.closure:
                continue
            self.closure.add(key)
            self.depth[key] = depth
            pending.extend((dependency, depth + 1) for dependency in self.meta.get(key, {}).get('dependencies', {}))
        self.used = {}
        self.keep = {}
        self.settings = self.meta[self.root]['config']

    def candidates(self, stage):
        return [key for key in self.meta if key.startswith(stage + '__')
                and (self.directory / (key + '.p')).is_file()]

    def resolve(self, stage):
        choices = self.candidates(stage)
        linked = [key for key in choices if key in self.closure]
        if linked:
            # Production experts are direct assignment dependencies; fold models are deeper.
            nearest = min(self.depth[key] for key in linked)
            choices = [key for key in linked if self.depth[key] == nearest]
        if len(choices) != 1:
            raise ValueError(f'Expected one completed artifact for {stage}: {choices}')
        return choices[0]

    def stage(self, stage, retain=False):
        key = self.resolve(stage)
        if key in self.keep:
            return self.keep[key]
        path = self.directory / (key + '.p')
        before = path.stat()
        LOG.info('Reading %s (%0.1f MB)', key, before.st_size / 1e6)
        with path.open('rb') as stream:
            obj = pickle.load(stream)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'Artifact changed while being read: {path}; retry after stage completes')
        self.used[stage] = dict(path=str(path), size_bytes=after.st_size,
                                mtime_ns=after.st_mtime_ns, completed=self.meta[key]['updated'])
        self.used[stage]['module_hash'] = self.meta[key].get('module_hash')
        if isinstance(obj, tuple) and all(isinstance(v, str) for v in obj):
            self.used[stage]['referenced_files'] = [dict(path=str(v), size_bytes=Path(v).stat().st_size, mtime_ns=Path(v).stat().st_mtime_ns) for v in obj if Path(v).is_file()]
        if retain:
            self.keep[key] = obj
        return obj

    def config(self, key, default=None):
        # Completed-stage metadata wins over subsequently edited YAML settings.
        if key in self.settings:
            return self.settings[key]
        candidates = self.candidates('synthesis.population.spatial.secondary_nn.locations')
        if len(candidates) == 1 and key in self.meta[candidates[0]]['config']:
            return self.meta[candidates[0]]['config'][key]
        return self.yaml['config'].get(key, default)

    @property
    def kind(self):
        if str(self.config('locations_algorithm', 'rda')).lower() not in ['nn', 'neural_network', 'v2', 'v.2', 'locations_v2']:
            return 'rda'
        return 'nn_distance' if self.config('secondary_nn_use_target_distance', True) else 'nn_no_distance'

    def provenance(self):
        keys = ['locations_algorithm', 'secondary_nn_use_target_distance', 'random_seed',
                'input_downsampling', 'scaling_year', 'census', 'specific_day_scenario',
                'secondary_nn_short_trip_threshold_m']
        return dict(config=str(self.path), config_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    cache=str(self.directory), root=self.root, method=self.kind,
                    recorded_settings={k: self.config(k) for k in keys}, artifacts=dict(self.used))
