"""Frozen input caches and resumable execution of the existing secondary stages."""
from collections import deque
from copy import deepcopy
import gc
import hashlib
import importlib
import json
import logging
from pathlib import Path
import pickle
import time

from synpp.pipeline import ConfigurationContext, get_stage_hash
from synpp.progress import ProgressContext

from cache_io import RunCache

LOG = logging.getLogger(__name__)
REPOSITORY = Path(__file__).resolve().parents[2]
PREFIX = 'synthesis.population.spatial.secondary_nn.'
MODELS = tuple(PREFIX + name for name in (
    'regional_model', 'subregional_model', 'local_model', 'short_range_model'))
ASSIGNMENT = PREFIX + 'locations'
SECONDARY = 'synthesis.population.spatial.secondary.locations'
STAGES = (*MODELS, ASSIGNMENT)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.chmod(0o600)
    temporary.replace(path)


def stamp(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def verify_stamp(record):
    if stamp(record['path']) != record:
        raise RuntimeError('Frozen input or completed output changed: ' + record['path'])


def source_fingerprint():
    # Include helpers because Synpp's per-module hash does not cover imported code.

    digest = hashlib.sha256()
    for package in ['secondary_nn', 'secondary']:
        directory = REPOSITORY / 'synthesis/population/spatial' / package
        for path in sorted(directory.rglob('*.py')):
            digest.update(str(path.relative_to(REPOSITORY)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def create_plan(config_path, output, weights, threads, people, draws):
    source = RunCache(config_path)
    if source.kind != 'nn_distance':
        raise ValueError('The sweep requires a completed distance-enabled NN run')
    if not weights or len(set(weights)) != len(weights) or any(w <= 0 or w == .1 for w in weights):
        raise ValueError('Supply distinct positive trial weights; 0.1 is the cached baseline')
    if min(threads, people, draws) < 1:
        raise ValueError('Threads, replay people and draws must be positive')

    # Recover the settings actually used, including defaults omitted from private YAML.

    settings = deepcopy(source.yaml['config'])
    for name in (*MODELS, ASSIGNMENT):
        key = source.resolve(name)
        record = source.meta[key]
        settings.update(record['config'])
        module = importlib.import_module(name)
        if record.get('module_hash') != get_stage_hash(module):
            raise RuntimeError('Baseline stage code differs; cannot isolate weight changes: ' + name)
        if name in MODELS:
            if record['config'].get('secondary_nn_distance_loss_kind') != 'crps_energy':
                raise ValueError('Baseline experts must use the distribution loss')
            if record['config'].get('secondary_nn_distribution_loss_weight') != .1:
                raise ValueError('All baseline experts must use weight 0.1')
            if name == MODELS[-1]:
                override = record['config'].get('secondary_nn_short_distribution_loss_weight')
                relative = record['config'].get('secondary_nn_short_relative_distance_loss_weight', 0.)
                if override not in (None, .1) or relative != 0:
                    raise ValueError('A global sweep requires a pure 0.1 short baseline: set the short '
                                     'distribution override to null and auxiliary relative weight to zero')
    settings.update(threads=threads, secondary_nn_use_target_distance=True,
                    secondary_nn_distance_loss_kind='crps_energy')

    # Pin the completed dependency graph and file identities without copying source data.

    metadata = {key: source.meta[key] for key in source.closure}
    files = [stamp(source.directory / (key + '.p')) for key in sorted(source.closure)]
    for name in MODELS:
        for path in source.stage(name):
            files.append(stamp(path))
    snapshot = deepcopy(source.yaml)
    snapshot['config'] = settings
    snapshot['run'] = [SECONDARY]
    plan = dict(version=1, source_config=str(source.path), baseline_root=source.root,
                configuration=snapshot, metadata=metadata, frozen_files=files,
                source_fingerprint=source_fingerprint(), weights=weights,
                threads=threads, replay_people=people, draws=draws,
                created_utc=time.time(), output=str(Path(output).resolve()))
    preflight(plan)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'plan.json', plan)
    return plan


def preflight(plan):
    cache = FrozenCache(plan)
    for name in STAGES:
        configured = ConfigurationContext(cache.options)
        importlib.import_module(name).configure(configured)
        for dependency in configured.required_stages:
            stage = dependency['descriptor']
            if stage not in STAGES:
                cache.resolve(stage)
    cache.verify()


class FrozenCache(RunCache):
    """Reuse a completed dependency graph; only explicitly listed stages can execute."""

    def __init__(self, plan, weight=.1):
        self.plan = plan
        self.weight = weight
        self.path = Path(plan['source_config'])
        self.yaml = deepcopy(plan['configuration'])
        self.directory = Path(self.yaml['working_directory'])
        self.meta = plan['metadata']
        self.root = plan['baseline_root']
        self.closure = set(self.meta)
        self.depth = {}
        pending = deque([(self.root, 0)])
        while pending:
            key, depth = pending.popleft()
            if key in self.depth:
                continue
            self.depth[key] = depth
            pending.extend((k, depth + 1) for k in self.meta[key]['dependencies'])
        self.options = self.yaml['config']
        self.options['secondary_nn_distribution_loss_weight'] = weight
        self.settings = self.options
        self.keep, self.used = {}, {}
        self.variant = Path(plan['output']) / f'weight_{weight:g}'

    def config(self, key, default=None):
        return self.options.get(key, default)

    def stage_directory(self, name):
        return self.variant / 'stages' / name.rsplit('.', 1)[-1]

    def stage(self, name, retain=False):
        actual = ASSIGNMENT if name == SECONDARY else name
        if self.weight != .1 and actual in STAGES:
            verify_completed(self.stage_directory(actual), self.plan, self.weight)
            with (self.stage_directory(actual) / 'result.p').open('rb') as stream:
                return pickle.load(stream)
        return super().stage(name, retain=retain)

    def provenance(self):
        return dict(weight=self.weight, baseline_root=self.root, source_config=str(self.path),
                    source_fingerprint=self.plan['source_fingerprint'], artifacts=self.used,
                    frozen_inputs=self.plan['frozen_files'])

    def verify(self):
        if source_fingerprint() != self.plan['source_fingerprint']:
            raise RuntimeError('Secondary-model code changed after the experiment was prepared')
        for record in self.plan['frozen_files']:
            verify_stamp(record)


class StageContext:
    def __init__(self, cache, name):
        self.cache = cache
        self.name = name
        self.configured = ConfigurationContext(cache.options)
        importlib.import_module(name).configure(self.configured)
        self.loaded = {}
        self.working_directory = str(cache.stage_directory(name))

    def config(self, key):
        return self.configured.config(key)

    def stage(self, name):
        declared = {entry['descriptor'] for entry in self.configured.required_stages}
        if name not in declared:
            raise ValueError('Undeclared stage dependency: ' + name)
        if name not in self.loaded:
            self.loaded[name] = self.cache.stage(name)
        return self.loaded[name]

    def path(self, name=None):
        if name is None:
            return self.working_directory
        if name in getattr(self.cache, 'execution_stages', STAGES):
            return str(self.cache.stage_directory(name))
        return str(self.cache.directory / (self.cache.resolve(name) + '.cache'))

    def progress(self, *args, **kwargs):
        return ProgressContext(*args, **kwargs, minimum_interval=30)


def completion_identity(plan, weight):
    return dict(weight=weight, baseline_root=plan['baseline_root'],
                source_fingerprint=plan['source_fingerprint'],
                configuration_sha256=hashlib.sha256(json.dumps(
                    plan['configuration'], sort_keys=True).encode()).hexdigest())


def verify_completed(directory, plan, weight):
    record = json.loads((directory / 'completed.json').read_text())
    if record['identity'] != completion_identity(plan, weight):
        raise RuntimeError('Completed stage belongs to a different experiment')
    for output in record['outputs']:
        verify_stamp(output)


def execute_variant(plan, weight):
    if weight not in plan['weights']:
        raise ValueError('Weight was not included in the prepared experiment')
    cache = FrozenCache(plan, weight)
    execute_stages(cache, STAGES)


def execute_stages(cache, stages):
    """Execute an explicit stage subset, leaving all other dependencies frozen."""
    plan, weight = cache.plan, cache.weight
    cache.verify()

    # Execute existing stage functions in their dependency order on the frozen inputs.

    for name in stages:
        directory = cache.stage_directory(name)
        if (directory / 'completed.json').exists():
            verify_completed(directory, plan, weight)
            LOG.info('Reusing completed weight=%s stage=%s', weight, name)
            continue
        LOG.info('Starting weight=%s stage=%s', weight, name)
        directory.mkdir(parents=True, exist_ok=True)
        context = StageContext(cache, name)
        module = importlib.import_module(name)
        result = module.execute(context)
        cache.verify()
        result_path = directory / 'result.p'
        temporary = directory / 'result.p.tmp'
        with temporary.open('wb') as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(result_path)
        outputs = [stamp(result_path)]
        if name in MODELS:
            outputs.extend(stamp(path) for path in result)
        write_json(directory / 'completed.json', dict(identity=completion_identity(plan, weight),
                   outputs=outputs, config=context.configured.required_config, stage=name))
        del result, context
        gc.collect()
    LOG.info('Completed weight=%s', weight)
