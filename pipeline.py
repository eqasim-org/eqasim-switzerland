import importlib
import os.path, shutil
import pickle
import itertools
import time

class safe_tqdm:
    def __init__(self, iterator = None, total = None, position = None, desc = "  Running something without description ...", *a, **k):
        self.desc = desc
        self.iterator = iterator
        self.total = total
        self.current = 0
        self.last_time = 0.0

        if position is not None:
            self.desc = self.desc + "(%d)" % (position + 1)

        self._print()

    def _print(self):
        current_time = time.time()

        if current_time - self.last_time > 10.0:
            self.last_time = current_time

            if self.total is None:
                if self.current == 0:
                    print("%s: running" % self.desc)
                else:
                    print("%s: %d" % (self.desc, self.current))
            else:
                print("%s: %d/%d (%.2f%%)" % (self.desc, self.current, self.total, 100.0 * self.current / self.total))

    def _print_done(self):
        print("%s: done" % self.desc)

    def __enter__(self):
        return self

    def __exit__(self ,type, value, traceback):
        self._print_done()

    def __iter__(self):
        def loop():
            for element in self.iterator:
                yield element
                self.update()
            self._print_done()

        return loop()
        #return iter(self.iterator)

    def update(self, count = 1, *a, **k):
        self.current += count
        self._print()

class Require:
    def __init__(self):
        self.config_defaults = {}
        self.config_names = list()
        self.stage_names = list()
        self.cache = True

    def config(self, name, default = None):
        self.config_names.append(name)

        if default is not None:
            self.config_defaults[name] = default

    def stage(self, name):
        self.stage_names.append(name)

class Context:
    def __init__(self, target_path, config):
        self.stages = {}
        self.target_path = target_path
        self.config = config
        self.cache_path = None

    def cache_path(self, name):
        return "%s/%s_cache" % (self.target_path, name)

    def stage_path(self, name):
        return "%s/%s.p" % (self.target_path, name)

    def save(self, name, data, cache = True):
        self.stages[name] = data

        if cache:
            with open(self.stage_path(name), "wb+") as f:
                pickle.dump(data, f)

    def stage(self, name):
        if not name in self.stages:
            with open(self.stage_path(name), "rb") as f:
                self.stages[name] = pickle.load(f)

        return self.stages[name]

def compute_dag(dependencies):
    linear = []
    remaining = list(dependencies.keys())

    while len(remaining) > 0:
        prior_count = len(remaining)

        for item in remaining[:]:
            insertable = True

            for dependency in dependencies[item]:
                insertable = insertable & (dependency in linear)

            if insertable:
                linear.append(item)
                remaining.remove(item)

        posterior_count = len(remaining)

        if prior_count == posterior_count:
            raise RuntimeError()

    parents = {}

    for seed in linear:
        parents[seed] = set(dependencies[seed])

        for parent in parents[seed]:
            parents[seed] = parents[seed] | parents[parent]

    direct_children = {}

    for seed in linear:
        direct_children[seed] = set()

        for candidate in linear:
            if seed in dependencies[candidate]:
                direct_children[seed].add(candidate)

    children = {}

    for seed in linear[::-1]:
        children[seed] = set(direct_children[seed])

        for child in children[seed]:
            children[seed] = children[seed] | children[child]

    return { "sequence" : linear, "parents" : parents, "children" : children }

def run(requested_stages, target_path = "target", config = {}):
    stage_names = requested_stages[:]

    stages = {}
    requirements = {}
    dependencies = {}

    config_defaults = {}

    while len(stage_names) > 0:
        stage_name = stage_names[0]
        del stage_names[0]

        if not stage_name in stages:
            require = Require()

            stage = importlib.import_module(stage_name)
            stage.configure(None, require)

            stages[stage_name] = stage
            requirements[stage_name] = require
            dependencies[stage_name] = list(set(require.stage_names))

            stage_names += require.stage_names

    config_defaults = {}
    multiple_defaults = []

    for stage_name, require in requirements.items():
        for config_name in require.config_defaults:
            if config_name in config_defaults:
                raise RuntimError("Multiple defaults")
            else:
                config_defaults[config_name] = require.config_defaults[config_name]

    for config_name, config_value in config_defaults.items():
        if not config_name in config:
            config[config_name] = config_value

    missing_config_values = []

    for stage_name, require in requirements.items():
        for config_name in require.config_names:
            if not config_name in config:
                missing_config_values.append((stage_name, config_name))

    if len(missing_config_values) > 0:
        print("Missing config values: ")

        for stage_name, config_name in missing_config_values:
            print("Stage", stage_name, "->", config_name)

        raise RuntimeError("Missing config values")

    dag = compute_dag(dependencies)

    uncached_stages = set()

    for stage_name in stages.keys():
        if not requirements[stage_name].cache:
            uncached_stages.add(stage_name)
            path = "%s/%s.p" % (target_path, stage_name)

            if os.path.exists(path):
                os.remove(path)

        elif not os.path.exists("%s/%s.p" % (target_path, stage_name)):
            uncached_stages.add(stage_name)

    active = set(requested_stages)

    for request_name in requested_stages:
        for stage_name in list(uncached_stages) + list(requested_stages):
            if request_name in dag["children"][stage_name]:
                active = active | ( dag["parents"][request_name] & dag["children"][stage_name] )
                active.add(stage_name)

    active_sequence = [a for a in dag["sequence"] if a in active]
    context = Context(target_path, config)

    print(active_sequence)

    for stage_name in active_sequence:
        print("Executing stage %s ..." % stage_name)

        cache_path = "%s/%s_cache" % (target_path, stage_name)
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        os.makedirs(cache_path)
        context.cache_path = cache_path

        data = stages[stage_name].execute(context)
        context.save(stage_name, data, requirements[stage_name].cache)
























































#
