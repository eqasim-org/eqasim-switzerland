"""Bind production routing functions to sampled inputs and private stage files."""
from pathlib import Path
from copy import deepcopy


class ConfigureView:
    def __init__(self, context, mapping=None, ignored=()):
        self.context, self.mapping, self.ignored = context, mapping or {}, set(ignored)

    def config(self, *args, **kwargs):
        return self.context.config(*args, **kwargs)

    def stage(self, name, *args, **kwargs):
        if name not in self.ignored:
            return self.context.stage(self.mapping.get(name, name), *args, **kwargs)


class ExecuteView:
    def __init__(self, context, stages, config=None, directory=None):
        self.context, self.stages, self.config_values = context, stages, config or {}
        self.directory = Path(directory or context.path()).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def config(self, name, *args, **kwargs):
        return self.config_values[name] if name in self.config_values else self.context.config(name, *args, **kwargs)

    def stage(self, name, *args, **kwargs):
        return self.stages[name] if name in self.stages else self.context.stage(name, *args, **kwargs)

    def path(self, name=None):
        return str(self.directory) if name is None else self.context.path(name)


def private_server(settings, directory):
    """Copy launch settings; source stage caches and input files stay untouched."""
    result = deepcopy(settings)
    directory = Path(directory).resolve()
    temporary = directory / 'java_tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    result['cwd'] = str(directory)
    command = result['command']
    # The comparison only requests PT; no road assignment is required.
    if '--marouter' in command:
        command[command.index('--marouter') + 1] = 'false'
    for index, argument in enumerate(command):
        if argument.startswith('-Djava.io.tmpdir='):
            command[index] = '-Djava.io.tmpdir=' + str(temporary)
    if '--export-routes' in command:
        command[command.index('--export-routes') + 1] = 'true'
    else:
        command.extend(['--export-routes', 'true'])
    return result
