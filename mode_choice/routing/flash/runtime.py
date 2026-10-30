"""Build the pinned upstream FLASH-TB headers with our native CSV adapter."""
from pathlib import Path
import shutil
import subprocess

COMMIT = '23963f5ff14e5f761c16a25ab4694da7900e64ba'
REPOSITORY = 'https://github.com/TransitRouting/FLASH-TB.git'


def configure(context):
    context.config('flash_tb_source_path', '')
    context.config('flash_tb_cxx', 'g++')
    context.config('flash_tb_numa_include', '')


def execute(context):

    # Build in the stage cache and leave any user-supplied checkout untouched.

    directory = Path(context.path()).resolve()
    source = context.config('flash_tb_source_path')
    with (directory / 'build.log').open('w') as log:
        if not source:
            source = directory / 'source'
            if not (source / '.git').is_dir():
                subprocess.run(['git', 'clone', REPOSITORY, str(source)], check=True, stdout=log, stderr=log)
            subprocess.run(['git', '-C', str(source), 'checkout', '--detach', COMMIT], check=True, stdout=log, stderr=log)
        return str(build(source, directory / 'flash_tb_router', context.config('flash_tb_cxx'),
                         context.config('flash_tb_numa_include'), log))


def build(source, target, compiler='g++', numa_include='', log=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != COMMIT:
        raise ValueError(f'FLASH-TB checkout must be pinned to {COMMIT}; found {revision}')
    executable = shutil.which(compiler)
    if executable is None:
        raise FileNotFoundError(f'C++ compiler not found: {compiler}')
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, '-std=c++17', '-O3', '-DNDEBUG', '-fopenmp', '-I' + str(source)]
    if numa_include:
        command.append('-I' + str(Path(numa_include).resolve()))
    command.extend([str(Path(__file__).parent / 'native/router.cpp'), '-Wl,-l:libnuma.so.1', '-o', str(target)])
    subprocess.run(command, check=True, stdout=log, stderr=log)
    return target


def validate(context):
    import hashlib
    digest = hashlib.sha256((Path(__file__).parent / 'native/router.cpp').read_bytes()).hexdigest()
    source = context.config('flash_tb_source_path')
    revision = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() if source else COMMIT
    return [digest, revision]
