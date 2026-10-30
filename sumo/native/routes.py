"""Compile independent route arithmetic kernels; no SUMO or Arrow build required."""

import hashlib
from pathlib import Path
import platform
import shutil
import subprocess


def configure(context):
    pass


def execute(context):

    # Compile the small C ABI library once, in its own cache stage.

    return build(Path(context.path()) / 'route_kernel.so')


def build(output):
    compiler = shutil.which('c++') or shutil.which('g++')
    if compiler is None:
        raise RuntimeError('Native route kernels require a C++ compiler; set sumo_native_routes: false to use Python')
    subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                    str(Path(__file__).with_name('route_kernel.cpp')), '-o', str(output)], check=True)
    return str(Path(output).resolve())


def validate(context):
    return [hashlib.sha256(Path(__file__).with_name('route_kernel.cpp').read_bytes()).hexdigest(), platform.machine()]
