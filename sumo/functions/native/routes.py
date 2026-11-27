"""Compile independent route arithmetic kernels; no SUMO or Arrow build required."""

import shutil
import subprocess
from pathlib import Path


def build(output):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError(
            "Native route kernels require a C++ compiler; set sumo_native_routes: false to use Python"
        )
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(Path(__file__).with_name("route_kernel.cpp")),
            "-o",
            str(output),
        ],
        check=True,
    )
    return str(Path(output).resolve())
