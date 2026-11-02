"""Build the Arrow-to-SUMO XML writer once in a Synpp cache."""

import hashlib
from pathlib import Path
import platform
import shutil
import subprocess

import pyarrow as pa


def configure(context):
    pass


def execute(context):

    # Compile against the Arrow headers and library already installed with PyArrow.

    return {"writer": build_writer(Path(context.path()) / "write_trips")}


def validate(context):
    return [hashlib.sha256(Path(__file__).with_name("write_trips.cpp").read_bytes()).hexdigest(),
            pa.__version__, platform.machine()]


def build_writer(output):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("SUMO native XML needs a C++ compiler; set sumo_native_xml: false for the lxml fallback")
    library_dir = Path(pa.get_library_dirs()[0])
    library = next(library_dir.glob("libarrow.so.*"))
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, "-O3", "-std=c++17", str(Path(__file__).with_name("write_trips.cpp")),
                    "-I", pa.get_include(), str(library), "-Wl,-rpath," + str(library_dir),
                    "-lz", "-pthread", "-o", str(output)], check=True)
    return str(output)
