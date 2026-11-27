"""Build and run bounded Arrow/XML processing outside the Python interpreter."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import pyarrow as pa

from sumo.functions.config.tools import positive


def source_token(command="all"):
    root = Path(__file__).with_suffix("")
    digest = hashlib.sha256()
    components = {
        "index": {"index.hpp"},
        "realize": {"routes.hpp", "network.hpp"},
        "cut": {"cut.hpp", "routes.hpp", "network.hpp"},
        "export": {"export.hpp", "routes.hpp", "network.hpp"},
    }
    selected = (
        None if command == "all" else components[command] | {"main.cpp", "io.hpp"}
    )
    for path in sorted(root.glob("*")):
        if selected is not None and path.name not in selected:
            continue
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    if command in {"all", "cut"}:
        digest.update(Path(__file__).with_name("route_kernel.cpp").read_bytes())
    return digest.hexdigest()


def build(output, command="all"):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError(
            "Native SUMO processing needs a C++ compiler; set sumo_native_routes: false for Python"
        )
    library_dir = Path(pa.get_library_dirs()[0])
    library = next(library_dir.glob("libarrow.so.*"))
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    defines = (
        []
        if command == "all"
        else ["-DSUMO_BUILD_SINGLE", "-DSUMO_BUILD_" + command.upper()]
    )
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            *defines,
            str(Path(__file__).with_suffix("") / "main.cpp"),
            "-I",
            pa.get_include(),
            str(library),
            "-Wl,-rpath," + str(library_dir),
            "-lexpat",
            "-lz",
            "-pthread",
            "-o",
            str(output),
        ],
        check=True,
    )
    return str(output)


def configure_options(context):
    context.config("threads", volatile=True)
    context.config("sumo_route_batch_size", 10000)
    context.config("sumo_route_batch_bytes", 16 * 1024 * 1024)
    context.config("sumo_route_processing_threads", None)
    context.config("sumo_xml_threads", 4)


def configure_stage(context, command):
    context.stage(
        "sumo.native.processing",
        {"sumo_native_processing_command": command},
        alias="sumo.native.processing",
    )
    configure_options(context)


def run(context, command, **paths):
    allocated = positive(context.config("threads"), "threads", integer=True)
    workers = context.config("sumo_route_processing_threads")
    workers = (
        min(allocated, 8)
        if workers is None
        else min(
            allocated, positive(workers, "sumo_route_processing_threads", integer=True)
        )
    )
    args = [
        context.stage("sumo.native.processing"),
        command,
        "--batch",
        str(
            positive(
                context.config("sumo_route_batch_size"),
                "sumo_route_batch_size",
                integer=True,
            )
        ),
        "--bytes",
        str(
            positive(
                context.config("sumo_route_batch_bytes"),
                "sumo_route_batch_bytes",
                integer=True,
            )
        ),
        "--threads",
        str(workers),
        "--xml-threads",
        str(
            min(
                allocated,
                positive(
                    context.config("sumo_xml_threads"), "sumo_xml_threads", integer=True
                ),
            )
        ),
    ]
    for key, value in paths.items():
        args += ["--" + key.replace("_", "-"), str(value)]
    subprocess.run(args, cwd=context.path(), check=True)
