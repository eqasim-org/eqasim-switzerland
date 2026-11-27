"""Build and run bounded Arrow/XML processing outside the Python interpreter."""

import platform
from pathlib import Path

import pyarrow as pa

from sumo.functions.monitoring.runtime import monitored
from sumo.functions.native.processing import build, source_token
from sumo.functions.native.processing import configure_options as configure_options
from sumo.functions.native.processing import configure_stage as configure_stage
from sumo.functions.native.processing import run as run
from sumo.functions.source import source_digest


def configure(context):
    command = context.config("sumo_native_processing_command", "all")
    if command not in {"all", "index", "realize", "cut", "export"}:
        raise ValueError("Unknown SUMO native processing component")


@monitored
def execute(context):

    # Compile against the existing Arrow wheel, system Expat and zlib.

    return build(
        Path(context.path()) / "process_routes",
        context.config("sumo_native_processing_command"),
    )


def validate(context):
    return {
        "inputs": (
            [
                source_token(context.config("sumo_native_processing_command")),
                pa.__version__,
                platform.machine(),
            ]
        ),
        "helpers": source_digest("sumo.functions.native.processing"),
    }
