"""Compile independent route arithmetic kernels; no SUMO or Arrow build required."""

import hashlib
import platform
from pathlib import Path

from scripts.monitoring import monitored
from sumo.functions.native.routes import build
from sumo.functions.source import source_digest


def configure(context):
    pass


@monitored
def execute(context):

    # Compile the small C ABI library once, in its own cache stage.

    return build(Path(context.path()) / "route_kernel.so")


def validate(context):
    return {
        "inputs": (
            [
                hashlib.sha256(
                    (
                        Path(__file__).parents[1]
                        / "functions"
                        / "native"
                        / "route_kernel.cpp"
                    ).read_bytes()
                ).hexdigest(),
                platform.machine(),
            ]
        ),
        "helpers": source_digest("sumo.functions.native.routes"),
    }
