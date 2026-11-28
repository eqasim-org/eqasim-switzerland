"""Build the Arrow-to-SUMO XML writer once in a Synpp cache."""

import hashlib
import platform
from pathlib import Path

import pyarrow as pa

from scripts.monitoring import monitored
from sumo.functions.native.runtime import build_writer
from sumo.functions.source import source_digest


def configure(context):
    pass


@monitored
def execute(context):

    # Compile against the Arrow headers and library already installed with PyArrow.

    return {"writer": build_writer(Path(context.path()) / "write_trips")}


def validate(context):
    return {
        "inputs": (
            [
                hashlib.sha256(
                    (
                        Path(__file__).parents[1]
                        / "functions"
                        / "native"
                        / "write_trips.cpp"
                    ).read_bytes()
                ).hexdigest(),
                pa.__version__,
                platform.machine(),
            ]
        ),
        "helpers": source_digest("sumo.functions.native.runtime"),
    }
