"""Build class-aware marouter without modifying the installed SUMO runtime.

SUMO 1.28.0 retains types in OD cells but routes them as DEFAULT_VEHTYPE.
The accompanying EPL-2.0/GPL-2.0-or-later patch uses each cell's vehicle type
and includes it in the bulk-routing key. All classes share edge flow costs.
"""

import logging
import shutil
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

from sumo.functions.monitoring.runtime import monitored
from sumo.functions.native.marouter import (
    PATCH,
    SOURCE_SHA256,
    _build_members,
    _digest,
    build,
)

logger = logging.getLogger("synpp")


from sumo.functions.source import source_digest


def configure(context):
    context.config("sumo_version", "1.28.0")
    context.config("threads", volatile=True)


@monitored
def execute(context):

    # Download and verify the exact source corresponding to the installed tools.

    if context.config("sumo_version") != "1.28.0":
        raise ValueError("Class-aware marouter currently requires sumo_version: 1.28.0")
    cache = Path(context.path()).resolve()
    archive = cache / "source.tar.gz"
    url = "https://codeload.github.com/eclipse-sumo/sumo/tar.gz/refs/tags/v1_28_0"
    logger.info("Preparing class-aware marouter from pinned SUMO 1.28.0 source")
    if not archive.is_file():
        partial = archive.with_suffix(".part")
        with urlopen(url, timeout=600) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(archive)
    if _digest(archive) != SOURCE_SHA256:
        archive.unlink()
        raise ValueError("SUMO source checksum mismatch; unverified source was removed")

    # Patch a private source tree and compile only the assignment executable.

    source = cache / "source" / "sumo-1_28_0"
    if source.exists():
        shutil.rmtree(source)
    with tarfile.open(archive) as compressed:
        compressed.extractall(
            cache / "source", members=_build_members(compressed), filter="data"
        )
    subprocess.run(
        ["patch", "--batch", "-p1", "-i", str(PATCH)], cwd=source, check=True
    )
    output = build(source, cache / "build", context.config("threads"))

    # Publish the executable and retain build diagnostics in this stage's cache.

    destination = cache / "marouter"
    shutil.copy2(output, destination)
    return str(destination)


def validate(context):
    output = Path(context.path()) / "marouter"
    return {
        "inputs": (
            {
                "patch": _digest(PATCH),
                "binary": _digest(output) if output.is_file() else None,
            }
        ),
        "helpers": source_digest("sumo.functions.native.marouter"),
    }
