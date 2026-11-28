"""Install SUMO separately from the project environment and run its tools."""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("synpp")

from scripts.monitoring import monitored
from sumo.functions.runtime.sumo import _binary_path, _environment
from sumo.functions.runtime.sumo import run as run
from sumo.functions.runtime.sumo import run_python as run_python
from sumo.functions.source import source_digest


def configure(context):
    context.config("sumo_version", "1.28.0")


@monitored
def execute(context):

    # Create a private environment inside this stage's cache.

    environment = Path(context.path()).resolve() / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        check=True,
        cwd=context.path(),
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Install the compiled applications and matching data without changing uv.

    version = context.config("sumo_version")
    subprocess.run(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--only-binary=:all:",
            f"eclipse-sumo=={version}",
            f"sumo-data=={version}",
        ],
        check=True,
        cwd=context.path(),
    )

    # Resolve the installed package in isolation from this repository's sumo package.

    home = Path(
        subprocess.check_output(
            [
                str(python),
                "-I",
                "-c",
                "import sumo; print(sumo.SUMO_HOME)",
            ],
            cwd=context.path(),
            text=True,
        ).strip()
    )
    settings = {"home": str(home), "python": str(python), "version": version}
    for binary in ("sumo", "netconvert"):
        subprocess.run(
            [str(_binary_path(settings, binary)), "--version"],
            check=True,
            env=_environment(settings),
            cwd=context.path(),
        )

    # Return paths for dependent stages; Synpp reuses the installation on later runs.

    return settings


def validate(context):
    """Invalidate a cached runtime if its environment or required tools disappear."""
    environment = Path(context.path()) / "venv"
    files = [environment / "pyvenv.cfg"]
    files.extend(environment.glob("**/sumo/bin/sumo*"))
    files.extend(environment.glob("**/sumo/bin/netconvert*"))
    return {
        "inputs": (
            [
                [
                    str(path.relative_to(environment)),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                ]
                for path in sorted(files)
                if path.is_file()
            ]
        ),
        "helpers": source_digest("sumo.functions.runtime.sumo"),
    }
