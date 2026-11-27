"""Build class-aware marouter without modifying the installed SUMO runtime.

SUMO 1.28.0 retains types in OD cells but routes them as DEFAULT_VEHTYPE.
The accompanying EPL-2.0/GPL-2.0-or-later patch uses each cell's vehicle type
and includes it in the bulk-routing key. All classes share edge flow costs.
"""

import hashlib
import logging
import os
import shutil
import subprocess
from pathlib import Path

SOURCE_SHA256 = "2f4d080fdfab9ab55e4e8fc18e28230b18155ed5b4b602fe238e5c623a80d03a"

PATCH = Path(__file__).with_suffix(".patch")

logger = logging.getLogger("synpp")


def build(source, directory, threads):
    """Compile with CMake, Xerces-C and FOX available in the launch environment."""
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "Building class-aware marouter requires CMake and Xerces-C; see run_sumo.sh modules"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    disabled = (
        "PROJ",
        "PARQUET",
        "EIGEN",
        "FMT",
        "GTEST",
        "NETEDIT",
        "PYTHON_BINDINGS",
        "JAVA_BINDINGS",
        "CS_BINDINGS",
        "FMI",
    )
    command = [
        cmake,
        "-S",
        str(source),
        "-B",
        str(directory),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCHECK_OPTIONAL_LIBS=OFF",
        "-DENABLE_FOX=ON",
    ]
    fox = shutil.which("fox-config")
    if fox is not None:
        prefix = Path(subprocess.check_output([fox, "--prefix"], text=True).strip())
        library = prefix / "lib" / "libFOX-1.6.so"
        if library.is_file():
            # fox-config emits static GUI dependencies that marouter never uses.
            command += [
                f"-DFOX_LIBRARY={library}",
                f"-DCMAKE_BUILD_RPATH={library.parent}",
            ]
    command += [f"-DENABLE_{option}=OFF" for option in disabled]
    for name, arguments in (
        ("configure", command),
        (
            "build",
            [
                cmake,
                "--build",
                str(directory),
                "--target",
                "marouter",
                "--parallel",
                str(max(1, int(threads))),
            ],
        ),
    ):
        log = directory / f"{name}.log"
        logger.info("Native marouter %s; diagnostics: %s", name, log)
        with log.open("w") as stream:
            result = subprocess.run(arguments, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(
                f"Native marouter {name} failed; see {log}\n" + log.read_text()[-4000:]
            )
        if name == "configure":
            config = (directory / "src" / "config.h").read_text()
            if not any(
                line.strip().startswith("#define HAVE_FOX")
                for line in config.splitlines()
            ):
                raise RuntimeError(
                    f"FOX routing threads unavailable; load fox and mesa-glu modules. See {log}"
                )
    return Path(source) / "bin" / ("marouter.exe" if os.name == "nt" else "marouter")


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _build_members(archive):
    """Skip large test/documentation trees when building only marouter."""
    for member in archive:
        parts = Path(member.name).parts
        if (
            len(parts) <= 2
            or parts[1] in {"src", "build_config"}
            or (
                len(parts) >= 3
                and parts[1] == "tools"
                and parts[2] in {"build_config", "sumolib", "traci"}
            )
        ):
            yield member
