"""Install SUMO separately from the project environment and run its tools."""

import logging
import os
from pathlib import Path
import subprocess
import sys

logger = logging.getLogger("synpp")


def configure(context):
    context.config("sumo_version", "1.28.0")


def execute(context):

    # Create a private environment inside this stage's cache.

    environment = Path(context.path()).resolve() / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)], check=True,
        cwd=context.path(),
    )
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    # Install the compiled applications and matching data without changing uv.

    version = context.config("sumo_version")
    subprocess.run([
        str(python), "-I", "-m", "pip", "--disable-pip-version-check", "install",
        "--only-binary=:all:", f"eclipse-sumo=={version}", f"sumo-data=={version}",
    ], check=True, cwd=context.path())

    # Resolve the installed package in isolation from this repository's sumo package.

    home = Path(subprocess.check_output([
        str(python), "-I", "-c", "import sumo; print(sumo.SUMO_HOME)",
    ], cwd=context.path(), text=True).strip())
    settings = {"home": str(home), "python": str(python), "version": version}
    for binary in ("sumo", "netconvert"):
        subprocess.run(
            [str(_binary_path(settings, binary)), "--version"],
            check=True, env=_environment(settings), cwd=context.path(),
        )

    # Return paths for dependent stages; Synpp reuses the installation on later runs.

    return settings


def run(context, arguments=None, binary="sumo", cwd=None, catch_output=False, executable=None):
    """Run an installed SUMO tool, raising on failure.

    Consumers must declare ``context.stage("sumo.runtime.sumo")`` in configure.
    ``executable`` can select a separately cached native build.
    Return the exit code, or decoded stdout when catch_output is True.
    """
    settings = context.stage("sumo.runtime.sumo")
    command = [str(_binary_path(settings, binary) if executable is None else Path(executable).resolve())]
    command.extend(str(argument) for argument in (arguments or ()))
    logger.info("Executing SUMO tool: %s", command)
    result = subprocess.run(
        command, cwd=context.path() if cwd is None else cwd,
        env=_environment(settings), check=True, text=True,
        stdout=subprocess.PIPE if catch_output else None,
    )
    return result.stdout if catch_output else result.returncode


def run_python(context, script, arguments=None, cwd=None):
    """Run an installed SUMO helper with its isolated Python and bundled libraries."""
    settings = context.stage("sumo.runtime.sumo")
    tools = (Path(settings["home"]) / "tools").resolve()
    filename = (tools / script).resolve()
    if not filename.is_relative_to(tools) or not filename.is_file():
        raise ValueError(f"Unknown installed SUMO Python helper: {script}")
    environment = _environment(settings)
    bootstrap = ("import os,runpy,sys; path=sys.argv[1]; sys.path.insert(0,os.path.dirname(path)); "
                 "sys.argv=sys.argv[1:]; runpy.run_path(path,run_name='__main__')")
    return subprocess.run([settings["python"], "-I", "-c", bootstrap, str(filename), *map(str, arguments or ())],
                          cwd=context.path() if cwd is None else cwd, env=environment, check=True).returncode


def validate(context):
    """Invalidate a cached runtime if its environment or required tools disappear."""
    environment = Path(context.path()) / "venv"
    files = [environment / "pyvenv.cfg"]
    files.extend(environment.glob("**/sumo/bin/sumo*"))
    files.extend(environment.glob("**/sumo/bin/netconvert*"))
    return [[str(path.relative_to(environment)), path.stat().st_size,
             path.stat().st_mtime_ns] for path in sorted(files) if path.is_file()]


def _binary_path(settings, binary):
    if not binary or Path(binary).name != binary or binary in {".", ".."}:
        raise ValueError("binary must be the name of a SUMO tool, such as 'sumo' or 'netconvert'")
    suffix = ".exe" if os.name == "nt" else ""
    path = Path(settings["home"]) / "bin" / (binary + suffix)
    if not path.is_file():
        raise FileNotFoundError(f"SUMO tool not found: {path}")
    return path


def _environment(settings):
    environment = os.environ.copy()
    environment["SUMO_HOME"] = settings["home"]
    environment["PATH"] = str(Path(settings["home"]) / "bin") + os.pathsep + environment.get("PATH", "")
    projection_data = Path(settings["home"]) / "data" / "proj"
    if (projection_data / "proj.db").is_file():
        environment.setdefault("PROJ_DATA", str(projection_data))
        environment.setdefault("PROJ_LIB", str(projection_data))
    return environment
