"""Validate Java route-export capabilities needed for SUMO border clipping."""

import zipfile

from mode_choice.defaults import ROUTER_SERVER_CLASS


def validate_sumo_runtime(jar):
    try:
        with zipfile.ZipFile(jar) as archive:
            data = archive.read(ROUTER_SERVER_CLASS.replace(".", "/") + ".class")
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("prepare_sumo requires a valid Java runtime containing MultiModeRouterServer") from error
    if b"export-routes" not in data or b"marouter" not in data:
        raise ValueError("The configured eqasim JAR predates reusable route export; set eqasim_path: '' "
                         "to build the pinned runtime, or provide a compatible JAR")
