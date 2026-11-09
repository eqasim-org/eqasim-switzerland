"""Validate the shared server API before starting a long population routing job."""
import zipfile
from mode_choice.defaults import ROUTER_SERVER_CLASS

SERVER_IMPLEMENTATION = "org/eqasim/switzerland/ch_cmdp/utils/routing/multimodalRouter/server/MultiModeRouterServer.class"


def validate_runtime(jar):
    try:
        with zipfile.ZipFile(jar) as archive:
            archive.getinfo(ROUTER_SERVER_CLASS.replace(".", "/") + ".class")
            implementation = archive.read(SERVER_IMPLEMENTATION)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("The eqasim JAR lacks multimodalRouter.RunServer. Set eqasim_path: '' "
                         "to build the pinned commit, or supply an updated JAR.") from error
    if not all(option in implementation for option in
               (b"marouter-iterations", b"use-rust-pt-router", b"initialization-log-file")):
        raise ValueError("The eqasim JAR lacks the required iterative assignment, Rust PT or logging API")
