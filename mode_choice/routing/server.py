"""Early Synpp dependency: cache launch settings, start Java without waiting."""
from pathlib import Path
import math
import shutil

from matsim.runtime.java import get_memories
from mode_choice.defaults import ROUTING_DEFAULTS, SERVER_DEFAULTS, ROUTER_SERVER_CLASS
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.routing.congestion import input_path


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("matsim.scenario.network.mapped")
    context.stage("calibration.pt_pricing.generate_config")
    context.config("threads", volatile=True)
    for name, default in {**ROUTING_DEFAULTS, **SERVER_DEFAULTS}.items():
        context.config(name, default)


def validate(context):
    # The server always provides PT; car events matter only for Java car routing.
    if context.config("mode_choice_car_routing_backend") != "java":
        return None
    path = input_path(context, "java")
    if path is not None:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino


def execute(context):
    runtime = context.stage("matsim.runtime.java")
    binary = shutil.which(runtime["binary"])
    if binary is None:
        raise FileNotFoundError(f"Java binary not found: {runtime['binary']}")
    config = str(Path(context.stage("calibration.pt_pricing.generate_config")).resolve())
    jar = str((Path(context.path("matsim.runtime.eqasim")) / context.stage("matsim.runtime.eqasim")).resolve())
    directory = Path(context.path()).resolve()
    temporary = directory / "java_tmp"
    temporary.mkdir(exist_ok=True)
    xmx, xms = get_memories(context, runtime, runtime["memory"])
    sizes = {name: context.config(name) for name in ("threads", "mode_choice_routing_batch_size", "mode_choice_server_chunk_size")}
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    settings = {name: context.config("mode_choice_server_" + name) for name in
                ("startup_timeout", "request_timeout", "shutdown_timeout", "poll_interval")}
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"mode_choice_server_{name} must be positive and finite")
    command = [binary, "-cp", jar, "-Xmx" + xmx, "-Xms" + xms, "-XX:+UseParallelGC",
               "-Djava.io.tmpdir=" + str(temporary), "-Dmatsim.useLocalDtds=true", "-Djava.awt.headless=true",
               ROUTER_SERVER_CLASS, "--network-path", str(context.stage("matsim.scenario.network.mapped")["network"]),
               "--config-path", config, "--host", "127.0.0.1", "--port", "0",
               "--threads", str(sizes["threads"]), "--batch-size", str(sizes["mode_choice_routing_batch_size"]),
               "--chunk-size", str(sizes["mode_choice_server_chunk_size"])]
    events = input_path(context, "java") if context.config("mode_choice_car_routing_backend") == "java" else None
    if events is not None:
        command.extend(["--events-path", str(events)])
    settings.update(command=command, config_path=config, cwd=str(directory))
    RoutingServer.run_server_if_not_running(settings)
    return settings


def ensure_started(context):
    # Synpp may load this stage's dictionary from disk in a new Python process.
    # Never mistake that cached dictionary for a live server.
    RoutingServer.run_server_if_not_running(context.stage("mode_choice.routing.server"))
