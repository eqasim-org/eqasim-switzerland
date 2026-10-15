"""Early Synpp dependency: cache launch settings, start Java without waiting."""
from pathlib import Path
import math
from numbers import Real
import shutil

from matsim.runtime.java import get_memories
from mode_choice.defaults import ROUTING_DEFAULTS, SERVER_DEFAULTS, ROUTER_SERVER_CLASS, RUN_DEFAULTS
from mode_choice.routing.RoutingServer import RoutingServer
from mode_choice.routing.congestion import input_path


def configure(context):
    context.stage("mode_choice.defaults")
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("matsim.scenario.network.mapped")
    context.stage("mode_choice.routing.config")
    context.stage("calibration.car_routing_vot.optimal_value")
    context.config("threads", volatile=True)
    context.config("car_cost_model", RUN_DEFAULTS["car_cost_model"])
    for name, default in {**ROUTING_DEFAULTS, **SERVER_DEFAULTS}.items():
        context.config(name, default)


def execute(context):
    # Resolve the runtime and create the server's private working directory.
    runtime = context.stage("matsim.runtime.java")
    binary = shutil.which(runtime["binary"])
    if binary is None:
        raise FileNotFoundError(f"Java binary not found: {runtime['binary']}")
    config = str(Path(context.stage("mode_choice.routing.config")).resolve())
    jar = str((Path(context.path("matsim.runtime.eqasim")) / context.stage("matsim.runtime.eqasim")).resolve())
    directory = Path(context.path()).resolve()
    temporary = directory / "java_tmp"
    temporary.mkdir(exist_ok=True)
    xmx, xms = get_memories(context, runtime, runtime["memory"])

    # Validate worker sizes, timeouts and the calibrated routing coefficient.
    sizes = {name: context.config(name) for name in ("threads", "mode_choice_routing_batch_size", "mode_choice_server_chunk_size")}
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    settings = {name: context.config("mode_choice_server_" + name) for name in
                ("startup_timeout", "request_timeout", "shutdown_timeout", "poll_interval")}
    for name, value in settings.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"mode_choice_server_{name} must be positive and finite")
    distance_utility = context.stage("calibration.car_routing_vot.optimal_value")
    if (isinstance(distance_utility, bool) or not isinstance(distance_utility, Real)
            or not math.isfinite(distance_utility) or distance_utility < 0):
        raise ValueError("calibration.car_routing_vot.optimal_value must return a finite, nonnegative routingDistanceUtility")
    # Explicit CLI values override the config in the current MultiModeRouterServer.
    # This coefficient is distinct from tolls-value-of-time/tolls-sigma, which
    # remain inherited from the supplied MATSim config.
    cost_model = context.config("car_cost_model").lower()
    if cost_model not in ("simple", "weiss"):
        raise ValueError(f"Unknown car_cost_model: {cost_model}")

    # Start asynchronously; routing stages wait for readiness when needed.
    command = [binary, "-cp", jar, "-Xmx" + xmx, "-Xms" + xms, "-XX:+UseParallelGC",
               "-Djava.io.tmpdir=" + str(temporary), "-Dmatsim.useLocalDtds=true", "-Djava.awt.headless=true",
               ROUTER_SERVER_CLASS, "--network-path", str(context.stage("matsim.scenario.network.mapped")["network"]),
               "--config-path", config, "--routingDistanceUtility", str(float(distance_utility)),
               "--compute-cost", "true", "--car-cost-model", cost_model,
               "--host", "127.0.0.1", "--port", "0",
               "--threads", str(sizes["threads"]), "--batch-size", str(sizes["mode_choice_routing_batch_size"]),
               "--chunk-size", str(sizes["mode_choice_server_chunk_size"])]
    events = input_path(context, "java") if context.config("mode_choice_car_routing_backend") == "java" else None
    if events is not None:
        command.extend(["--events-path", str(events)])
    settings.update(command=command, config_path=config, cwd=str(directory))
    RoutingServer.run_server_if_not_running(settings)
    return settings


def validate(context):
    # The server always provides PT; car events matter only for Java car routing.
    if context.config("mode_choice_car_routing_backend") != "java":
        return None
    path = input_path(context, "java")
    if path is not None:
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size, stat.st_ino


def ensure_server_started(context):
    # Synpp may load this stage's dictionary from disk in a new Python process.
    # Never mistake that cached dictionary for a live server.
    RoutingServer.run_server_if_not_running(context.stage("mode_choice.routing.server"))
