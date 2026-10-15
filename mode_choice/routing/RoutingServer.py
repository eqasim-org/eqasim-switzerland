"""Process-local owner and CSV/HTTP client for eqasim's multimodal router.

Only serializable launch settings belong in Synpp's cache, never a process or a
readiness flag. The live process is reused within this Python invocation.
"""
import atexit
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, ProxyHandler, build_opener

from mode_choice.defaults import (
    SERVER_HEALTH_TIMEOUT, ROUTER_SERVER_MODES, SERVER_LOG_FILENAME_FORMAT,
)


class RoutingServer:
    _process = None
    _settings = None
    _lock = threading.RLock()
    server_endpoint = None
    config_path = None
    log_path = None

    @staticmethod
    def run_server_if_not_running(settings):
        """Start immediately without waiting for Java or scenario initialization."""
        with RoutingServer._lock:
            if RoutingServer._process is not None:
                if RoutingServer._settings != settings:
                    raise RuntimeError("A routing server with different settings is already owned by this process")
                RoutingServer._check_process()
                return
            RoutingServer.start_server(settings)

    @staticmethod
    def start_server(settings):
        with RoutingServer._lock:
            if RoutingServer._process is not None:
                raise RuntimeError("Routing server is already started")
            directory = Path(settings["cwd"])
            directory.mkdir(parents=True, exist_ok=True)
            # Choose a fresh file on every actual launch, including cache reuse.
            # Keep it out of cached settings so idempotent startup still matches.
            log_path = directory / datetime.now(timezone.utc).strftime(SERVER_LOG_FILENAME_FORMAT)
            with log_path.open("xb") as log:
                process = subprocess.Popen(settings["command"], cwd=directory, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            RoutingServer.log_path = str(log_path)
            RoutingServer._settings = settings
            RoutingServer._process = process
            RoutingServer.config_path = settings["config_path"]
            RoutingServer.server_endpoint = None

    @staticmethod
    def _check_process():
        process = RoutingServer._process
        if process is None:
            raise RuntimeError("Routing server has not been started")
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Routing server exited with code {code}; see {RoutingServer.log_path}")

    @staticmethod
    def _endpoint():
        RoutingServer._check_process()
        if RoutingServer.server_endpoint is None:
            # Port zero lets Java bind an unused port atomically. Read its startup
            # announcement, without polling or blocking in the startup stage.
            with open(RoutingServer.log_path, "rb") as log:
                for line in log:
                    match = re.search(rb"Multi-mode router listening on http://[^\s:]+:(\d+)", line)
                    if match:
                        RoutingServer.server_endpoint = "http://127.0.0.1:" + match[1].decode()
                        break
        return RoutingServer.server_endpoint

    @staticmethod
    def _request(path, payload=None, timeout=SERVER_HEALTH_TIMEOUT):
        endpoint = RoutingServer._endpoint()
        if endpoint is None:
            return None
        request = Request(endpoint + path, data=None if payload is None else json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            body = json.loads(exc.read())
            if path == "/ready" and exc.code == 503:
                return body
            raise RuntimeError(f"Routing server HTTP {exc.code}: {body}") from exc

    @staticmethod
    def is_server_running():
        return RoutingServer._process is not None and RoutingServer._process.poll() is None

    @staticmethod
    def is_server_ready():
        if not RoutingServer.is_server_running():
            return False
        try:
            state = RoutingServer._request("/ready")
        except (URLError, TimeoutError):
            return False
        if state and state.get("status") == "failed":
            raise RuntimeError(f"Routing server initialization failed: {state.get('error')}")
        return bool(state and state.get("ready"))

    @staticmethod
    def wait_until_ready(mode=None):
        RoutingServer._check_process()
        settings = RoutingServer._settings
        deadline = time.monotonic() + settings["startup_timeout"]
        while True:
            RoutingServer._check_process()
            if RoutingServer.is_server_ready():
                state = RoutingServer._request("/ready")
                if mode is not None and not state.get("modes", {}).get(mode, {}).get("ready"):
                    raise RuntimeError(f"Routing server does not provide mode {mode}: {state}")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Routing server initialization timed out; see {RoutingServer.log_path}")
            time.sleep(settings["poll_interval"])

    @staticmethod
    def route_csv(input_csv_path, output_csv_path, mode=None, *, compute_cost=False):
        return RoutingServer._csv_request("/route", input_csv_path, output_csv_path, mode,
                                          compute_cost=compute_cost)

    @staticmethod
    def cost_csv(input_csv_path, output_csv_path, mode=None):
        """Price saved routes or distances using the already initialized workers."""
        return RoutingServer._csv_request("/cost", input_csv_path, output_csv_path, mode)

    @staticmethod
    def _csv_request(endpoint, input_csv_path, output_csv_path, mode, *, compute_cost=None):
        if mode == "car_passenger":
            mode = "car"
        if mode is not None and mode not in ROUTER_SERVER_MODES:
            raise ValueError(f"Unsupported routing mode: {mode}")
        RoutingServer.wait_until_ready(mode)
        payload = {"inputCsv": str(Path(input_csv_path).resolve()),
                   "outputCsv": str(Path(output_csv_path).resolve())}
        if mode is not None:
            payload["mode"] = mode
        if compute_cost is not None:
            payload["computeCost"] = compute_cost
        result = RoutingServer._request(endpoint, payload, RoutingServer._settings["request_timeout"])
        if result is None or not Path(output_csv_path).is_file():
            raise RuntimeError("Routing server did not produce the requested CSV")
        return result

    @staticmethod
    def route_batch(origins, destinations, mode, departure_times=None):
        """Convenience API for coordinate pairs; population routing uses route_csv."""
        if len(origins) != len(destinations):
            raise ValueError("Origins and destinations must have equal lengths")
        times = [0.0] * len(origins) if departure_times is None else departure_times
        if len(times) != len(origins):
            raise ValueError("Expected one departure time per trip")
        with tempfile.TemporaryDirectory(prefix="eqasim-routing-") as directory:
            source, target = Path(directory) / "input.csv", Path(directory) / "output.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["identifier", "origin_x", "origin_y", "destination_x", "destination_y", "departure_time"])
                for i, (origin, destination, departure) in enumerate(zip(origins, destinations, times)):
                    writer.writerow([str(i), *origin, *destination, departure])
            RoutingServer.route_csv(source, target, mode)
            with target.open(newline="") as stream:
                return list(csv.DictReader(stream))

    @staticmethod
    def route(origin, destination, mode, departure_time=0.0):
        return RoutingServer.route_batch([origin], [destination], mode, [departure_time])[0]

    @staticmethod
    def stop_server():
        with RoutingServer._lock:
            process = RoutingServer._process
            if process is None:
                return
            try:
                if process.poll() is None:
                    process.terminate()  # Java shutdown hook closes workers and HTTP.
                    try:
                        process.wait(timeout=RoutingServer._settings["shutdown_timeout"])
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                else:
                    process.wait()
            finally:
                RoutingServer._process = RoutingServer._settings = None
                RoutingServer.server_endpoint = RoutingServer.config_path = None


atexit.register(RoutingServer.stop_server)
