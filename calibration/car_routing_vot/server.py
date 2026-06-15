import json
import os
import shutil
import subprocess
import time
import urllib.request
import urllib.error
import logging

logger = logging.getLogger("synpp")

def start_server(context, network_path, events_path, port=8080, interval=900, routing_distance_utility=0.0):
    logger.info(f"\t Starting routing server with config: port={port}, interval={interval}, routing_distance_utility={routing_distance_utility}")
    # get java binary and eqasim jar file from context
    java_binary = context.stage("matsim.runtime.java")["binary"]
    java_memory = context.stage("matsim.runtime.java")["memory"]    
    eqasim_jar_path = "{}/{}".format(context.path("matsim.runtime.eqasim"), context.stage("matsim.runtime.eqasim"))
    cpu_count = len(os.sched_getaffinity(0))
    threads = int(max(min(context.config("threads"), cpu_count),1))

    # assertions
    assert java_binary is not None, "Java binary not found in context"
    assert os.path.exists(eqasim_jar_path), "Eqasim jar not found at path: {}".format(eqasim_jar_path)
    assert os.path.exists(network_path), "Network file not found at path: {}".format(network_path)
    assert os.path.exists(events_path), "Events file not found at path: {}".format(events_path)

    # Adjust paths for your machine.
    cmd = [
        shutil.which(java_binary),
        "-Xmx" + java_memory,
        "-cp",
        eqasim_jar_path,
        "org.eqasim.switzerland.ch_cmdp.utils.routing.TripsRouterServer",        
        "--network-path", network_path,
        "--events-path", events_path,
        "--port", str(port),
        "--threads", str(threads),
        "--interval", str(interval),
        "--routingDistanceUtility", str(routing_distance_utility),
    ]

    # Keep server running in background.
    return subprocess.Popen(cmd)


def route_trips(trips):
    wait_until_server_ready()

    req = urllib.request.Request(
        url="http://127.0.0.1:8080/route",
        data=json.dumps({"trips": trips}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=20*60) as resp:
        routed = json.loads(resp.read().decode("utf-8"))

    return routed


def set_server_config(routing_distance_utility=None, return_links=None):
    wait_until_server_ready()

    payload = {}
    if routing_distance_utility is not None:
        payload["routing_distance_utility"] = float(routing_distance_utility)
    if return_links is not None:
        payload["return_links"] = bool(return_links)

    req = urllib.request.Request(
        url="http://127.0.0.1:8080/config",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stop_server(server, wait_seconds=10):
    if server is None:
        return
    if server.poll() is not None:
        return

    server.terminate()
    try:
        server.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()


def wait_until_server_ready(base_url="http://127.0.0.1:8080", timeout_seconds=60*20, poll_seconds=2):
    start = time.time()
    health_url = f"{base_url}/health"

    while True:
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Server not ready after {timeout_seconds}s. Still waiting for startup/travel-time loading."
            )

        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8").strip().lower()
                if resp.status == 200 and body == "ok":
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass

        time.sleep(poll_seconds)
