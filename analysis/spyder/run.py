"""Launch with: python -m analysis.spyder.run CONFIG.yml_OR_OUTPUT_DIRECTORY."""

import argparse
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote

import yaml


def available_threads():
    """Respect process affinity, Slurm allocation, and container CPU quota."""
    limits = [len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1]
    for name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(name, "")
        if value.isdigit() and int(value) > 0:
            limits.append(int(value))
    # Locate the current cgroup, including ancestor quotas.
    roots = [Path("/sys/fs/cgroup")]
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                current = roots[0] / line.split("::", 1)[1].lstrip("/")
                roots.extend([current, *current.parents])
    except OSError:
        pass
    for root in roots:
        try:
            quota, period = (root / "cpu.max").read_text().split()
            if quota != "max":
                limits.append(max(1, math.floor(int(quota) / int(period))))
        except (OSError, ValueError):
            pass
    return max(1, min(limits))


def unique_file(folder, pattern):
    files = sorted(folder.glob(pattern))
    if len(files) != 1:
        raise ValueError(f"Expected one {pattern} in {folder}, found {len(files)}. Use an explicit path option.")
    return files[0].resolve()


def resolve_source(source, simulation_override=None):
    """Resolve a pipeline YAML or an exported scenario/results directory."""
    source = source.expanduser().resolve()
    if source.suffix.lower() in (".yml", ".yaml"):
        config = yaml.safe_load(source.read_text())
        if not isinstance(config, dict) or not isinstance(config.get("config", {}), dict):
            raise ValueError(f"Expected a pipeline YAML mapping in {source}")
        settings = config.get("config", {})
        if simulation_override is not None:
            simulation = simulation_override
        else:
            try:
                simulation = Path(settings["output_path"]) / str(settings["output_id"]) / settings.get("simulation_directory", "simulation_output")
            except KeyError as error:
                raise ValueError(f"Missing config.{error.args[0]} in {source}; use --simulation to select results") from error
        if "working_directory" not in config:
            raise ValueError(f"Missing working_directory in {source}")
        working_root = Path(config["working_directory"])
    else:
        if not source.is_dir():
            raise ValueError(f"Expected a .yml/.yaml config or an existing output directory: {source}")
        settings = {}
        working_root = Path.cwd()
        if simulation_override is not None:
            simulation = simulation_override
        elif (source / "simulation_output").is_dir():
            simulation = source / "simulation_output"
        elif any(source.glob("*output_events.xml*")):
            simulation = source
        else:
            candidates = sorted(p for p in source.glob("*/simulation_output") if p.is_dir())
            if len(candidates) != 1:
                raise ValueError(f"Expected simulation_output or one scenario containing it in {source}, found {len(candidates)}; select the scenario directory or use --simulation")
            simulation = candidates[0]
    simulation = simulation.expanduser().resolve()
    if not simulation.is_dir():
        raise ValueError(f"Simulation output directory does not exist: {simulation}")
    return settings, simulation, working_root.expanduser().resolve()



def discover_population(simulation, settings):
    """Prefer simulation person attributes; fall back to streamed population plans."""
    for stem, suffix in (("*output_persons", ".csv"), ("*output_plans", ".xml"), ("*output_population", ".xml")):
        candidates = sorted(p for compressed in ("", ".gz", ".zst") for p in simulation.glob(stem + suffix + compressed))
        if len(candidates) > 1:
            raise ValueError(f"Multiple population inputs in {simulation}; specify --population")
        if candidates:
            return candidates[0].resolve()
    prefix = settings.get("output_prefix", "")
    for folder in (simulation, simulation.parent):
        preferred = [folder / f"{prefix}population.xml{compressed}" for compressed in ("", ".gz", ".zst")]
        candidates = [p for p in preferred if p.is_file()]
        if not candidates:
            candidates = sorted(p for compressed in ("", ".gz", ".zst") for p in folder.glob("*population.xml" + compressed))
        if len(candidates) > 1:
            raise ValueError(f"Multiple population XMLs in {folder}; specify --population")
        if candidates:
            return candidates[0].resolve()
    return None



def ensure_backend(native, work, threads):
    """Use the existing executable without invoking Cargo when sources match."""
    digest = hashlib.sha256()
    for relative in ("Cargo.toml", "Cargo.lock", "src/main.rs"):
        digest.update(relative.encode())
        digest.update((native / relative).read_bytes())
    signature = {"sources": digest.hexdigest()}
    binary = work / "rust-target" / "release" / "eqasim-spyder"
    record = work / "rust-build.json"
    try:
        if binary.is_file() and os.access(binary, os.X_OK) and json.loads(record.read_text()) == signature:
            print("Reusing compiled Rust backend.", flush=True)
            return binary
    except (OSError, ValueError):
        pass
    cargo = shutil.which("cargo")
    if not cargo:
        raise RuntimeError("Rust cargo is required for the first build or source changes. Load the cluster Rust module, then run again.")
    env = dict(os.environ, CARGO_TARGET_DIR=str(work / "rust-target"), CARGO_BUILD_JOBS=str(threads))
    subprocess.run([cargo, "build", "--release", "--locked", "--manifest-path", str(native / "Cargo.toml")], env=env, check=True)
    record.write_text(json.dumps(signature, indent=2))
    return binary



def ensure_highways(binary, network, work, threads, prepared=False):
    """Upgrade old display caches by scanning only network link attributes once."""
    stat = network.stat()
    identity = {"schema": 1, "network": [str(network), stat.st_size, stat.st_mtime_ns]}
    manifest = work / "highways-manifest.json"
    try:
        if (work / "highways.json").exists() and json.loads(manifest.read_text()) == identity:
            return False
    except (OSError, ValueError):
        pass
    if not prepared:
        print("Caching highway types from the network once; processed event trips are reused.", flush=True)
        subprocess.run([str(binary), "highways", str(network), str(work), str(threads)], check=True)
    manifest.write_text(json.dumps(identity, indent=2))
    return True


def ensure_geometry_endpoints(binary, network, work, threads, prepared=False):
    """Close small shape/node gaps once, retaining processed trips and link indices."""
    stat = network.stat()
    identity = {"schema": 1, "network": [str(network), stat.st_size, stat.st_mtime_ns]}
    manifest = work / "geometry-endpoints-manifest.json"
    try:
        if json.loads(manifest.read_text()) == identity:
            return
    except (OSError, ValueError):
        pass
    if not prepared:
        print("Aligning cached geometry endpoints from the network once; event trips are reused.", flush=True)
        subprocess.run([str(binary), "align-endpoints", str(network), str(work), str(threads)], check=True)
    manifest.write_text(json.dumps(identity, indent=2))


def ensure_hierarchy(binary, work, threads, prepared=False):
    """Classify from cached highway types; never reopen events or raw network XML."""
    source = work / "highways.json"
    stat = source.stat()
    identity = {"schema": 1, "highways": [stat.st_size, stat.st_mtime_ns]}
    manifest = work / "hierarchy-manifest.json"
    try:
        if (work / "hierarchy.json").exists() and json.loads(manifest.read_text()) == identity:
            return
    except (OSError, ValueError):
        pass
    if not prepared or not (work / "hierarchy.json").exists():
        print("Classifying cached highway types for zoom levels (no raw data scan).", flush=True)
        subprocess.run([str(binary), "hierarchy", str(work), str(threads)], check=True)
    manifest.write_text(json.dumps(identity, indent=2))


def viewer_urls(host, port):
    """Return local/tunnel and Euler JupyterHub proxy URLs."""
    local_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if ":" in local_host:
        local_host = f"[{local_host}]"
    user = os.environ.get("JUPYTERHUB_USER") or getpass.getuser()
    prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX") or f"/user/{quote(user, safe='')}/"
    return {
        "Local browser / SSH tunnel": f"http://{local_host}:{port}/",
        "Euler JupyterHub": f"https://jupyter.euler.hpc.ethz.ch/{prefix.strip('/')}/proxy/{port}/",
    }



def run_server(binary, work, host, port, threads):
    """Print clickable links only after the cached-data server is listening."""
    command = [str(binary), "serve", str(work), f"{host}:{port}", str(threads)]
    print("Loading processed files and preparing the in-memory query index…", flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("Viewer ready at "):
                    for label, url in viewer_urls(host, port).items():
                        print(f"Click here ({label}): {url}", flush=True)
                    print("For Euler JupyterHub, run this command in a terminal on your Jupyter session's node.", flush=True)
                    print(f"For SSH access: ssh -L {port}:127.0.0.1:{port} <compute-host>", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Pipeline .yml/.yaml config, exported scenario directory, or simulation_output directory")
    parser.add_argument("--simulation", type=Path, help="Override the simulation output directory")
    parser.add_argument("--events", type=Path)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--population", type=Path, help="Override output_persons CSV or population XML (optionally gzip or Zstandard compressed)")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--data-path", type=Path, help="Spatial input root for the offline background, overriding config.data_path")
    parser.add_argument("--threads", type=int, help="Upper bound; clamped to the allocation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Reprocess inputs even if unchanged")
    args = parser.parse_args()
    try:
        settings, simulation, working_root = resolve_source(args.source, args.simulation)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.data_path is not None:
        settings["data_path"] = str(args.data_path.expanduser().resolve())
    print(f"Simulation output: {simulation}", flush=True)
    events = args.events.expanduser().resolve() if args.events else unique_file(simulation, "*output_events.xml*")
    network = args.network.expanduser().resolve() if args.network else unique_file(simulation, "*output_network.xml*")
    population = args.population.expanduser().resolve() if args.population else discover_population(events.parent, settings)
    if population is None:
        print("No persons CSV or population XML found; population filters will show unknown trips only.", flush=True)
    else:
        print(f"Population attributes: {population}", flush=True)
    geometry = args.geometry.expanduser().resolve() if args.geometry else None
    if geometry is None:
        for folder in (network.parent, network.parent.parent):
            candidates = sorted(folder.glob("*detailed_network.csv*"))
            if len(candidates) > 1:
                raise ValueError(f"Multiple geometry CSVs in {folder}; specify --geometry")
            if candidates:
                geometry = candidates[0]
                break
    if geometry is None:
        print("No detailed geometry found; using network node endpoints.", flush=True)
    allocation = available_threads()
    requested = args.threads if args.threads is not None else int(settings.get("threads", allocation))
    if requested < 1:
        parser.error("--threads must be positive")
    threads = min(allocation, requested)
    key = hashlib.sha256(str(events).encode()).hexdigest()[:12]
    work = (args.work_dir or working_root / "analysis_spyder" / key).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    native = Path(__file__).with_name("native")
    print(f"Using {threads} threads (allocation limit: {allocation}). Working directory: {work}", flush=True)
    binary = ensure_backend(native, work, threads)
    signature = {"schema": 2, "inputs": [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in (network, events, geometry, population) if p]}
    manifest = work / "manifest.json"
    try:
        stored = json.loads(manifest.read_text())
        cached = all(stored.get(key) == value for key, value in signature.items()) and all((work / n).exists() for n in ("network.json", "trips.json"))
    except (OSError, ValueError):
        cached = False
    if args.rebuild or not cached:
        print("Preparing raw inputs (--rebuild requested or processed cache is missing/outdated).", flush=True)
        subprocess.run([str(binary), "prepare", str(network), str(events), str(geometry) if geometry else "-", str(work), str(population) if population else "-", str(threads)], check=True)
        manifest.write_text(json.dumps(signature, indent=2))
    else:
        print("Loading processed data; raw events and population will not be reread.", flush=True)
    ensure_geometry_endpoints(binary, network, work, threads, prepared=args.rebuild or not cached)
    metadata_prepared = ensure_highways(binary, network, work, threads, prepared=args.rebuild or not cached)
    ensure_hierarchy(binary, work, threads, prepared=metadata_prepared or args.rebuild or not cached)
    from analysis.spyder.background import prepare_background
    prepare_background(settings, work)
    for name in ("viewer.html", "viewer.worker.js", "viewer.runtime.js"):
        shutil.copyfile(Path(__file__).with_name(name), work / ("index.html" if name == "viewer.html" else name))
    if not args.prepare_only:
        run_server(binary, work, args.host, args.port, threads)


if __name__ == "__main__":
    main()
