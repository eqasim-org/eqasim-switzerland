"""Build or reuse the latest cached network inspector, then start its server."""
import argparse
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import pickle

import yaml

from .cleaning_process import build_inspector
from .cleaning_server import serve

RAW_STAGE = "matsim.scenario.network.convert_osm_pt2matsim"
CLEAN_STAGE = "matsim.scenario.network.convert_osm"
logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=Path("config_dib_10.yml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--build-only", action="store_true", help="Prepare the viewer cache and exit")
    parser.add_argument("--show-sources", action="store_true", help="Show selected stages/files without building")
    args = parser.parse_args(argv)
    plan = resolve_plan(args.config)
    if args.show_sources:
        print(json.dumps(plan, indent=2), flush=True)
        return
    directory = Path(plan["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".viewer.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This dataset already has a viewer or build running. Stop that process before restarting.") from error
        if args.build_only:
            prepare_data(plan)
        else:
            serve(directory, host=args.host, port=args.port, prepare=lambda: prepare_data(plan))


def source_identity(sources):
    identity = {}
    for name, value in sources.items():
        path = Path(value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Cached {name} file is missing: {path}. Rerun its producer stage.")
        stat = path.stat()
        identity[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return identity


def resolve_plan(config_path):
    # Resolve the newest completed cleaned output, using pipeline metadata rather
    # than cache-directory modification times (which change during inspections).
    settings = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(settings, dict) or not settings.get("working_directory"):
        raise ValueError("The configuration must define a top-level working_directory")
    working = Path(settings["working_directory"]).expanduser().resolve()
    metadata_path = working / "pipeline.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No Synpp metadata at {metadata_path}. Run the network stages first.")
    metadata = json.loads(metadata_path.read_text())
    candidates = sorted((key for key in metadata if key.startswith(CLEAN_STAGE+"__")),
                        key=lambda key: (metadata[key].get("updated", 0), key), reverse=True)
    cleaned_key = None
    for key in candidates:
        result_path = working / (key+".p")
        if result_path.is_file():
            with result_path.open("rb") as stream:
                cleaned = pickle.load(stream)
            if isinstance(cleaned, str) and Path(cleaned).is_file():
                cleaned_key = key
                break
        logger.warning("Skipping incomplete cleaned-network cache: %s", key)
    if cleaned_key is None:
        raise FileNotFoundError("No completed convert_osm output in this working directory. Run the network stages first.")

    # The paired raw network must be the recorded dependency of this exact run.
    dependencies = metadata[cleaned_key].get("dependencies", {})
    raw_keys = [key for key in dependencies if key.startswith(RAW_STAGE+"__")]
    if len(raw_keys) != 1:
        raise ValueError(f"Expected one raw-network dependency for {cleaned_key}; found {len(raw_keys)}")
    raw_key = raw_keys[0]
    if raw_key not in metadata or metadata[raw_key].get("updated") != dependencies[raw_key]:
        raise RuntimeError("The paired raw network is being rebuilt or was updated after convert_osm. Finish/rerun convert_osm before opening this dataset.")
    with (working / (raw_key+".p")).open("rb") as stream:
        raw = pickle.load(stream)
    if not isinstance(raw, (tuple, list)) or len(raw) != 2:
        raise ValueError(f"Invalid raw-network result for {raw_key}: expected network and geometry paths")
    sources = {"before": str(Path(raw[0]).resolve()), "before_geometry": str(Path(raw[1]).resolve()),
               "after": str(Path(cleaned).resolve()),
               "after_geometry": str(working / (cleaned_key+".cache") / "detailed_network.csv")}
    audit = working / (cleaned_key+".cache") / "cleaning_events.jsonl.gz"
    if audit.is_file():
        sources["audit"] = str(audit)
    identity = source_identity(sources)
    # Separate datasets can be served independently without replacing the files
    # used by an older viewer. Builder/preprocessor hashes control reuse inside it.
    network_identity = {name: value for name, value in identity.items() if name != "audit"}
    fingerprint = hashlib.sha256(json.dumps(network_identity, sort_keys=True).encode()).hexdigest()[:24]
    directory = working / "analysis.network.run.cache" / fingerprint
    options = {}
    config = flatten_options(settings.get("config", {}))
    for option, argument in (("center", "center"), ("zoom", "zoom"),
                             ("offline_radius_m", "offline_radius"), ("max_features", "max_features")):
        key = "analysis.network.cleaning_process."+option
        if key in config:
            options[argument] = config[key]
    return {"directory": str(directory), "stages": {"before": raw_key, "after": cleaned_key},
            "sources": sources, "identity": identity, "options": options}


def flatten_options(values, prefix=""):
    result = {}
    for name, value in values.items():
        name = prefix+name
        if isinstance(value, dict):
            result.update(flatten_options(value, name+"."))
        else:
            result[name] = value
    return result


def prepare_data(plan):
    logger.info("Cleaned stage: %s", plan["stages"]["after"])
    logger.info("Paired raw stage: %s", plan["stages"]["before"])
    logger.info("Viewer cache: %s", plan["directory"])
    if "audit" not in plan["sources"]:
        logger.warning("This network has no cleaning audit; removed links will be hidden.")
    result = build_inspector(output=plan["directory"], **plan["sources"], **plan["options"])
    if source_identity(plan["sources"]) != plan["identity"]:
        raise RuntimeError("Network inputs changed while building. Wait for the producer stage to finish and restart the viewer.")
    manifest_path = Path(plan["directory"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["producer_stages"] = plan["stages"]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
