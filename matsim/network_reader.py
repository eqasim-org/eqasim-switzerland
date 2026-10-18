"""Shared binary network cache and optional compiled XML parser."""
from functools import lru_cache
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import sysconfig
import tempfile

import fcntl
import pandas as pd
from pandas.api.types import is_numeric_dtype

logger = logging.getLogger(__name__)
CACHE_VERSION = 1


@lru_cache(maxsize=None)
def _native_parser(build_root):
    """Compile once, outside the source tree, with a Python fallback on failure."""
    source = Path(__file__).with_name("native") / "network.cpp"
    digest = hashlib.sha256(source.read_bytes() + sys.version.encode()
                            + str(sysconfig.get_config_var("SOABI")).encode()).hexdigest()[:20]
    directory = Path(build_root) / "network_native" / digest
    extension = directory / ("_network_native" + sysconfig.get_config_var("EXT_SUFFIX"))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "build.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not extension.exists():
                compiler = shlex.split(os.environ.get("CXX", "g++"))
                with tempfile.TemporaryDirectory(dir=directory) as temporary:
                    target = Path(temporary) / extension.name
                    command = compiler + ["-O3", "-shared", "-fPIC", "-std=c++17",
                                          "-I" + sysconfig.get_paths()["include"], str(source),
                                          "-o", str(target), "-lexpat", "-lz"]
                    logger.info("Building native network reader in %s", directory)
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    os.replace(target, extension)
        spec = importlib.util.spec_from_file_location("matsim._network_native", extension)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except (OSError, subprocess.CalledProcessError, ImportError) as error:
        logger.warning("Native network reader unavailable; using Python: %s",
                       getattr(error, "stderr", None) or error)
        return None


def _read_xml(filename, skip_attributes, build_root):
    from matsim.readers import Network, _read_network_xml

    native = _native_parser(str(build_root))
    # zlib handles plain XML and gzip; retain xopen support for other formats.
    if native is None or Path(filename).suffix.lower() in {".bz2", ".xz", ".zst", ".zstd"}:
        return _read_network_xml(filename, skip_attributes=skip_attributes)

    logger.info("Reading network XML with C++: %s", filename)
    was_enabled = gc.isenabled()
    try:
        # These temporary records have no cycles. Reference counting still runs.
        gc.disable()
        nodes, links, node_attrs, link_attrs, metadata, attributes = native.read(filename, skip_attributes)
        node_columns = ["node_id", "x", "y", "z"]
        link_columns = ["link_id", "from_node", "to_node", "length", "freespeed",
                        "capacity", "permlanes", "modes", "oneway"]
        nodes = pd.DataFrame(nodes, columns=node_columns) if nodes else pd.DataFrame({c: [] for c in node_columns})
        if not nodes.empty and nodes.z.isna().all():
            nodes = nodes.drop(columns="z")
        links = pd.DataFrame(links, columns=link_columns) if links else pd.DataFrame({c: [] for c in link_columns})
        node_attrs = pd.DataFrame.from_records(node_attrs, columns=["node_id", "name", "value"])
        link_attrs = pd.DataFrame.from_records(link_attrs, columns=["link_id", "name", "value"])
        # Match the existing reader's string IDs and attribute-table coercions.
        nodes["node_id"] = nodes.node_id.astype(str)
        links["link_id"] = links.link_id.astype(str)
        if not link_attrs.empty:
            link_attrs["link_id"] = link_attrs.link_id.astype(str)
            if links.link_id.is_unique and not is_numeric_dtype(link_attrs.value.dtype):
                links["attributes"] = attributes
        return Network(nodes, links, node_attrs, link_attrs, metadata)
    finally:
        if was_enabled:
            gc.enable()


def read_network_cached(filename, context, skip_attributes, only_car_links):
    """Cache the unfiltered network; apply the car filter only to the loaded copy."""
    filename = str(Path(filename).expanduser().resolve())
    source_stat = os.stat(filename)
    root = getattr(context, "working_directory", None) if context is not None else None
    if root is None:
        build_root = Path(tempfile.gettempdir()) / f"matsim-native-{os.getuid()}"
        network = _read_xml(filename, skip_attributes, build_root)
    else:
        root = Path(root).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        identity = (CACHE_VERSION, filename, source_stat.st_mtime_ns,
                    source_stat.st_size, bool(skip_attributes))
        digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        cache = root / f"network_{digest}.pkl"
        with cache.with_suffix(".lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            network = None
            if cache.exists():
                try:
                    with cache.open("rb") as stream:
                        network = pickle.load(stream)
                    logger.info("Loaded network binary cache: %s", cache)
                except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ValueError) as error:
                    logger.warning("Rebuilding unreadable network cache %s: %s", cache, error)
            if network is None:
                network = _read_xml(filename, skip_attributes, root)
                current = os.stat(filename)
                if (current.st_mtime_ns, current.st_size) != (source_stat.st_mtime_ns, source_stat.st_size):
                    raise RuntimeError(f"Network changed while being read: {filename}")
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=root, prefix=cache.name + ".", delete=False) as stream:
                        temporary = stream.name
                        pickle.dump(network, stream, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(temporary, cache)
                finally:
                    if temporary is not None and os.path.exists(temporary):
                        os.unlink(temporary)
                logger.info("Saved network binary cache: %s", cache)
    if only_car_links:
        network.filter_car_links(inplace=True)
    return network
