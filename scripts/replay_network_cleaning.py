"""Rerun only network cleaning from resolved caches; publish a verified audit."""
import argparse
from contextlib import closing
import hashlib
import json
import logging
from pathlib import Path
import pickle
import sqlite3

import pandas as pd
import yaml

from matsim.scenario.network import convert_osm
from matsim.scenario.network.utils.network_handler import NetworkHandler
from matsim.scenario.network.utils.link_geometry import attach_link_geometries
from matsim.scenario.network.utils.cleaning_audit import original_ids


class CachedContext:
    def __init__(self, config, output, stage_key):
        settings = yaml.safe_load(Path(config).read_text())
        self.cache = Path(settings["working_directory"])
        metadata = json.loads((self.cache / "pipeline.json").read_text())[stage_key]
        self.values = {**metadata["config"], **settings["config"]}
        self.dependencies = {key.split("__")[0]: key for key in metadata["dependencies"]}
        self.output = Path(output)
        self.results = {}
        self.working_directory = str(self.output)

    def config(self, name, *args, **kwargs):
        if name not in self.values:
            self.values[name] = args[0] if args else kwargs.get("default")
        return self.values[name]

    def stage(self, name):
        if name not in self.results:
            with (self.cache / (self.dependencies[name]+".p")).open("rb") as stream:
                self.results[name] = pickle.load(stream)
        return self.results[name]

    def path(self, name=None):
        return str(self.output if name is None else self.cache / (self.dependencies[name]+".cache"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--stage-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "verification.json").unlink(missing_ok=True)
    context = CachedContext(args.config, args.output, args.stage_key)
    convert_osm.configure(context)
    raw, geometry = context.stage("matsim.scenario.network.convert_osm_pt2matsim")
    source = Path(raw).resolve()
    stat = source.stat()
    digest = hashlib.sha256(json.dumps((1, str(source), stat.st_mtime_ns, stat.st_size, False)).encode()).hexdigest()
    binary = context.cache / f"network_{digest}.pkl"
    if not binary.is_file():
        raise FileNotFoundError(f"Expected existing raw-network reader cache: {binary}")
    logging.info("Loading raw network from %s", binary)
    network = pd.read_pickle(binary)
    attach_link_geometries(network, geometry)
    handler = NetworkHandler.__new__(NetworkHandler)
    handler.net, handler.context = network, context
    handler.schedule_path = context.stage("matsim.scenario.network.convert_pt_schedule")["schedule"]
    # Tolls affect chain boundaries. Municipality/elevation/traffic-light attributes
    # do not alter topology or merging criteria; later speed/capacity steps are omitted.
    handler._add_tolls_if_requested()
    handler._simplify_network_if_requested()
    logging.info("Cleaning finished; checking identities against the displayed cleaned network")
    with closing(sqlite3.connect((args.viewer / "network.sqlite").resolve().as_uri()+"?mode=ro&immutable=1", uri=True)) as connection:
        expected = {row[0] for row in connection.execute("SELECT id FROM links WHERE network='after'")}
        actual = set(handler.net.links.link_id.astype(str))
        if expected != actual:
            raise RuntimeError(f"Cleaned link IDs differ: {len(expected-actual)} missing, {len(actual-expected)} new; audit not published")
        expected_merges = {}
        for identifier, member in connection.execute("SELECT a.id,m.before_id FROM links a JOIN members m ON m.after_key=a.k WHERE a.network='after' AND a.status='merged'"):
            expected_merges.setdefault(identifier, set()).add(member)
        actual_merges = {}
        for row in handler.net.links.itertuples():
            ids = original_ids(row.link_id, row.attributes)
            if len(ids) > 1:
                actual_merges[str(row.link_id)] = set(ids)
        if expected_merges != actual_merges:
            raise RuntimeError("Merge provenance differs from displayed network; audit not published")
    verification = {"verified": True, "links": len(actual), "merged_links": len(actual_merges),
                    "raw_network": str(source), "raw_mtime_ns": stat.st_mtime_ns,
                    "compared_database": str((args.viewer / 'network.sqlite').resolve()),
                    "scope": "Exact cleaned link IDs and original-link merge membership; topology cleaning only",
                    "stage_key": args.stage_key}
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
    logging.info("VERIFIED: %s", verification)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
