"""Run the six-way activity-zone comparison with normal Synpp or read-only inputs."""

import argparse
import logging
import os
from pathlib import Path
import pickle

import pandas as pd
import yaml

from data.microcensus import persons, activity_chains, activity_chains_clusters
from analysis.microcensus.activity_chains_clusters import write_report
from . import activity_chain_zones_experiment as experiment


REBUILT_STAGES = {
    "data.microcensus.persons": persons,
    "data.microcensus.activity_chains": activity_chains,
    "data.microcensus.activity_chains_clusters": activity_chains_clusters,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument(
        "--cached-inputs",
        action="store_true",
        help="Reuse unique trusted upstream caches; rebuild only diary audit and zones in memory",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if not args.cached_inputs:
        from synpp import Synpp

        pipeline = Synpp.build_from_yml(
            args.config,
            run=[
                "analysis.microcensus.activity_chains_clusters",
                "analysis.matching.activity_chain_zones_experiment",
            ],
        )
        print(pipeline.run_pipeline()[-1], flush=True)
        return

    # Rebuild the prepared audit through its actual producers, preserving old caches.

    context = CachedContext(args.config)
    for module in [persons, activity_chains, activity_chains_clusters, experiment]:
        module.configure(context)
    context.configuring = False
    baseline = pd.read_pickle(context.cache_file("data.microcensus.persons"))
    current = context.stage("data.microcensus.persons")
    pd.testing.assert_frame_equal(current[baseline.columns], baseline)
    print(
        "Prepared person attributes unchanged; reported_trip_count added.", flush=True
    )

    # Publish the default regions and run paired production matchers on cached rosters.

    zones = context.stage("data.microcensus.activity_chains_clusters")
    output = Path(context.config("analysis_path")) / "activity_chains_clusters"
    print(write_report(zones, output), flush=True)
    print(experiment.execute(context), flush=True)


class CachedContext:
    """Read trusted local caches without modifying upstream files or stage results.

    Ambiguous cache identities are rejected. Normal Synpp remains the supported
    production route; this adapter is limited to the requested analysis on an
    existing population, anchors and survey trips.
    """

    def __init__(self, config):
        settings = yaml.safe_load(Path(config).read_text())
        self.values = settings["config"].copy()
        self.cache = Path(settings["working_directory"])
        self.configuring = True
        self.declared = set()
        self.dependencies = set()
        self.aliases = {}
        self.loaded = {}
        available = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count() or 1
        )
        self.values["threads"] = min(
            int(self.values.get("threads", available)), available
        )

    def config(self, name, default=None, volatile=False):
        if self.configuring:
            self.declared.add(name)
            self.values.setdefault(name, default)
        elif name not in self.declared:
            raise ValueError(f"Undeclared configuration: {name}")
        return self.values[name]

    def stage(self, name, alias=None):
        if self.configuring:
            self.dependencies.add(name)
            if alias is not None:
                self.aliases[alias] = name
            return None
        name = self.aliases.get(name, name)
        if name not in self.dependencies:
            raise ValueError(f"Undeclared dependency: {name}")
        if name not in self.loaded:
            if name in REBUILT_STAGES:
                print(f"PREPARE {name}", flush=True)
                self.loaded[name] = REBUILT_STAGES[name].execute(self)
            else:
                with self.cache_file(name).open("rb") as stream:
                    self.loaded[name] = pickle.load(stream)
        return self.loaded[name]

    def cache_file(self, name):
        matches = list(self.cache.glob(name + "__*.p"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one trusted cache for {name}, found {len(matches)}; use normal Synpp to resolve dependencies"
            )
        return matches[0]

    def path(self, name=None):
        if name is None:
            return str(Path(self.values["analysis_path"]) / "matching")
        return str(self.cache_file(self.aliases.get(name, name)).with_suffix(".cache"))

    def progress(self, iterable, **kwargs):
        return iterable


if __name__ == "__main__":
    main()
