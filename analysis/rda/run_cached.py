"""Run the RDA comparison using frozen cached inputs in a separate workspace."""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
from synpp import Synpp


INPUTS = (
    "data.constants", "synthesis.population.enriched", "data.microcensus.trips",
    "data.microcensus.persons", "synthesis.population.spatial.primary.work.work_remotly",
    "synthesis.population.sampled", "synthesis.population.spatial.home.locations",
    "synthesis.population.spatial.primary.locations",
    "synthesis.population.spatial.secondary.distance_distributions",
    "synthesis.population.spatial.secondary.detour_factors.factors",
    "synthesis.population.destinations", "data.statent.statent", "data.osm.businesses",
)


def snapshot(settings):
    source = Path(settings["working_directory"]).resolve()
    target = source / ("rda_comparison_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    target.mkdir()
    manifest = {}
    for name in INPUTS:
        if name.endswith("detour_factors.factors") and not settings["config"].get("consider_detour_factor_in_sl", True):
            continue
        candidates = [path for path in source.glob(name + "__*.p") if path.with_suffix(".cache").is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No complete input cache for {name}")
        chosen = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
        for path in (chosen, chosen.with_suffix(".cache")):
            (target / path.name).symlink_to(path, target_is_directory=path.is_dir())
        manifest[name] = str(chosen)
    (target / "input_manifest.json").write_text(json.dumps(manifest, indent=2))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--threads", type=int, default=24)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with open(args.config) as stream:
        settings = yaml.safe_load(stream)
    working = snapshot(settings)
    config = {**settings["config"], "threads": args.threads}
    print("Isolated comparison workspace:", working, flush=True)
    definitions = [{"descriptor": name, "config": {}} for name in (
        "synthesis.population.trips", "synthesis.population.destinations_opening_hours", "analysis.rda.run")]
    # Debug mode reruns only the requested stages and connecting RDA variants;
    # all boundary inputs are read-only links to the selected existing caches.
    pipeline = Synpp(config=config, working_directory=str(working), definitions=definitions,
                     debug=True, externals=settings.get("externals", {}), aliases=settings.get("aliases", {}))
    results = pipeline.run_pipeline()
    print("RDA comparison completed:", results[-1], flush=True)


if __name__ == "__main__":
    main()
