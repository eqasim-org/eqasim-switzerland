"""Capture and compare network conversions, with focus on the investigated A1 ways."""
import argparse
import json
from pathlib import Path
import pickle
import re
import subprocess

from lxml import etree
import xopen
import yaml

STAGES = ("matsim.runtime.pt2matsim", "matsim.scenario.network.convert_osm_pt2matsim",
          "matsim.scenario.network.convert_osm")
WAYS = {"61908233", "61908234"}


def stage_results(config, latest=False):
    settings = yaml.safe_load(Path(config).read_text())
    cache = Path(settings["working_directory"])
    metadata = json.loads((cache / "pipeline.json").read_text())
    if latest:
        keys = {name: max((k for k in metadata if k.startswith(name+"__")), key=lambda k: metadata[k]["updated"]) for name in STAGES}
    else:
        from synpp.pipeline import process_stages
        registry = process_stages([{"descriptor": STAGES[-1], "config": {}}], settings["config"])
        keys = {v["descriptor"]: k for k, v in registry.items() if v["descriptor"] in STAGES}
    results = {}
    for name, key in keys.items():
        with (cache / (key+".p")).open("rb") as stream:
            value = pickle.load(stream)
        results[name] = {"key": key, "result": value}
    jar = Path(results[STAGES[0]]["result"])
    repository = jar.parent.parent
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    return {"commit": revision, "stages": results}


def inspect_network(path):
    count, matches = 0, []
    with xopen.xopen(str(path), "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",), tag=("node", "link"),
                                         load_dtd=False, no_network=True, resolve_entities=False):
            if element.tag == "link":
                count += 1
                attrs = {a.get("name"): "".join(a.itertext()) for a in element.findall("attributes/attribute")}
                ways = set(re.split(r"[_,;\s]+", attrs.get("osm:way:id", ""))) | set(re.split(r"[_,;\s]+", attrs.get("old_osm_id", "")))
                if ways & WAYS:
                    matches.append({**dict(element.attrib), "attributes": attrs, "investigated_ways": sorted(ways & WAYS)})
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    stat = Path(path).stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "links": count, "investigated_links": matches}


def compare(config, baseline, output):
    before = json.loads(Path(baseline).read_text())
    after = stage_results(config)
    settings = yaml.safe_load(Path(config).read_text())
    if after["commit"] != settings["config"]["pt2matsim_commit"]:
        raise RuntimeError("Completed runtime does not match the requested commit")
    report = {"baseline": before, "updated": after}
    for label, run in (("baseline", before), ("updated", after)):
        raw = run["stages"][STAGES[1]]["result"][0]
        cleaned = run["stages"][STAGES[2]]["result"]
        for phase, path in (("raw_network", raw), ("cleaned_network", cleaned)):
            previous = run.get(phase, {})
            stat = Path(path).stat()
            if previous.get("path") != str(path) or previous.get("size") != stat.st_size or previous.get("mtime_ns") != stat.st_mtime_ns:
                run[phase] = inspect_network(path)
        stats = Path(cleaned).parent / "statistics_of_cleaning_network.json"
        report[label]["cleaning_statistics"] = json.loads(stats.read_text())
    target_ids = {r["id"] for r in report["updated"]["raw_network"]["investigated_links"]}
    audit = Path(after["stages"][STAGES[2]]["result"]).parent / "cleaning_events.jsonl.gz"
    events = []
    if audit.exists():
        with xopen.xopen(str(audit), "rt") as stream:
            for line in stream:
                event = json.loads(line)
                if target_ids.intersection(event["original_link_ids"]):
                    events.append(event)
    report["updated"]["investigated_cleaning_events"] = events
    for run in (before, after):
        run["a1_car_access"] = {
            way: {phase: [row["id"] for row in run[phase]["investigated_links"]
                          if way in row["investigated_ways"] and "car" in row.get("modes", "").split(",")]
                  for phase in ("raw_network", "cleaned_network")}
            for way in sorted(WAYS)}
    report["changes"] = {name: {"old": before["cleaning_statistics"].get(name), "new": after["cleaning_statistics"].get(name)}
                         for name in ("removed_loops", "successful_merge", "removed_unconnected_links", "removed_truck_mode_from_unconnected_links")}
    Path(output).write_text(json.dumps(report, indent=2))
    print("Compared commits:", before["commit"], "->", after["commit"], flush=True)
    for label in ("baseline", "updated"):
        print(label, "raw", report[label]["raw_network"]["links"], "cleaned", report[label]["cleaned_network"]["links"],
              "A1 raw/cleaned", len(report[label]["raw_network"]["investigated_links"]), len(report[label]["cleaned_network"]["investigated_links"]), flush=True)
    print("Cleaning changes:", report["changes"], flush=True)
    print("Updated A1 car access:", after["a1_car_access"], flush=True)
    print("Detailed report:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config_dib_10.yml")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-baseline", action="store_true")
    args = parser.parse_args()
    if args.capture_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(stage_results(args.config, latest=True), indent=2))
        print("Baseline recorded:", args.baseline)
    else:
        if args.output is None:
            parser.error("--output is required for comparison")
        compare(args.config, args.baseline, args.output)
