"""Assemble an existing cut without running Synpp, Java, routing or simulations.

Usage: uv run -m scripts.assemble_cut_scenario config_cutter.yml
"""
import argparse
import logging
from pathlib import Path
import tempfile

import yaml
from synpp.pipeline import flatten

from matsim.config.tools import read_config, resolve_output_file
from matsim.cutter import scenario


class ExistingCutContext:
    """Expose scenario's context contract with one immutable, precomputed cut."""

    def __init__(self, settings, cut, directory):
        self.settings = flatten(settings)
        self.cut = cut
        self.directory = directory
        self.working_directory = str(directory)

    def config(self, name, *args, **kwargs):
        if name not in self.settings:
            if args:
                self.settings[name] = args[0]
            elif "default" in kwargs:
                self.settings[name] = kwargs["default"]
            else:
                raise KeyError(name)
        return self.settings[name]

    def stage(self, name):
        if name != "matsim.cutter.runCutter":
            raise RuntimeError(f"Refusing to execute any stage: {name}")
        return str(self.cut)

    def path(self):
        return str(self.directory)


def assemble(config, cut_output=None):
    settings = yaml.safe_load(config.read_text())
    options = flatten(settings["config"])
    destination = Path(options["cutter.output_path"]).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing scenario: {destination}")
    if not options.get("cutter.scenario_path"):
        raise ValueError("An existing cutter.scenario_path is required")
    prefix = options["extent_prefix"]
    if cut_output is None:
        candidates = [path for path in Path(settings["working_directory"]).glob(
            "matsim.cutter.runCutter*.cache/output") if (path / f"{prefix}config.xml").is_file()]
        if len(candidates) != 1:
            raise ValueError(f"Expected one existing cut, found {candidates}; specify --cut-output")
        cut_output = candidates[0]
    cut_output = cut_output.resolve()
    for suffix in ("population.xml", "households.xml", "facilities.xml", "network.xml",
                   "transit_schedule.xml", "transit_vehicles.xml", "vehicles.xml"):
        resolve_output_file(cut_output / (prefix + suffix))
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".assemble_{prefix}", dir=destination.parent))
    context = ExistingCutContext(settings["config"], cut_output, work)
    scenario.configure(context)
    output = Path(scenario.execute(context))
    validate_inputs(output, prefix)
    output.rename(destination)
    print(f"Prepared scenario: {destination}", flush=True)
    return destination


def validate_inputs(directory, prefix):
    tree = read_config(directory / f"{prefix}config.xml")
    for group, name in (
        ("plans", "inputPlansFile"), ("network", "inputNetworkFile"),
        ("households", "inputFile"), ("facilities", "inputFacilitiesFile"),
        ("vehicles", "vehiclesFile"), ("transit", "transitScheduleFile"),
        ("transit", "vehiclesFile"), ("eqasim", "modeParametersPath"),
        ("eqasim", "costParametersPath"), ("ptZones", "ptZonesFilePath"),
        ("ptZones", "sbbDistancesPath"), ("ptZones", "pricingDescriptionPath"),
    ):
        value = tree.xpath("string(/config/module[@name=$group]/param[@name=$name]/@value)",
                           group=group, name=name)
        if not value or Path(value).is_absolute() or not (directory / value).is_file():
            raise ValueError(f"Missing or nonportable input {group}.{name}: {value!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--cut-output", type=Path)
    args = parser.parse_args()
    assemble(args.config, args.cut_output)
