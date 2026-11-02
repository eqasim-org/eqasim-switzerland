"""Prepare and validate a separate QSim cutter source from an exported simulation."""
import argparse
import gzip
import json
from pathlib import Path

from lxml import etree

from matsim.config.tools import change_params, read_config, resolve_output_file, write_config


def prepare(source, target, refresh=False):
    if refresh and (target / "simulation_output/output_plans.xml.gz").exists():
        raise FileExistsError("Routed population already exists; refusing to replace its config")
    target.mkdir(parents=True, exist_ok=refresh)
    simulation = target / "simulation_output"
    simulation.mkdir(exist_ok=refresh)
    for pattern in ("*.yml", "*.csv", "pricingDescription.xml", "network_calibration_files"):
        for path in source.glob(pattern):
            link_input(target / path.name, path)
    inputs = {}
    for parameter, filename in {
        "plans.inputPlansFile": "output_plans.xml",
        "households.inputFile": "output_households.xml",
        "facilities.inputFacilitiesFile": "output_facilities.xml",
        "network.inputNetworkFile": "output_network.xml",
        "vehicles.vehiclesFile": "output_vehicles.xml",
    }.items():
        path = Path(resolve_output_file(source / "simulation_output" / filename))
        inputs[parameter] = str(path)
        if parameter != "plans.inputPlansFile":
            link_input(simulation / path.name, path)
    # Use the QSim schedule, retaining network routes for road buses.
    for parameter, filename, output_name in (
        ("transit.transitScheduleFile", "switzerland_transit_schedule.xml.gz", "output_transitSchedule.xml.gz"),
        ("transit.vehiclesFile", "switzerland_transit_vehicles.xml.gz", "output_transitVehicles.xml.gz"),
    ):
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[parameter] = str(path)
        link_input(simulation / output_name, path)
    events = Path(resolve_output_file(source / "simulation_output/output_events.xml"))
    link_input(simulation / events.name, events)
    mode_parameters = source / "simulation_output/output_mode_parameters.yml"
    if mode_parameters.is_file():
        link_input(simulation / mode_parameters.name, mode_parameters)
    params = {
        **inputs,
        "controller.mobsim": "qsim",
        "transit.usingTransitInMobsim": True,
        "eqasim.useScheduleBasedTransport": False,
        "eqasim.modeParametersPath": str(mode_parameters if mode_parameters.is_file() else source / "calibrated_dmc_parameters.yml"),
        "eqasim.costParametersPath": str(source / "cost_parameters.yml"),
        "ptZones.ptZonesFilePath": str(source / "gtfs_zones.csv"),
        "ptZones.sbbDistancesPath": str(source / "SBB_all_distances.csv"),
        "ptZones.pricingDescriptionPath": str(source / "pricingDescription.xml"),
        "eqasim:alphaCalibration.activate": False,
        "eqasim:calibration.activate": False,
        "eqasim:calibration.runCalibration": False,
        "eqasim:networkCalibration.activate": False,
        "eqasim:networkCalibration.calibrate": "",
    }
    change_params(source / "switzerland_config.xml", params,
                  output_path=target / "routing_config.xml", create=True)
    change_params(target / "routing_config.xml",
                  {"plans.inputPlansFile": str(simulation / "output_plans.xml.gz")},
                  output_path=simulation / "output_config.xml")
    migrate(target)
    check_config(simulation / "output_config.xml")


def link_input(destination, source):
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    destination.symlink_to(source)


def migrate(target):
    # Current Java uses the parent's objective/calibrate selectors; child
    # activation flags in older exported configs are no longer accepted.
    for path in (target / "routing_config.xml", target / "simulation_output/output_config.xml"):
        tree = read_config(path)
        flags = tree.xpath('/config/module[@name="eqasim:networkCalibration"]'
                           '/parameterset/param[@name="activate"]')
        for flag in flags:
            flag.getparent().remove(flag)
        if flags:
            write_config(tree, path)
        print(f"{path}: removed {len(flags)} obsolete child activation flags", flush=True)


def check_config(path):
    tree = etree.parse(str(path))
    def value(module, name):
        return tree.xpath("string(/config/module[@name=$module]/param[@name=$name]/@value)",
                          module=module, name=name)
    assert value("controller", "mobsim") == "qsim"
    assert "bike" in value("routing", "networkModes").split(",")
    assert "bike" in value("qsim", "mainMode").split(",")
    assert value("transit", "usingTransitInMobsim") == "true"
    assert value("eqasim", "useScheduleBasedTransport") == "false"
    assert "bus" not in value("SBBTransit", "deterministicServiceModes").split(",")


def validate(target):
    path = target / "simulation_output/output_plans.xml.gz"
    persons = bikes = 0
    with gzip.open(path, "rb") as stream:
        for _, person in etree.iterparse(stream, events=("end",), tag="person"):
            persons += 1
            for leg in person.findall("plan/leg"):
                if leg.get("mode") == "bike":
                    bikes += 1
                    route = leg.find("route")
                    if route is None or route.get("type") != "links" or not route.get("start_link") or not route.get("end_link"):
                        raise ValueError(f"Unrouted bike leg for person {person.get('id')}")
            person.clear()
            while person.getprevious() is not None:
                del person.getparent()[0]
    if not persons or not bikes:
        raise ValueError(f"Unexpected empty result: {persons} persons, {bikes} bike legs")
    check_config(target / "simulation_output/output_config.xml")
    result = {"persons": persons, "network_routed_bike_legs": bikes}
    (target / "routing_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "refresh", "validate", "migrate"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    if args.action in ("prepare", "refresh"):
        prepare(args.source.resolve(), args.target.resolve(), refresh=args.action == "refresh")
    elif args.action == "migrate":
        migrate(args.target.resolve())
    else:
        validate(args.target.resolve())
