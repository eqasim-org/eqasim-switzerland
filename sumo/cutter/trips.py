"""Cut every contiguous routed visit, departing at its loaded boundary-entry time."""

import json
from pathlib import Path

from sumo.cutter.tools import cut_vehicle
from sumo.scenario.network.tools import read_network, validate_routes
from sumo.scenario.tools import read_routes, write_routes


def configure(context):
    context.stage("sumo.simulation.load_network")
    context.stage("sumo.cutter.network")


def execute(context):

    # Load explicit routes and the directed mapping onto the clipped network.

    loaded = context.stage("sumo.simulation.load_network")
    region = context.stage("sumo.cutter.network")
    source_network, cut_network = map(read_network, (loaded["network"], region["network"]))
    mapping = json.loads(Path(region["mapping"]).read_text())
    types, vehicles = read_routes(loaded["routes"])

    # Keep each inside segment, without bridging an outside gap or changing its path.

    cut, outside, repeated = [], 0, 0
    transitions = {(connection["from"], connection["to"]) for connection in cut_network["connections"]}
    for vehicle in vehicles:
        visits = cut_vehicle(vehicle, mapping, source_network, cut_network, transitions)
        outside += not visits
        repeated += len(visits) > 1
        cut.extend(visits)
    validate_routes(cut, cut_network, types)

    # Write departure-sorted routes and report exclusions and split visits.

    cache = Path(context.path()).resolve()
    output = cache / "region.routes.xml.gz"
    write_routes(output, types, cut)
    result = {"routes": str(output), "network": region["network"], "count": len(cut),
              "source_count": len(vehicles), "outside_or_discarded": outside, "repeated_visits": repeated,
              "interpolated_visits": sum(vehicle["parameters"]["sumo_boundary_timing"] == "interpolated" for vehicle in cut)}
    (cache / "cut_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
