"""Shared input and output preparation for microscopic and mesoscopic presets."""

from pathlib import Path

from lxml import etree

from sumo.config import tools


def configure(context, inputs=True):
    if inputs:
        tools.configure_network(context)
        context.config("sumo_route_files", None)
        if context.config("sumo_route_files") is None:
            context.stage("sumo.scenario.population.plans")
            context.stage("sumo.scenario.population.trips")
    context.config("sumo_additional_files", [])
    context.config("random_seed")
    context.config("sumo_begin", 0)
    context.config("sumo_end", 108000)
    context.config("sumo_edge_output_period", 900)
    context.config("sumo_rerouting_probability", 0.0)
    context.config("sumo_rerouting_period", 300)


def sections(context, network=None, routes=None):
    if routes is None:
        routes = context.config("sumo_route_files")
        if routes is None:
            routes = [context.stage("sumo.scenario.population.plans"), context.stage("sumo.scenario.population.trips")]
    routes = tools.file_list(routes)
    begin, end = float(context.config("sumo_begin")), tools.positive(context.config("sumo_end"), "sumo_end")
    if not 0 <= begin < end:
        raise ValueError("Require 0 <= sumo_begin < sumo_end")
    period = tools.positive(context.config("sumo_edge_output_period"), "sumo_edge_output_period")
    probability = float(context.config("sumo_rerouting_probability"))
    if not 0 <= probability <= 1:
        raise ValueError("sumo_rerouting_probability must be between zero and one")
    reroute_period = tools.positive(context.config("sumo_rerouting_period"), "sumo_rerouting_period")
    additional = etree.Element("additional")
    etree.SubElement(additional, "edgeData", id="traffic", file="edge_data.xml.gz", period=str(period),
                     excludeEmpty="false", withInternal="true")
    statistics = Path(context.path()).resolve() / "edge_statistics.add.xml"
    etree.ElementTree(additional).write(str(statistics), encoding="utf-8", xml_declaration=True, pretty_print=True)
    files = [tools.existing_file(filename) for filename in context.config("sumo_additional_files")]
    files.append(str(statistics))
    return {
        "input": {"net-file": tools.network_path(context) if network is None else str(network), "route-files": ",".join(routes),
                  "additional-files": ",".join(files)},
        "time": {"begin": begin, "end": end},
        "processing": {"time-to-teleport": -1, "ignore-route-errors": False},
        "routing": {"routing-algorithm": "astar", "device.rerouting.probability": probability,
                           "device.rerouting.period": reroute_period,
                           "device.rerouting.adaptation-interval": 60,
                           "device.rerouting.adaptation-steps": 5},
        "output": {"tripinfo-output": "tripinfo.xml.gz", "tripinfo-output.write-unfinished": True,
                   "tripinfo-output.write-undeparted": True, "summary-output": "summary.xml.gz",
                   "vehroute-output": "routes.xml.gz"},
        "report": {"no-step-log": True, "duration-log.statistics": True,
                   "log": "simulation.log", "error-log": "warnings.log"},
        "random_number": {"seed": context.config("random_seed")},
    }


def validate(context):
    return tools.external_token(context, ("sumo_route_files", "sumo_additional_files"))
