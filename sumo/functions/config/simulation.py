"""Shared input and output preparation for microscopic and mesoscopic presets."""

import math
import tempfile
from pathlib import Path

from lxml import etree

from sumo.functions.config import tools


def configure(context, inputs=True):
    if inputs:
        tools.configure_network(context)
        context.config("sumo_route_files", None)
        if context.config("sumo_route_files") is None:
            context.stage("sumo.macro_scenario.plans")
            context.stage("sumo.macro_scenario.vehicles")
        elif any(
            str(path).endswith(".arrow") for path in context.config("sumo_route_files")
        ):
            from sumo.functions.native import processing

            processing.configure_stage(context, "export")
    context.config("sumo_additional_files", [])
    context.config("random_seed")
    context.config("threads", volatile=True)
    context.config("sumo_begin", 0)
    context.config("sumo_end", None)
    context.config("sumo_edge_output_period", 900)
    context.config("sumo_rerouting_probability", 1.0)
    context.config("sumo_rerouting_period", 60)
    context.config("sumo_time_to_teleport", 600)


def sections(context, network=None, routes=None):
    if routes is None:
        routes = context.config("sumo_route_files")
        if routes is None:
            routes = [
                context.stage("sumo.macro_scenario.plans"),
                context.stage("sumo.macro_scenario.vehicles"),
            ]
    routes = tools.file_list(routes)
    network = tools.network_path(context) if network is None else str(network)
    routes = _route_files(context, routes, network)
    begin, end = tools.time_window(context)
    period = tools.positive(
        context.config("sumo_edge_output_period"), "sumo_edge_output_period"
    )
    probability = float(context.config("sumo_rerouting_probability"))
    if not 0 <= probability <= 1:
        raise ValueError("sumo_rerouting_probability must be between zero and one")
    reroute_period = tools.positive(
        context.config("sumo_rerouting_period"), "sumo_rerouting_period"
    )
    teleport = float(context.config("sumo_time_to_teleport"))
    if not math.isfinite(teleport):
        raise ValueError(
            "sumo_time_to_teleport must be finite; use -1 to disable gridlock recovery"
        )
    additional = etree.Element("additional")
    etree.SubElement(
        additional,
        "edgeData",
        id="traffic",
        file="edge_data.xml.gz",
        period=str(period),
        excludeEmpty="false",
        withInternal="true",
    )
    statistics = Path(context.path()).resolve() / "edge_statistics.add.xml"
    etree.ElementTree(additional).write(
        str(statistics), encoding="utf-8", xml_declaration=True, pretty_print=True
    )
    files = [
        tools.existing_file(filename)
        for filename in context.config("sumo_additional_files")
    ]
    files.append(str(statistics))
    return {
        "input": {
            "net-file": network,
            "route-files": ",".join(routes),
            "additional-files": ",".join(files),
        },
        "time": {"begin": begin, "end": end},
        "processing": {"time-to-teleport": teleport, "ignore-route-errors": False},
        "routing": {
            "routing-algorithm": "astar",
            "device.rerouting.probability": probability,
            "device.rerouting.threads": tools.positive(
                context.config("threads"), "threads", integer=True
            ),
            "device.rerouting.period": reroute_period,
            "device.rerouting.adaptation-interval": 10,
            "device.rerouting.adaptation-steps": 6,
        },
        "output": {
            "tripinfo-output": "tripinfo.xml.gz",
            "tripinfo-output.write-unfinished": True,
            "tripinfo-output.write-undeparted": True,
            "summary-output": "summary.xml.gz",
            "vehroute-output": "routes.xml.gz",
            "statistic-output": "statistics.xml.gz",
        },
        "report": {
            "no-step-log": True,
            "duration-log.statistics": True,
            "log": "simulation.log",
            "error-log": "warnings.log",
        },
        "random_number": {"seed": context.config("random_seed")},
    }


def validate(context):
    token = tools.external_token(context, ("sumo_route_files", "sumo_additional_files"))
    from sumo.functions.scenario.route_io import metadata

    for path in context.config("sumo_route_files") or []:
        if str(path).endswith(".arrow"):
            info = metadata(path)
            for key in ("sumo_edges", "sumo_types"):
                source = Path(info[key])
                stat = source.stat()
                token.append([str(source), stat.st_size, stat.st_mtime_ns])
    return token


def _route_files(context, routes, network):
    """Export IPC only when a standalone config explicitly requests those routes."""
    from sumo.functions.native import processing
    from sumo.functions.scenario.route_io import metadata

    result = []
    for index, source in enumerate(routes):
        if not source.endswith(".arrow"):
            result.append(source)
            continue
        info = metadata(source)
        target = Path(context.path()).resolve() / f"routes_{index}.xml.gz"
        with tempfile.TemporaryDirectory(
            prefix="route_export_", dir=context.path()
        ) as temporary:
            pending = Path(temporary) / target.name
            processing.run(
                context,
                "export",
                routes=source,
                network=network,
                edges=info["sumo_edges"],
                types=info["sumo_types"],
                output=pending,
            )
            pending.replace(target)
        result.append(str(target))
    return result
