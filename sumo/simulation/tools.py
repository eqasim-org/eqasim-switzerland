"""Shared execution of microscopic and mesoscopic regional scenarios."""

import json
from pathlib import Path

from sumo.config import simulation, tools
from sumo.runtime import sumo


def configure_regional(context, profile):
    context.stage("sumo.cutter.scenario")
    context.stage("sumo.runtime.sumo")
    simulation.configure(context, inputs=False)
    profile.configure_profile(context)


def run_regional(context, profile, model):
    """Generate local config/diagnostics and read the cached cut inputs unchanged."""
    scenario = context.stage("sumo.cutter.scenario")
    cache = Path(context.path()).resolve()
    network = tools.existing_file(scenario["network"])
    routes = tools.existing_file(scenario["routes"])
    config = profile.write(context, network=network, routes=[routes])
    sumo.run(context, ["--configuration-file", config])
    result = {"model": model, "config": config, "network": network, "input_routes": routes,
              "count": scenario["count"], "tripinfo": str(cache / "tripinfo.xml.gz"),
              "edge_data": str(cache / "edge_data.xml.gz"), "summary": str(cache / "summary.xml.gz"),
              "routes": str(cache / "routes.xml.gz")}
    for name in ("tripinfo", "edge_data", "summary", "routes"):
        if not Path(result[name]).is_file():
            raise RuntimeError(f"SUMO did not produce its configured {name} output: {result[name]}")
    (cache / "simulation_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def validate_regional(context):
    return tools.external_token(context, ("sumo_additional_files",))
