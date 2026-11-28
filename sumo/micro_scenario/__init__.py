"""Package the regional network and routed demand with a reusable city config."""

import json
from pathlib import Path

from sumo.functions.config import microscopic, simulation
from scripts.monitoring import monitored


def configure(context):
    context.stage("sumo.micro_scenario.network")
    context.stage("sumo.micro_scenario.demand")
    simulation.configure(context, inputs=False)
    microscopic.configure_profile(context)


@monitored
def execute(context):

    # Reference the independently cached regional network and demand.

    network = context.stage("sumo.micro_scenario.network")
    trips = context.stage("sumo.micro_scenario.demand")
    cache = Path(context.path()).resolve()

    # Write the local config; sumo.output creates portable copies at export time.

    config = microscopic.write(
        context, network=network["network"], routes=[trips["routes"]]
    )
    result = {
        "network": network["network"],
        "routes": trips["routes"],
        "config": config,
        "count": trips["count"],
        "network_report": network,
        "trip_report": trips,
    }
    (cache / "scenario.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
