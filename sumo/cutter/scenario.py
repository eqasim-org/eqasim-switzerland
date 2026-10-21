"""Package the regional network and routed demand with a reusable city config."""

import json
from pathlib import Path
import shutil

from sumo.config import microscopic, simulation


def configure(context):
    context.stage("sumo.cutter.network")
    context.stage("sumo.cutter.trips")
    simulation.configure(context, inputs=False)
    microscopic.configure_profile(context)


def execute(context):

    # Export copies so both upstream cutter caches remain reusable.

    network = context.stage("sumo.cutter.network")
    trips = context.stage("sumo.cutter.trips")
    cache = Path(context.path()).resolve()
    for source, name in ((network["network"], "network.net.xml.gz"), (network["mapping"], "edge_mapping.json"),
                         (trips["routes"], "routes.rou.xml.gz")):
        shutil.copyfile(source, cache / name)

    # Reuse the shared microscopic generator with regional inputs and loaded departure times.

    config = microscopic.write(context, network=cache / "network.net.xml.gz", routes=[str(cache / "routes.rou.xml.gz")])
    result = {"network": str(cache / "network.net.xml.gz"), "routes": str(cache / "routes.rou.xml.gz"),
              "config": config, "count": trips["count"], "network_report": network, "trip_report": trips}
    (cache / "scenario.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
