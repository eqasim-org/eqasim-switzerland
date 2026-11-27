"""Mesoscopic urban queues with explicit junction control and LTM gap propagation."""

from sumo.functions.config import simulation
from sumo.functions.config.mesoscopic import configure_profile, write
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.source import source_digest


def configure(context):
    simulation.configure(context)
    configure_profile(context)


@monitored
def execute(context):

    # Write the shared mesoscopic preset for the configured scenario inputs.

    return write(context)


def validate(context):
    return {
        "inputs": (simulation.validate(context)),
        "helpers": source_digest("sumo.functions.config.mesoscopic"),
    }
