"""Urban microscopic simulation preset; local behavioral calibration is separate."""

from sumo.functions.config import simulation
from sumo.functions.config.microscopic import configure_profile, write
from scripts.monitoring import monitored
from sumo.functions.source import source_digest


def configure(context):
    simulation.configure(context)
    configure_profile(context)


@monitored
def execute(context):

    # Prepare scenario inputs and shared congestion and diagnostic outputs.

    return write(context)


def validate(context):
    return {
        "inputs": (simulation.validate(context)),
        "helpers": source_digest("sumo.functions.config.microscopic"),
    }
