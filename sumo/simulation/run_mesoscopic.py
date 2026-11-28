"""Run the cut scenario with the urban mesoscopic SUMO preset."""

from sumo.functions.config import mesoscopic
from scripts.monitoring import monitored
from sumo.functions.simulation.tools import (
    configure_regional,
    run_regional,
    validate_regional,
)


def configure(context):
    configure_regional(context, mesoscopic)


@monitored
def execute(context):

    # Generate the local mesoscopic config and run the cached regional scenario.

    return run_regional(context, mesoscopic, "mesoscopic")


def validate(context):
    return validate_regional(context, mesoscopic)
