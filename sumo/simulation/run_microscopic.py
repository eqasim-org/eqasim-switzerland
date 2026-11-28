"""Run the cut scenario with the urban microscopic SUMO preset."""

from sumo.functions.config import microscopic
from scripts.monitoring import monitored
from sumo.functions.simulation.tools import (
    configure_regional,
    run_regional,
    validate_regional,
)


def configure(context):
    configure_regional(context, microscopic)


@monitored
def execute(context):

    # Generate the local microscopic config and run the cached regional scenario.

    return run_regional(context, microscopic, "microscopic")


def validate(context):
    return validate_regional(context, microscopic)
