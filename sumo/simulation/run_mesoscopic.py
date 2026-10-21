"""Run the cut scenario with the urban mesoscopic SUMO preset."""

from sumo.config import mesoscopic
from sumo.simulation.tools import configure_regional, run_regional, validate_regional


def configure(context):
    configure_regional(context, mesoscopic)


def execute(context):

    # Generate the local mesoscopic config and run the cached regional scenario.
    return run_regional(context, mesoscopic, "mesoscopic")


def validate(context):
    return validate_regional(context)
