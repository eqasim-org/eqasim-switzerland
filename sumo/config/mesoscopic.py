"""Mesoscopic urban queues with explicit junction control and LTM gap propagation."""

from pathlib import Path

from sumo.config import simulation, tools


def configure(context):
    simulation.configure(context)
    configure_profile(context)


def execute(context):

    # Write the shared mesoscopic preset for the configured scenario inputs.

    return write(context)


def configure_profile(context):
    context.config("sumo_meso_segment_length", 98)
    context.config("sumo_meso_options", None)


def write(context, network=None, routes=None):

    # Prepare the same scenario inputs and traffic outputs as the microscopic preset.

    sections = simulation.sections(context, network=network, routes=routes)
    sections["time"]["step-length"] = 1

    # Avoid SUMO 1.28's internal-junction lane-queue crash; retain lane permissions.

    sections["processing"]["no-internal-links"] = True

    # Model urban spillback, lane permissions and signal-controlled junctions.

    sections["mesoscopic"] = {
        "mesosim": True, "meso-ltm": True, "meso-junction-control": True,
        "meso-lane-queue": True, "meso-multi-queue": True,
        "meso-edgelength": tools.positive(context.config("sumo_meso_segment_length"), "sumo_meso_segment_length"),
        "meso-tls-penalty": 0, "meso-tls-flow-penalty": 0,
    }
    return tools.write_config(Path(context.path()) / "mesoscopic.sumocfg", "sumoConfiguration", sections,
                              context.config("sumo_meso_options"))


def validate(context):
    return simulation.validate(context)
