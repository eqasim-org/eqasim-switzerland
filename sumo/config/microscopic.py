"""Urban microscopic simulation preset; local behavioral calibration is separate."""

from pathlib import Path

from sumo.config import simulation, tools


def configure(context):
    simulation.configure(context)
    configure_profile(context)


def execute(context):

    # Prepare scenario inputs and shared congestion and diagnostic outputs.

    return write(context)


def configure_profile(context):
    context.config("sumo_micro_step_length", 0.5)
    context.config("sumo_micro_action_step_length", 1.0)
    context.config("sumo_micro_options", None)


def write(context, network=None, routes=None):
    sections = simulation.sections(context, network=network, routes=routes)
    step = tools.positive(context.config("sumo_micro_step_length"), "sumo_micro_step_length")
    action = tools.positive(context.config("sumo_micro_action_step_length"), "sumo_micro_action_step_length")
    if step > 1 or action < step or abs(action / step - round(action / step)) > 1e-8:
        raise ValueError("Use step <= 1 s and an action interval that is a multiple of the step")

    # Use the established Krauss model with subsecond integration and explicit diagnostics.

    sections["time"]["step-length"] = step
    sections["processing"].update({"default.carfollowmodel": "Krauss", "default.action-step-length": action,
                                   "step-method.ballistic": True, "collision.action": "warn",
                                   "collision.check-junctions": True})
    return tools.write_config(Path(context.path()) / "microscopic.sumocfg", "sumoConfiguration", sections,
                              context.config("sumo_micro_options"))


def validate(context):
    return simulation.validate(context)
