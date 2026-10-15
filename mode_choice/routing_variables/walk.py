"""Walk travel time and distance."""
from mode_choice.routing_variables import active


def configure(context):
    active.configure(context, "walk")


def execute(context):
    return active.execute(context, "walk")
