"""Bike travel time and distance."""
from mode_choice.routing_variables import active
from scripts.monitoring import monitored


def configure(context):
    active.configure(context, "bike")


@monitored
def execute(context):
    return active.execute(context, "bike")
