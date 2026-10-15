"""Bike travel time and distance."""
from mode_choice.routing_variables import active


def configure(context):
    active.configure(context, "bike")


def execute(context):
    return active.execute(context, "bike")
