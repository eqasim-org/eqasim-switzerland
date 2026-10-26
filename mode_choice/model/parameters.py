"""Compatibility entry point; parameter I/O is owned by dmc.parameters."""
from dmc.parameters import Parameters


def load(context):
    return Parameters.from_context(context)
