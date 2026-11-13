"""Observed PT trips routed by eqasim's Rust engine."""
from analysis.pt_routers.microcensus import eqasim


def configure(context):
    eqasim.configure(context)


def execute(context):

    # Route with the production server's Rust switch enabled.

    return eqasim.execute(context, rust=True)


def validate(context):
    return eqasim.validation_token()
