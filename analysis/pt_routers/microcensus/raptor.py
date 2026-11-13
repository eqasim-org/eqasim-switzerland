"""Observed PT trips routed by eqasim's SwissRailRaptor engine."""
from analysis.pt_routers.microcensus import eqasim


def configure(context):
    eqasim.configure(context)
    context.stage('analysis.pt_routers.microcensus.rust')


def execute(context):

    # Keep the experiment sequential to avoid holding both server timetables in memory.

    return eqasim.execute(context, rust=False)


def validate(context):
    return eqasim.validation_token()
