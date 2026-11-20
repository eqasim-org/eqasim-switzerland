"""Render a saved repeated experiment without new fitting or donor assignment."""
from . import repeat_experiments, repeated_report


def configure(context):
    context.stage('analysis.matching.repeat_experiments')


def execute(context):

    # Summarise the saved population assignments and regenerate all report figures.

    directory = context.stage('analysis.matching.repeat_experiments')
    repeated_report.summarize_population(directory)
    return repeated_report.render(directory)
