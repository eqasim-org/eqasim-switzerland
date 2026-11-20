"""Inspect airport passenger demand in a standalone, filterable HTML report."""

from analysis.airports import trips_html


def configure(context):
    trips_html.configure(context)


def execute(context):

    # Check airport trips and render the canton, municipality and individual views.

    return trips_html.execute(context)


def validate(context):
    return trips_html.validate(context)
