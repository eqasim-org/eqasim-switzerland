"""Transcality peak-period scatter plots and interactive network map."""

from . import transcality
from ..cantons.transcality_peak_hour import configure_peak_hours, get_peak_hours
from ..paths import configure_simulation_path
from ..construction_sites import configure_filter


def configure(context):
    configure_filter(context)
    context.stage("analysis.counts.cantons.transcality_peak_hour")
    context.stage("analysis.counts.matching.compare_peak_hour")
    context.stage("analysis.counts.matching.network")
    context.stage("data.spatial.swiss_border")
    context.config("input_downsampling")
    context.config("only_weekday", default=False)
    configure_simulation_path(context)
    configure_peak_hours(context)


def execute(context):
    start, end = get_peak_hours(context)
    return transcality.execute(
        context, counts_stage="analysis.counts.cantons.transcality_peak_hour",
        comparison_stage="analysis.counts.matching.compare_peak_hour",
        peak_hours=(start, end),
    )
