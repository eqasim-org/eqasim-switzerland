"""Runs all the PT analyses of a simulation output in one stage:
- the TL comparisons (Lausanne): passenger counts by stop and line map
- the TPG comparisons (Geneva): passenger counts by stop and line map
- the cross-border flow maps by car and by PT (Geneva and Lausanne extents)

The counts of the operators are the processed ones already in the processed folders, unless analysis.pt.tl_rebuild_counts /
analysis.pt.tpg_rebuild_counts is true: analysis.pt.TL_comparison.tl_2024 / analysis.pt.TPG_comparison.tpg_raw_stats_2025
(gigabytes of raw counts) then run first. The TPG line map needs
the counts per direction of tpg_raw_stats_2025 (tpg<year>_agg_workday_<date>_by_direction.csv): it is left out, with a
warning, as long as they do not exist."""
import glob
import logging
import os

logger = logging.getLogger("synpp")

STAGES = [
    "analysis.pt.TL_comparison.comparison_passenger_counts_lausanne",
    "analysis.pt.TL_comparison.line_comparison",
    "analysis.pt.TPG_comparison.comparison_passenger_counts_geneva",
    "analysis.pt.TPG_comparison.line_comparison",
    "analysis.counts.cross_border_flow_cars",
    "analysis.counts.cross_border_flow_pt",
]


def configure(context):
    context.config("data_path")
    context.config("analysis.pt.tpg_processed_counts_path", default = "Geneva_passenger_counts/TPG/processed")
    context.config("analysis.pt.tpg_year", default = 2025)
    context.config("analysis.pt.tpg_rebuild_counts", default = False)

    if not _tpg_counts_have_directions(context):
        logger.warning("No TPG counts per direction: the TPG line map is left out (run analysis.pt.TPG_comparison.tpg_raw_stats_2025 once)")
    for stage in _stages(context):
        context.stage(stage)

def execute(context):

    # Collect the reports produced by the configured comparison stages.

    paths = {stage: context.stage(stage)["path"] for stage in _stages(context)}
    for stage, path in paths.items():
        logger.info("%s -> %s", stage, path)

    return paths


def _tpg_counts_have_directions(context):
    if context.config("analysis.pt.tpg_rebuild_counts"):  # written by tpg_raw_stats_2025 before the comparisons
        return True
    folder = os.path.join(context.config("data_path"), context.config("analysis.pt.tpg_processed_counts_path"))
    year = context.config("analysis.pt.tpg_year")
    return year == 2024 or len(glob.glob(os.path.join(folder, f"tpg{year}_agg_workday_*_by_direction.csv"))) > 0


def _stages(context):
    if _tpg_counts_have_directions(context):
        return STAGES
    return [stage for stage in STAGES if stage != "analysis.pt.TPG_comparison.line_comparison"]




