import logging
import os

from .builder import build_dashboard

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("analysis.mode_shares.run")
    context.stage("analysis.counts.run")
    context.stage("analysis.travel_times.run")
    context.stage("data.spatial.cantons")
    context.stage("data.spatial.swiss_border")
    context.stage("analysis.counts.matching.network")

    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default="simulation_output")


def execute(context):
    simulation_root = os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory"),
    )

    os.makedirs(simulation_root, exist_ok=True)
    dashboard_path = build_dashboard(simulation_root)
    logger.info("Dashboard generated at %s", dashboard_path)

    return {"done": True, "path": dashboard_path, "directory": os.path.dirname(dashboard_path)}
