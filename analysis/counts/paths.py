import logging
import os

logger = logging.getLogger("synpp")


SIMULATION_PATH_CONFIG = "analysis.counts.simulation_path"
DETAILED_NETWORK_PATH_CONFIG = "analysis.counts.detailed_network_path"


def configure_simulation_path(context):
    """Register the count-analysis simulation path and its legacy fallback."""
    context.config(SIMULATION_PATH_CONFIG, default="")
    context.config(DETAILED_NETWORK_PATH_CONFIG, default="")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default="simulation_output")


def get_simulation_path(context):
    """Return the MATSim output directory used by the counts analysis."""
    configured_path = context.config(SIMULATION_PATH_CONFIG)
    if configured_path:
        return os.path.abspath(os.path.expanduser(configured_path))

    return os.path.join(
        context.config("output_path"),
        context.config("output_id"),
        context.config("simulation_directory"),
    )


def get_analysis_output_path(context):
    folder = (
        "compare_counts_weekdays"
        if context.config("only_weekday")
        else "compare_counts_all_days"
    )
    return os.path.join(get_simulation_path(context), folder)


def matches_found(data, city, source="simulation network"):
    """Return false, with a useful log entry, when a regional run has no data."""
    if data is not None and not data.empty:
        return True

    logger.warning("No count links for %s matched the %s; skipping.", city, source)
    return False
