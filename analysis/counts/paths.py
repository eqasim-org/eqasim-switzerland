import logging
import os

logger = logging.getLogger("synpp")


def configure_simulation_path(context):
    """Register the count-analysis simulation path and its legacy fallback."""
    context.config("analysis.counts.simulation_path", default="")
    context.config("analysis.counts.detailed_network_path", default="")
    context.config("output_path")
    context.config("output_id")
    context.config("simulation_directory", default="simulation_output")


def get_simulation_path(context):
    """Return the MATSim output directory used by the counts analysis."""
    configured_path = context.config("analysis.counts.simulation_path")
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


# Areas of the cross-border flow maps (one map per area): name -> shapefile/geopackage, absolute or relative to data_path
DEFAULT_FLOW_MAP_EXTENTS = {
    "geneva":   "spatial/MMT/MMT_extent_cleaned.gpkg",
    "lausanne": "spatial/LausanneExtent/LausanneExtent.shp",
}


def configure_flow_map_extents(context):
    context.config("data_path")
    context.config("cross_border_flow_extents", default = DEFAULT_FLOW_MAP_EXTENTS)


def get_flow_map_extents(context):
    """{name: path of the extent}, the paths made absolute"""
    return {
        name: path if os.path.isabs(path) else os.path.join(context.config("data_path"), path)
        for name, path in context.config("cross_border_flow_extents").items()
    }
