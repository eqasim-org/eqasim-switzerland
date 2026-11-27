"""Reuse a prepared SUMO network or build one from a single OSM extract."""

from pathlib import Path

from sumo.functions.scenario.network.tools import resolve_input_path


def _configured_filename(context):
    filename = context.config("sumo_network_file")
    if filename is None:
        filename = context.config("sumo_osm_file")
        if filename is None:
            filename = context.config("latest_osm_for_correction")
            if filename is None:
                filename = "switzerland-261004.osm.pbf"
    return filename


def _input_path(context):
    filename = _configured_filename(context)
    data_path = (
        None
        if Path(filename).expanduser().is_absolute()
        else context.config("data_path")
    )
    return resolve_input_path(filename, data_path)


def _osm_path(context):
    """Compatibility name for the unified network input resolver."""
    return _input_path(context)
