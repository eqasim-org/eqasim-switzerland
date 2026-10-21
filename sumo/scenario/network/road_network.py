"""Reuse a prepared SUMO network or build one from a single OSM extract."""

import logging
from pathlib import Path
import tempfile
from sumo.config.tools import existing_file
from data.osm.conversion_functions import convert_pbf_to_osm
from sumo.runtime import sumo
from sumo.scenario.network.tools import detect_network_format, resolve_input_path

logger = logging.getLogger("synpp")
DEFAULT_NETCONVERT_OPTIONS = [
    "--geometry.remove", "--ramps.guess", "--junctions.join",
    "--tls.guess-signals", "--tls.discard-simple", "--tls.join",
    "--tls.default-type", "actuated",
    "--keep-edges.by-vclass", "passenger,taxi,delivery,bus,truck,motorcycle,moped",
    "--keep-edges.components", "1",
    "--remove-edges.isolated",
    "--osm.turn-lanes",
    "--osm.lane-access",
    "--proj","EPSG:2056",
    "--ignore-errors"
]


def configure(context):
    context.config("sumo_network_file", None)
    if context.config("sumo_network_file") is None:
        context.config("sumo_osm_file", None)
        if context.config("sumo_osm_file") is None:
            context.config("latest_osm_for_correction", "switzerland-261004.osm.pbf")
    filename = _configured_filename(context)
    if not isinstance(filename, (str, Path)) or not str(filename).strip():
        raise ValueError("SUMO network input must identify a single OSM file or SUMO network")
    if not Path(filename).expanduser().is_absolute():
        context.config("data_path")
    if detect_network_format(_input_path(context)) == "sumo":
        return
    context.stage("sumo.runtime.sumo")
    context.config("random_seed")
    context.config("sumo_netconvert_options", DEFAULT_NETCONVERT_OPTIONS)


def execute(context):

    # Reuse a supplied network without converting inputs or installing SUMO.

    source = _input_path(context)
    source_format = detect_network_format(source)
    if source_format == "sumo":
        return existing_file(source)

    # Resolve one extract independently of the MATSim osm_file list.

    options = context.config("sumo_netconvert_options")
    if not isinstance(options, (list, tuple)) or not all(isinstance(value, str) for value in options):
        raise ValueError("sumo_netconvert_options must be a list of command-line strings")
    cache = Path(context.path()).resolve()
    network = cache / "sumo_road_network.net.xml.gz"

    # Convert PBF to temporary XML without modifying the source extract.

    with tempfile.TemporaryDirectory(prefix="osm_", dir=cache) as temporary:
        osm_input = source
        if source_format == "osm_pbf":
            osm_input = Path(temporary) / "input.osm.xml.gz"
            pbf_input = source
            if source.suffix.lower() != ".pbf":
                pbf_input = Path(temporary) / "input.osm.pbf"
                pbf_input.symlink_to(source)
            convert_pbf_to_osm(pbf_input, osm_input)

        # Build the road network and retain the converter's log in this cache.

        sumo.run(context, [
            *options,
            "--seed", str(context.config("random_seed")),
            "--osm-files", str(osm_input),
            "--output-file", str(network),
            "--output.original-names", "true",
            "--log", str(cache / "netconvert.log"),
        ], binary="netconvert")

    # Return the complete output path for downstream stages.

    if not network.is_file() or network.stat().st_size == 0:
        raise RuntimeError(f"netconvert did not produce a network: {network}")
    return str(network)


def validate(context):
    """Invalidate the cache when the selected prepared network or OSM changes."""
    source = _input_path(context)
    stat = source.stat()
    return [str(source), stat.st_size, stat.st_mtime_ns]


def _configured_filename(context):
    filename = context.config("sumo_network_file")
    if filename is None:
        filename = context.config("sumo_osm_file")
        if filename is None:
            filename = context.config("latest_osm_for_correction")
    return filename


def _input_path(context):
    filename = _configured_filename(context)
    data_path = None if Path(filename).expanduser().is_absolute() else context.config("data_path")
    return resolve_input_path(filename, data_path)


def _osm_path(context):
    """Compatibility name for the unified network input resolver."""
    return _input_path(context)

