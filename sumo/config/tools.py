"""Shared SUMO configuration input resolution, XML writing and relocation."""

import math
from pathlib import Path

from lxml import etree


def configure_network(context):
    context.stage("sumo.scenario.network.road_network")


def network_path(context):
    return existing_file(context.stage("sumo.scenario.network.road_network"))


def existing_file(filename):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if "," in str(path):
        raise ValueError(f"SUMO file lists cannot contain commas in paths: {path}")
    return str(path)


def file_list(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("SUMO input files must be a nonempty list of filenames")
    return [existing_file(value) for value in values]


def positive(value, name, integer=False):
    number = float(value)
    if not math.isfinite(number) or number <= 0 or (integer and number != int(number)):
        raise ValueError(f"{name} must be a positive {'integer' if integer else 'number'}")
    return int(number) if integer else number


def write_config(path, root_name, sections, overrides=None):
    sections = {section: dict(options) for section, options in sections.items()}
    for section, options in (overrides or {}).items():
        if not isinstance(options, dict):
            raise ValueError("SUMO option overrides must map section names to option dictionaries")
        options = _native_options(options)
        managed = {"net-file", "route-files", "additional-files", "weight-files", "output-file", "netload-output",
                   "tripinfo-output", "summary-output", "vehroute-output"}
        if managed.intersection(options):
            raise ValueError("Input/output filenames are managed by the pipeline; use its file configuration options")
        sections.setdefault(section, {}).update(options)
    root = etree.Element(root_name)
    for name, options in sections.items():
        section = etree.SubElement(root, name)
        for key, value in options.items():
            if value is not None:
                text = str(value).lower() if isinstance(value, bool) else str(value)
                etree.SubElement(section, key, value=text)
    path = Path(path)
    etree.ElementTree(root).write(str(path), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return str(path.resolve())


def relocate_config(source, destination, local_files=None):
    """Resolve input paths while keeping all outputs in the consuming cache."""
    source, destination = Path(source), Path(destination)
    tree = etree.parse(str(source))
    local_files = local_files or {}
    for element in tree.findall("input/*"):
        if not (element.tag.endswith("file") or element.tag.endswith("files")):
            continue
        paths = []
        for filename in element.get("value").split(","):
            path = Path(filename)
            if not path.is_absolute():
                path = source.parent / path
            paths.append(str(local_files.get(path.resolve(), path.resolve())))
        element.set("value", ",".join(paths))
    tree.write(str(destination), encoding="utf-8", xml_declaration=True, pretty_print=True)
    return str(destination.resolve())


def external_token(context, options):
    token = []
    for option in options:
        filenames = context.config(option)
        if filenames is None:
            continue
        for filename in filenames if isinstance(filenames, (list, tuple)) else [filenames]:
            path = Path(existing_file(filename))
            stat = path.stat()
            token.append([str(path), stat.st_size, stat.st_mtime_ns])
    return token


def _native_options(options, prefix=""):
    """Restore dotted SUMO keys expanded into nested dictionaries by Synpp."""
    result = {}
    for key, value in options.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_native_options(value, name))
        else:
            result[name] = value
    return result
