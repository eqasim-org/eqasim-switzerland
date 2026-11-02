"""Portable SUMO exports without editing producer caches or recorded results."""

import gzip
from pathlib import Path
import re
import shutil

from lxml import etree

OUTPUT_TAGS = {"edgeData", "laneData", "inductionLoop", "instantInductionLoop", "laneAreaDetector",
               "entryExitDetector", "e1Detector", "e2Detector", "e3Detector", "routeProbe"}


def copy_compressed_xml(source, target):
    with open(source, "rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    if compressed:
        shutil.copy2(source, target)
    else:
        with open(source, "rb") as reader, gzip.open(target, "wb", compresslevel=1) as writer:
            shutil.copyfileobj(reader, writer, length=4*1024*1024)


def output_paths(tree, directory):
    for element in tree.findall("output/*"):
        if element.tag in {"tripinfo-output", "summary-output", "vehroute-output"}:
            element.set("value", directory + "/" + Path(element.get("value")).name)
    for element in tree.findall("report/*"):
        if element.tag in {"log", "error-log"}:
            element.set("value", directory + "/" + Path(element.get("value")).name)


def copy_additional(source, destination, prefix, seen=None):
    source = Path(source).resolve()
    seen = {} if seen is None else seen
    if source in seen:
        return seen[source]
    target = destination / "additional" / (prefix + "_" + source.name)
    relative = target.relative_to(destination).as_posix()
    seen[source] = relative
    if not source.name.endswith((".xml", ".xml.gz")):
        shutil.copy2(source, target)
        return relative
    tree = etree.parse(str(source))
    for index, element in enumerate(tree.iter()):
        for key in ("file", "imgFile"):
            value = element.get(key)
            if not value:
                continue
            if key == "file" and element.tag in OUTPUT_TAGS:
                element.set(key, "../replay/" + prefix + "_" + Path(value).name)
                continue
            dependency = Path(value)
            if not dependency.is_absolute():
                dependency = source.parent / dependency
            if dependency.is_file():
                copied = copy_additional(dependency, destination, prefix + "_" + str(index), seen)
                element.set(key, Path(copied).name)
            elif key == "file":
                element.set(key, "../replay/" + prefix + "_" + Path(value).name)
    opener = gzip.open if target.suffix == ".gz" else open
    with opener(target, "wb") as stream:
        tree.write(stream, encoding="utf-8", xml_declaration=True)
    return relative


def cartesian_copy(source, target):
    with gzip.open(source, "rb") as reader, gzip.open(target, "wb", compresslevel=1) as writer:
        head = reader.read(65536)
        # Change only location metadata, never any road/lane/junction coordinates.
        head = re.sub(rb'(<location\b[^>]*\bprojParameter=")[^"]*(")', rb'\g<1>!\2', head, count=1)
        writer.write(head)
        shutil.copyfileobj(reader, writer, length=4*1024*1024)
