"""Network format detection, input resolution and SUMO network readers."""

from contextlib import nullcontext
import gzip
from pathlib import Path

from lxml import etree
import numpy as np
from shapely.geometry import LineString

from sumo.scenario.tools import elements


def resolve_input_path(filename, data_path=None):
    """Use an existing supplied path before looking under data_path/osm."""
    if not isinstance(filename, (str, Path)) or not str(filename).strip():
        raise ValueError("SUMO network input must identify a single OSM file or SUMO network")
    path = Path(filename).expanduser()
    if not path.is_file() and data_path is not None:
        path = Path(data_path).expanduser() / "osm" / path
    if not path.is_file():
        raise FileNotFoundError(f"SUMO network input not found: {path}")
    return path.resolve()


def detect_network_format(filename):
    """Return 'sumo', 'osm' or 'osm_pbf' from at most 64 KiB of the header.

    Detect gzip by magic bytes, OSM PBF by its first BlobHeader, and XML by its
    root element. Filenames and large comments cannot silently change the type.
    """
    path = Path(filename).expanduser()
    limit = 65536
    with path.open("rb") as raw:
        compressed = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        with gzip.GzipFile(fileobj=raw) if compressed else nullcontext(raw) as stream:
            prefix = stream.read(4)
            size = int.from_bytes(prefix, "big")
            if len(prefix) == 4 and 0 < size <= limit - 4:
                header = stream.read(size)
                if len(header) == size and b"\x0a\x09OSMHeader" in header:
                    return "osm_pbf"
                prefix += header
            parser = etree.XMLPullParser(events=("start",), resolve_entities=False, no_network=True)
            consumed = len(prefix)
            try:
                while prefix:
                    parser.feed(prefix)
                    for _, element in parser.read_events():
                        root = etree.QName(element).localname
                        if root in {"net", "osm"}:
                            return "sumo" if root == "net" else "osm"
                        raise ValueError(f"Unsupported network XML root <{root}> in {path}; expected <net> or <osm>")
                    if consumed >= limit:
                        break
                    prefix = stream.read(min(4096, limit - consumed))
                    consumed += len(prefix)
            except etree.XMLSyntaxError as error:
                raise ValueError(f"Unrecognized network format in {path}; expected SUMO XML, OSM XML or OSM PBF") from error
    raise ValueError(f"No SUMO/OSM root or PBF header found within the first {limit} bytes of {path}")


def coordinates(value):
    return [tuple(float(component) for component in point.split(",")) for point in value.split()]


def shape_text(shape):
    return " ".join(",".join(format(value, ".12g") for value in point) for point in shape.coords)


def lane_allows(lane, vehicle_class):
    allow = lane.get("allow")
    disallow = lane.get("disallow", "").split()
    return (allow is None or "all" in allow.split() or vehicle_class in allow.split()) and "all" not in disallow and vehicle_class not in disallow


def read_network(path, context=None):
    if context is not None:
        from sumo.scenario.network.cache import read_cached
        return read_cached(path, context, _read_network_xml)
    return _read_network_xml(path)


def _read_network_xml(path):
    edges, connections, location = {}, [], {}
    for element in elements(path, ("location", "edge", "connection")):
        if element.tag == "location":
            location = dict(element.attrib)
        elif element.tag == "edge":
            if element.get("function") == "internal":
                continue
            lanes = [dict(lane.attrib) for lane in element.findall("lane")]
            if lanes:
                edges[element.get("id")] = {
                    "from": element.get("from"), "to": element.get("to"), "lanes": lanes,
                    "length": float(lanes[0]["length"]), "speed": max(float(lane["speed"]) for lane in lanes),
                    "shape": LineString(coordinates(lanes[0]["shape"])),
                }
        elif not element.get("from", "").startswith(":") and not element.get("to", "").startswith(":"):
            connections.append(dict(element.attrib))
    return {"edges": edges, "connections": connections, "location": location}


def validate_routes(vehicles, network, types=None, state=None):
    state = {} if state is None else state
    if "transitions" not in state:
        state["transitions"] = {(connection["from"], connection["to"]) for connection in network["connections"]}
    transitions = state["transitions"]
    class_transitions = state.setdefault("classes", {})
    for vehicle in vehicles:
        edges = vehicle["edges"]
        if any(edge not in network["edges"] for edge in edges):
            raise ValueError(f"Route {vehicle['attributes']['id']} references absent edges")
        if any(pair not in transitions for pair in zip(edges, edges[1:])):
            raise ValueError(f"Route {vehicle['attributes']['id']} has a forbidden/disconnected transition")
        if types is not None:
            definition = types.get(vehicle["attributes"].get("type"))
            vehicle_class = "passenger" if definition is None else definition.get("vClass", "passenger")
            if any(not any(lane_allows(lane, vehicle_class) for lane in network["edges"][edge]["lanes"]) for edge in edges):
                raise ValueError(f"Route {vehicle['attributes']['id']} uses an edge forbidden to {vehicle_class}")
            if vehicle_class not in class_transitions:
                class_transitions[vehicle_class] = {
                    (connection["from"], connection["to"]) for connection in network["connections"]
                    if lane_allows(network["edges"][connection["from"]]["lanes"][int(connection["fromLane"])], vehicle_class)
                    and lane_allows(network["edges"][connection["to"]]["lanes"][int(connection["toLane"])], vehicle_class)
                    and lane_allows(connection, vehicle_class)}
            if any(pair not in class_transitions[vehicle_class] for pair in zip(edges, edges[1:])):
                raise ValueError(f"Route {vehicle['attributes']['id']} has a turn forbidden to {vehicle_class}")


def position(value, length):
    if value in {"max", "last"}:
        return length
    if value == "base":
        return 0
    number = float(value)
    if number < 0:
        number += length
    if not -1e-6 <= number <= length + 1e-6:
        raise ValueError(f"Vehicle position {value} is outside an edge of length {length}")
    return min(length, max(0, number))


def route_length(edges, network, attributes=None):
    length = float(np.sum([network["edges"][edge]["length"] for edge in edges]))
    if attributes is not None:
        length -= position(attributes.get("departPos", "0"), network["edges"][edges[0]]["length"])
        last = network["edges"][edges[-1]]["length"]
        length -= last - position(attributes.get("arrivalPos", "max"), last)
    return length
