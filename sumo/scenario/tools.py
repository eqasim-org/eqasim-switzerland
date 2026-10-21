"""Reusable native scenario XML readers, trip writers and routed-vehicle writers."""

import copy
import gzip
import math

from lxml import etree

from sumo.scenario.population.tools import scalar_text, xml_stream
from sumo.scenario.population.writer import PopulationWriter


def elements(path, tags):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",), tag=tags):
            yield element
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]


def write_trips(path, types, records, loading_only=False):
    with xml_stream(path) as stream, etree.xmlfile(stream, encoding="utf-8") as writer:
        writer.write_declaration()
        with writer.element("routes"):
            for element in types.values():
                writer.write(copy.deepcopy(element))
            for record in records:
                if loading_only and not record["load"]:
                    continue
                element = etree.Element("trip", attrib=record["attributes"])
                for key, value in record["parameters"].items():
                    text = scalar_text(value)
                    if text is not None:
                        etree.SubElement(element, "param", key=key, value=text)
                writer.write(element)


def filter_loading_trips(source, destination):
    count = 0
    with xml_stream(destination) as stream, etree.xmlfile(stream, encoding="utf-8") as writer:
        writer.write_declaration()
        with writer.element("routes"):
            for element in elements(source, ("vType", "trip")):
                marker = element.find("param[@key='_sumo_loads_network']")
                if element.tag == "vType" or (marker is not None and marker.get("value") == "true"):
                    writer.write(element)
                    count += element.tag == "trip"
    return count


def read_routes(path):
    """Read explicit routes and finite exit times, preserving IDs and metadata."""
    types, vehicles = {}, []
    for element in elements(path, ("vType", "vehicle")):
        if element.tag == "vType":
            types[element.get("id")] = copy.deepcopy(element)
            continue
        route = element.find("route")
        if route is None:
            raise ValueError(f"Vehicle {element.get('id')} needs a single explicit route")
        edges = route.get("edges", "").split()
        exits = [float(value) for value in route.get("exitTimes", "").split()]
        depart = float(element.get("depart"))
        if len(edges) != len(exits) or not edges or not all(math.isfinite(value) for value in [depart, *exits]):
            raise ValueError(f"Missing or invalid edge exitTimes for {element.get('id')}")
        if depart < 0 or any(end < start for start, end in zip([depart, *exits[:-1]], exits)):
            raise ValueError(f"Unfinished/nonmonotone edge exitTimes for {element.get('id')}")
        vehicles.append({"attributes": dict(element.attrib), "edges": edges, "exit_times": exits,
                         "route_attributes": {key: value for key, value in route.attrib.items() if key not in {"edges", "exitTimes"}},
                         "parameters": {param.get("key"): param.get("value") for param in element.findall("param")}})
    return types, vehicles


def write_routes(path, types, vehicles):
    """Write sorted explicit routes with one timing value per normal edge."""
    ids = set()
    with PopulationWriter(path) as writer:
        for element in types.values():
            writer.element(copy.deepcopy(element))
        for vehicle in sorted(vehicles, key=lambda row: (float(row["attributes"]["depart"]), row["attributes"]["id"])):
            vehicle_id = vehicle["attributes"]["id"]
            if vehicle_id in ids:
                raise ValueError(f"Duplicate routed vehicle ID: {vehicle_id}")
            ids.add(vehicle_id)
            writer.vehicle(vehicle["attributes"], vehicle["edges"], vehicle.get("exit_times"),
                           vehicle.get("parameters"), vehicle.get("route_attributes"))


def write_xml(path, root_name, children):
    """Write plain SUMO network definitions for native netconvert rebuilding."""
    with xml_stream(path) as stream, etree.xmlfile(stream, encoding="utf-8") as writer:
        writer.write_declaration()
        with writer.element(root_name):
            for child in children:
                writer.write(child)
