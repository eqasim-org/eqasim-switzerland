"""Cache selected, untrimmed SUMO road demand before route completion."""

import copy
import hashlib
import logging
from pathlib import Path

from lxml import etree
from sumo.tables import read_table

from sumo.config import tools as config_tools
from sumo.scenario.tools import elements
from sumo.scenario.demand_tools import TripSpool
from sumo.scenario import modes
from sumo.scenario.population.tools import (
    identifier, seconds,
)
from sumo.scenario.population.vehicle_tools import BATCH_SIZE, population_trip_batches, trip_records

logger = logging.getLogger("synpp")


def configure(context):
    modes.configure(context)
    context.config("sumo_assignment_trip_files", None)
    if context.config("sumo_assignment_trip_files") is None:
        context.stage("sumo.scenario.population.population")
        context.stage("sumo.scenario.population.trips")
    modes.require_mode_choice(context)
    context.config("clip_trips_to_border_in_sumo", True)
    if context.config("clip_trips_to_border_in_sumo") and context.config("sumo_assignment_trip_files") is not None:
        config_tools.configure_network(context)
    context.config("sumo_assignment_all_as_car", False)
    context.config("sumo_population_crs", "EPSG:2056")


def execute(context):

    # Read supplied vehicle trips or construct queries from the synthetic population.

    types = {"sumo_query_car": etree.Element("vType", id="sumo_query_car", vClass="passenger", speedDev="0")}
    cache = Path(context.path()).resolve()
    from sumo.cutter.demand_tools import EndpointResolver
    resolver = (EndpointResolver(lambda: config_tools.network_path(context))
                if context.config("clip_trips_to_border_in_sumo") else None)
    with TripSpool(cache / "demand.arrow", endpoint_resolver=resolver) as records:
        files = context.config("sumo_assignment_trip_files")
        if files is None:
            for batch in _population_batches(context):
                records.add_population(batch, context.config("sumo_assignment_all_as_car"))
            files = [context.stage("sumo.scenario.population.trips")]
        for filename in config_tools.file_list(files):
            _read_vehicle_trips(filename, types, records, modes.selected(context.config("sumo_modes")))

        # Sort Arrow-backed trip columns in RAM and write XML in bulk.

        return records.write(cache / "queries.trips.xml.gz", types, write_xml=False)


def validate(context):
    fingerprint = hashlib.sha256()
    for name in ("demand_tools.py", "tools.py"):
        fingerprint.update(Path(__file__).with_name(name).read_bytes())
    fingerprint.update((Path(__file__).parents[1] / "cutter" / "demand_tools.py").read_bytes())
    return [config_tools.external_token(context, ("sumo_assignment_trip_files",)), fingerprint.hexdigest()]


def _population_batches(context):
    # The producer returns a tiny manifest of IPC paths, never a pickled population table.
    population = context.stage("sumo.scenario.population.population")
    if "vehicle_trips" in population:
        path = Path(population["vehicle_trips"])
        logger.info("Reading memory-mapped SUMO IPC demand (%.1f MB).", path.stat().st_size / 1e6)
        yield from read_table(path).iter_slices(BATCH_SIZE)
    else:
        # Accept in-memory external fixtures and older callers at the same conversion boundary.
        yield from population_trip_batches(population, context.config("sumo_population_crs"),
                                          modes.selected(context.config("sumo_modes")))


def _population_trips(context):
    yield from trip_records(_population_batches(context), context.config("sumo_assignment_all_as_car"))


def _read_vehicle_trips(filename, types, records, selected=None):
    for element in elements(filename, ("vType", "trip", "flow", "vehicle", "person", "vTypeDistribution")):
        if element.tag == "vType":
            type_id = element.get("id")
            if type_id in types and _type_definition(types[type_id]) != _type_definition(element):
                raise ValueError(f"Conflicting vehicle type definition: {type_id}")
            types[type_id] = copy.deepcopy(element)
        elif element.tag == "trip":
            if len(element) != len(element.findall("param")):
                raise ValueError("Assignment trip inputs may contain params but not scheduled stops")
            attributes = dict(element.attrib)
            if records.endpoint_resolver is not None and any(key in attributes for key in ("via", "viaXY", "viaLonLat")):
                raise ValueError("Swiss border clipping requires individual OD requests without via constraints; "
                                 "set clip_trips_to_border_in_sumo: false for constrained trip files")
            definition = types.get(attributes.get("type"))
            vehicle_class = "passenger" if definition is None else definition.get("vClass", "passenger")
            if not modes.includes(modes.vehicle_mode(vehicle_class), selected):
                continue
            attributes["id"] = identifier(attributes["id"])
            attributes["depart"] = str(seconds(attributes["depart"]))
            parameters = {param.get("key"): param.get("value") for param in element.findall("param")}
            parameters.setdefault("source_trip_id", attributes["id"])
            parameters.setdefault("mode", modes.vehicle_mode(vehicle_class))
            records.append({"attributes": attributes, "parameters": parameters, "load": True})
        else:
            raise ValueError(f"Assignment inputs require individual <trip> elements, found <{element.tag}> in {filename}")


def _type_definition(element):
    """Compare type semantics independently of formatting and attribute order."""
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(_type_definition(child) for child in element if isinstance(child.tag, str)))
