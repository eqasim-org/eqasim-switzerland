"""Prepare vehicle loading demand and car-network queries for every source trip."""

import copy
from pathlib import Path

from lxml import etree

from sumo.config import tools as config_tools
from sumo.scenario.tools import elements, write_trips
from sumo.scenario import modes
from sumo.scenario.population.tools import (
    activity_coordinates, identifier, lonlat_transformer, scalar_text, seconds, to_lonlat,
)


def configure(context):
    modes.configure(context)
    context.config("sumo_assignment_trip_files", None)
    if context.config("sumo_assignment_trip_files") is None:
        context.stage("sumo.scenario.population.population")
        context.stage("sumo.scenario.population.trips")
    context.config("sumo_assignment_all_as_car", False)
    context.config("sumo_population_crs", "EPSG:2056")


def execute(context):

    # Read supplied vehicle trips or construct queries from the synthetic population.

    types = {"sumo_query_car": etree.Element("vType", id="sumo_query_car", vClass="passenger", speedDev="0")}
    records = []
    files = context.config("sumo_assignment_trip_files")
    if files is None:
        records.extend(_population_trips(context))
        files = [context.stage("sumo.scenario.population.trips")]
    for filename in config_tools.file_list(files):
        _read_vehicle_trips(filename, types, records, modes.selected(context.config("sumo_modes")))

    # Preserve source keys and sort all queries by their requested departure time.

    ids = [record["attributes"]["id"] for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate trip IDs across SUMO assignment inputs")
    records.sort(key=lambda record: (float(record["attributes"]["depart"]), record["attributes"]["id"]))
    for record in records:
        record["parameters"]["_sumo_loads_network"] = "true" if record["load"] else "false"
    path = Path(context.path()).resolve() / "queries.trips.xml.gz"
    write_trips(path, types, records)

    # Return every query together with the number of trips that contribute road flow.

    return {"queries": str(path), "count": len(records), "loading_count": sum(record["load"] for record in records),
            "first_departure": None if not records else float(records[0]["attributes"]["depart"]),
            "last_departure": None if not records else float(records[-1]["attributes"]["depart"])}


def validate(context):
    return config_tools.external_token(context, ("sumo_assignment_trip_files",))


def _population_trips(context):
    population = context.stage("sumo.scenario.population.population")
    transformer = lonlat_transformer(context.config("sumo_population_crs"))
    selected = modes.selected(context.config("sumo_modes"))
    persons = {person.person_id: person._asdict() for person in population["persons"].itertuples(index=False)}
    for person_id, frame in population["activities"].groupby("person_id", sort=False):
        activities = frame.to_dict("records")
        coordinates = [to_lonlat(transformer, activity_coordinates(activity, persons[person_id])) for activity in activities]
        for index, activity in enumerate(activities[:-1]):
            mode = scalar_text(activity.get("following_mode"))
            if mode is None:
                raise ValueError(f"Missing source mode for person {person_id}")
            if not modes.includes(mode, selected):
                continue
            trip_index = activity.get("trip_index")
            trip_index = activity["activity_index"] if scalar_text(trip_index) is None else trip_index
            depart = seconds(activity["end_time"])
            arrival = activities[index + 1].get("start_time")
            reference = None if scalar_text(arrival) is None else seconds(arrival) - depart
            yield {
                "attributes": {"id": f"person_{identifier(person_id)}_{identifier(trip_index)}", "depart": str(depart),
                               "fromLonLat": coordinates[index], "toLonLat": coordinates[index + 1], "type": "sumo_query_car"},
                "parameters": {"person_id": person_id, "trip_index": trip_index, "source_trip_id": activity.get("trip_id"),
                               "mode": mode, "source_travel_time_s": reference},
                "load": context.config("sumo_assignment_all_as_car") or mode.removesuffix("_loop") in {"car", "taxi"},
            }


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
            definition = types.get(attributes.get("type"))
            vehicle_class = "passenger" if definition is None else definition.get("vClass", "passenger")
            if not modes.includes(modes.vehicle_mode(vehicle_class), selected):
                continue
            attributes["id"] = identifier(attributes["id"])
            attributes["depart"] = str(seconds(attributes["depart"]))
            parameters = {param.get("key"): param.get("value") for param in element.findall("param")}
            parameters.setdefault("source_trip_id", attributes["id"])
            parameters.setdefault("mode", "road_vehicle")
            records.append({"attributes": attributes, "parameters": parameters, "load": True})
        else:
            raise ValueError(f"Assignment inputs require individual <trip> elements, found <{element.tag}> in {filename}")


def _type_definition(element):
    """Compare type semantics independently of formatting and attribute order."""
    return (element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
            tuple(_type_definition(child) for child in element if isinstance(child.tag, str)))
