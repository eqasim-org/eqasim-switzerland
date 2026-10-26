"""Streaming writer for native SUMO persons, plans, vehicle types and trips."""

from contextlib import contextmanager

from lxml import etree

from sumo.scenario.population.tools import identifier, scalar_text, seconds, xml_stream


class PopulationWriter:
    """Write one person at a time instead of keeping the full XML tree in memory."""

    def __init__(self, path, compresslevel=1):
        self.path = path
        self.compresslevel = compresslevel

    def __enter__(self):
        from contextlib import ExitStack

        self._stack = ExitStack()
        try:
            stream = self._stack.enter_context(xml_stream(self.path, self.compresslevel))
            self._xml = self._stack.enter_context(etree.xmlfile(stream, encoding="utf-8"))
            self._xml.write_declaration()
            self._stack.enter_context(self._xml.element("routes", nsmap={"xsi": "http://www.w3.org/2001/XMLSchema-instance"},
                attrib={"{http://www.w3.org/2001/XMLSchema-instance}noNamespaceSchemaLocation":
                        "http://sumo.dlr.de/xsd/routes_file.xsd"}))
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exception):
        return self._stack.__exit__(*exception)

    @contextmanager
    def person(self, person_id, depart=0, attributes=None):
        with self._xml.element("person", id=identifier(person_id), depart=str(seconds(depart))):
            self.parameters(attributes or {})
            yield self

    def parameters(self, attributes):
        for key, value in attributes.items():
            text = scalar_text(value)
            if text is not None:
                self._xml.write(etree.Element("param", key=str(key), value=text))

    def stop(self, lonlat=None, until=None, duration=None, activity_type=None,
             edge=None, lane=None, attributes=None):
        location = {}
        if lonlat is not None:
            location["lon"], location["lat"] = lonlat.split(",")
        if edge is not None:
            location["edge"] = str(edge)
        if lane is not None:
            location["lane"] = str(lane)
        if until is not None:
            location["until"] = str(seconds(until, "Activity end"))
        if duration is not None:
            location["duration"] = str(seconds(duration, "Activity duration"))
        if activity_type is not None:
            location["actType"] = str(activity_type)
        with self._xml.element("stop", attrib=location):
            self.parameters(attributes or {})

    def person_trip(self, origin, destination, modes, attributes=None, vehicle_types=None):
        arguments = {"fromLonLat": origin, "toLonLat": destination, "modes": modes}
        if vehicle_types:
            arguments["vTypes"] = vehicle_types
        with self._xml.element("personTrip", attrib=arguments):
            self.parameters(attributes or {})

    def walk(self, origin, destination, attributes=None):
        with self._xml.element("walk", fromLonLat=origin, toLonLat=destination):
            self.parameters(attributes or {})

    def ride(self, origin, destination, lines="ANY", attributes=None):
        with self._xml.element("ride", fromLonLat=origin, toLonLat=destination, lines=str(lines)):
            self.parameters(attributes or {})

    def vehicle_type(self, type_id, vehicle_class, **attributes):
        arguments = {"id": identifier(type_id), "vClass": vehicle_class}
        arguments.update({key: scalar_text(value) for key, value in attributes.items() if scalar_text(value) is not None})
        self._xml.write(etree.Element("vType", attrib=arguments))

    def trip(self, trip_id, depart, origin, destination, vehicle_type=None, attributes=None):
        arguments = {"id": identifier(trip_id), "depart": str(seconds(depart)),
                     "fromLonLat": origin, "toLonLat": destination}
        if vehicle_type is not None:
            arguments["type"] = str(vehicle_type)
        with self._xml.element("trip", attrib=arguments):
            self.parameters(attributes or {})

    def element(self, element):
        """Write an existing native type/definition without reinterpreting it."""
        self._xml.write(element)

    def vehicle(self, attributes, edges, exit_times=None, parameters=None, route_attributes=None):
        arguments = {str(key): str(value) for key, value in attributes.items()}
        route = dict(route_attributes or {})
        route["edges"] = " ".join(edges)
        if exit_times is not None:
            route["exitTimes"] = " ".join(format(value, ".12g") for value in exit_times)
        with self._xml.element("vehicle", attrib=arguments):
            self._xml.write(etree.Element("route", attrib=route))
            self.parameters(parameters or {})
