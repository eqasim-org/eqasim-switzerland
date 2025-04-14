import numpy as np
from xml.sax.saxutils import escape
from typing import Dict, Union

class XmlWriter:
    def __init__(self, writer):
        self.writer = writer
        self.scope = []
        self.indent = 0

    def _write_line(self, content):
        self._write_indent()
        self._write(content + "\n")

    def _write_indent(self):
        self._write("  " * self.indent)

    def _write(self, content):
        self.writer.write(bytes(content, "utf-8"))

    def _require_scope(self, expected_scope):
        if expected_scope is None:
            if not len(self.scope) == 0:
                raise RuntimeError("Execpted initial scope")
            else:
                return
        else:
            if not type(expected_scope) == tuple and not type(expected_scope) == list:
                expected_scope = [expected_scope]

            if len(self.scope) == 0 or not self.scope[-1] in expected_scope:
                raise RuntimeError("Expected different scope")

    def _push_scope(self, scope):
        self.scope.append(scope)
        self.indent += 1

    def _pop_scope(self):
        del self.scope[-1]
        self.indent -= 1

    def yes_no(self, value):
        return "yes" if value else "no"

    def true_false(self, value):
        return "true" if value else "false"

    def time(self, time):
        time = int(time)
        hours = time // 3600
        minutes = (time % 3600) // 60
        seconds = (time % 60)
        return "%02d:%02d:%02d" % (hours, minutes, seconds)

    def location(self, x, y, facility_id = None):
        return (x, y,
                None if facility_id is None or (type(facility_id) == float and np.isnan(facility_id)) else facility_id)
    
    @staticmethod
    def get_java_type(python_type: type):
        if 'float' in str(python_type):
            return "java.lang.Double"
        elif 'int64' in str(python_type):
            return "java.lang.Long"
        elif 'int' in str(python_type):
            return "java.lang.Integer"
        elif 'str' in str(python_type):
            return "java.lang.String"
        elif 'bool' in str(python_type):
            return "java.lang.Boolean"
        else:
            return "java.lang.Object"

def _write_preface_attributes(writer, attributes):
    if len(attributes) > 0:
        writer._write_line('<attributes>')
        writer.indent += 1

        for item in attributes.items():
            writer._write_line('<attribute name="%s" class="java.lang.String">%s</attribute>' % item)

        writer.indent -= 1
        writer._write_line('</attributes>')
        
class PopulationWriter(XmlWriter):
    POPULATION_SCOPE = 0
    # FINISHED_SCOPE = 1
    PERSON_SCOPE = 2
    PLAN_SCOPE = 3
    ATTRIBUTES_SCOPE = 4
    ACTIVITY_SCOPE = 5

    def __init__(self, writer):
        XmlWriter.__init__(self, writer)

    def start_population(self):
        self._require_scope(None)

        self._write_line('<?xml version="1.0" encoding="utf-8"?>')
        self._write_line('<!DOCTYPE population SYSTEM "http://www.matsim.org/files/dtd/population_v6.dtd">')
        self._write_line('<population desc="Switzerland Baseline">')

        self._push_scope(self.POPULATION_SCOPE)

    def end_population(self):
        self._require_scope(self.POPULATION_SCOPE)
        self._write_line('</population>')
        self._pop_scope()

    def start_person(self, person_id):
        self._require_scope(self.POPULATION_SCOPE)
        self._write_line('<person id="%s">' % person_id)
        self._push_scope(self.PERSON_SCOPE)

    def end_person(self):
        self._require_scope(self.PERSON_SCOPE)
        self._pop_scope()
        self._write_line('</person>')

    def start_attributes(self):
        self._require_scope([self.PERSON_SCOPE, self.ACTIVITY_SCOPE])
        self._write_line('<attributes>')
        self._push_scope(self.ATTRIBUTES_SCOPE)

    def end_attributes(self):
        self._require_scope(self.ATTRIBUTES_SCOPE)
        self._pop_scope()
        self._write_line('</attributes>')

    def add_attribute(self, name, type, value):
        self._require_scope(self.ATTRIBUTES_SCOPE)
        self._write_line('<attribute name="%s" class="%s">%s</attribute>' % (
            name, type, value
        ))

    def start_plan(self, selected):
        self._require_scope(self.PERSON_SCOPE)
        self._write_line('<plan selected="%s">' % self.yes_no(selected))
        self._push_scope(self.PLAN_SCOPE)

    def end_plan(self):
        self._require_scope(self.PLAN_SCOPE)
        self._pop_scope()
        self._write_line('</plan>')

    def _start_activity(self, type, location, start_time = None, end_time = None):
        self._write_indent()
        self._write('<activity ')
        self._write('type="%s" ' % type)
        self._write('x="%f" y="%f" ' % (location[0], location[1]))
        if location[2] is not None: self._write('facility="%s" ' % str(location[2]))
        if start_time is not None: self._write('start_time="%s" ' % self.time(start_time))
        if end_time is not None: self._write('end_time="%s" ' % self.time(end_time))

    def start_activity(self, type, location, start_time = None, end_time = None):
        self._require_scope(self.PLAN_SCOPE)
        self._start_activity(type, location, start_time, end_time)
        self._write('>\n')
        self._push_scope(self.ACTIVITY_SCOPE)

    def end_activity(self):
        self._require_scope(self.ACTIVITY_SCOPE)
        self._pop_scope()
        self._write_line('</activity>')

    def add_activity(self, type, location, start_time = None, end_time = None):
        self._require_scope(self.PLAN_SCOPE)

        self._start_activity(type, location, start_time, end_time)
        self._write('/>\n')

    def add_leg(self, mode, departure_time, travel_time):
        self._require_scope(self.PLAN_SCOPE)

        self._write_indent()
        self._write('<leg ')
        self._write('mode="%s" ' % mode)
        self._write('dep_time="%s" ' % self.time(departure_time))
        self._write('trav_time="%s" ' % self.time(travel_time))
        self._write('/>\n')

class HouseholdsWriter(XmlWriter):
    HOUSEHOLDS_SCOPE = 0
    FINISHED_SCOPE = 1
    HOUSEHOLD_SCOPE = 2
    ATTRIBUTES_SCOPE = 3

    def __init__(self, writer):
        XmlWriter.__init__(self, writer)

    def start_households(self):
        self._require_scope(None)
        self._write_line('<?xml version="1.0" encoding="utf-8"?>')
        self._write_line('<households xmlns="http://www.matsim.org/files/dtd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/households_v1.0.xsd">')
        self._push_scope(self.HOUSEHOLDS_SCOPE)

    def end_households(self):
        self._require_scope(self.HOUSEHOLDS_SCOPE)
        self._write_line('</households>')
        self._pop_scope()

    def start_household(self, household_id):
        self._require_scope(self.HOUSEHOLDS_SCOPE)
        self._write_line('<household id="%d">' % household_id)
        self._push_scope(self.HOUSEHOLD_SCOPE)

    def end_household(self):
        self._require_scope(self.HOUSEHOLD_SCOPE)
        self._pop_scope()
        self._write_line('</household>')

    def start_attributes(self):
        self._require_scope(self.HOUSEHOLD_SCOPE)
        self._write_line('<attributes>')
        self._push_scope(self.ATTRIBUTES_SCOPE)

    def end_attributes(self):
        self._require_scope(self.ATTRIBUTES_SCOPE)
        self._pop_scope()
        self._write_line('</attributes>')

    def add_attribute(self, name, type, value):
        self._require_scope(self.ATTRIBUTES_SCOPE)
        self._write_line('<attribute name="%s" class="%s">%s</attribute>' % (
            name, type, value
        ))

    def add_members(self, person_ids):
        self._require_scope(self.HOUSEHOLD_SCOPE)
        self._write_line('<members>')
        self.indent += 1
        for person_id in person_ids: self._write_line('<personId refId="%s" />' % person_id)
        self.indent -= 1
        self._write_line('</members>')

    def add_income(self, income):
        self._require_scope(self.HOUSEHOLD_SCOPE)
        self._write_line('<income currency="CHF" period="month">%f</income>' % income)

class FacilitiesWriter(XmlWriter):
    FACILITIES_SCOPE = 0
    FINISHED_SCOPE = 1
    FACILITY_SCOPE = 2

    def __init__(self, writer):
        XmlWriter.__init__(self, writer)

    def start_facilities(self):
        self._require_scope(None)
        self._write_line('<?xml version="1.0" encoding="utf-8"?>')
        self._write_line('<!DOCTYPE facilities SYSTEM "http://www.matsim.org/files/dtd/facilities_v1.dtd">')
        self._write_line('<facilities name="Facilities from different sources">')
        self._push_scope(self.FACILITIES_SCOPE)

    def end_facilities(self):
        self._require_scope(self.FACILITIES_SCOPE)
        self._write_line('</facilities>')
        self._pop_scope()

    def start_facility(self, facility_id, x, y):
        self._require_scope(self.FACILITIES_SCOPE)
        self._write_line('<facility id="%s" x="%f" y="%f">' % (
            str(facility_id), x, y
        ))
        self._push_scope(self.FACILITY_SCOPE)

    def end_facility(self):
        self._require_scope(self.FACILITY_SCOPE)
        self._pop_scope()
        self._write_line('</facility>')

    def add_activity(self, purpose):
        self._require_scope(self.FACILITY_SCOPE)
        self._write_line('<activity type="%s" />' % purpose)

class VehiclesWriter(XmlWriter):
    VEHICLES_SCOPE = 0
    FINISHED_SCOPE = 1

    def __init__(self, writer):
        XmlWriter.__init__(self, writer)

    def start_vehicles(self, attributes = {}):
        self._require_scope(None)
        self._write_line('<?xml version="1.0" encoding="utf-8"?>')
        self._write_line('<vehicleDefinitions xmlns="http://www.matsim.org/files/dtd" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.matsim.org/files/dtd http://www.matsim.org/files/dtd/vehicleDefinitions_v2.0.xsd">')

        self._push_scope(self.VEHICLES_SCOPE)
        self.indent += 1

        _write_preface_attributes(self, attributes)

    def end_vehicles(self):
        self._require_scope(self.VEHICLES_SCOPE)
        self.indent -= 1
        self._write_line('</vehicleDefinitions>')
        self._pop_scope()

    def add_type(self, vehicle_type_id, nb_seats = 4, length = 5.0, width = 1.0, pce = 1.0, mode = "car", attributes = {}, engine_attributes = {}):
        self._require_scope(self.VEHICLES_SCOPE)
        self._write_line('<vehicleType id="%s">' % str(vehicle_type_id))

        self.indent += 1

        if len(attributes) > 0:
            self._write_line('<attributes>')
            self.indent += 1
            for key, item in attributes.items():
                self._write_line('<attribute name="%s" class="java.lang.String">%s</attribute>' % (key, escape(item)))
            self.indent -= 1
            self._write_line('</attributes>')

        if not np.isnan(nb_seats):
            self._write_line('<capacity seats="%d" standingRoomInPersons="0" />' % nb_seats)

        self._write_line('<length meter="%f"/>' % length)
        self._write_line('<width meter="%f"/>' % width)

        if len(engine_attributes) > 0:
            self._write_line('<engineInformation>')
            self.indent += 1
            self._write_line('<attributes>')
            self.indent += 1
            for key, item in engine_attributes.items():
                self._write_line('<attribute name="%s" class="java.lang.String">%s</attribute>' % (key, escape(item)))
            self.indent -= 1
            self._write_line('</attributes>')
            self.indent -= 1
            self._write_line('</engineInformation>')

        if not np.isnan(pce):
            self._write_line('<passengerCarEquivalents pce="%f"/>' % pce)

        self._write_line('<networkMode networkMode="%s"/>' % mode)

        self.indent -= 1
        self._write_line('</vehicleType>')


    def add_vehicle(self, vehicle_id, type_id, attributes = {}):
        self._require_scope(self.VEHICLES_SCOPE)

        if len(attributes) > 0:
            self._write_line('<vehicle id="%s" type="%s">' % (str(vehicle_id), str(type_id)))
            self.indent += 1
            self._write_line('<attributes>')
            self.indent += 1
            for key, item in attributes.items():
                self._write_line('<attribute name="%s" class="java.lang.String">%s</attribute>' % (str(key), str(item)))
            self.indent -= 1
            self._write_line('</attributes>')
            self.indent -= 1
            self._write_line('</vehicle>')
        else:
            self._write_line('<vehicle id="%s" type="%s" />' % (str(vehicle_id), str(type_id)))




        
class NetworkWriter(XmlWriter):

    def __init__(self, writer, write_attrbs = True):
        XmlWriter.__init__(self, writer)
        self.write_attrbs = write_attrbs

    def start_network(self, attributes: Dict[str, str] = None):
        self._write_line('<?xml version="1.0" encoding="utf-8"?>')
        self._write_line('<!DOCTYPE network SYSTEM "http://www.matsim.org/files/dtd/network_v2.dtd">\n')
        self._write_line('<network>')
        self._write_line('\n<!-- ====================================================================== --> \n ')
        self.indent += 2
        if attributes is not None:
            self.write_attributes(attributes)

    def end_network(self):
        self.indent -= 2
        self._write_line("\n<!-- ====================================================================== -->")
        self._write_line('</network>')

        
    def start_nodes(self, attributes: Dict[str, str] = None):
        self._write_line('<nodes>')
        self.indent += 1
        if attributes is not None:
            self.write_attributes(attributes)

    def end_nodes(self):
        self.indent -= 1
        self._write_line('</nodes>')

        
    def start_links(self, attributes: Dict[str, str] = None):
        self._write_line('\n \n<links capperiod="01:00:00" effectivecellsize="7.5" effectivelanewidth="3.75"> \n \n')
        self.indent += 1
        if attributes is not None:
            self.write_attributes(attributes)

    def end_links(self):
        self.indent -= 1
        self._write_line('</links>')
   

    def write_node(self, _id: str, x: float, y:float):
        self._write_line(f'<node  id="{_id}"  x="{x}"  y="{y}"/>')        

    def write_link(self, _id: str, _from:str, to:str, length:float,
                   freespeed:float, capacity:int, permlanes:int,
                   oneway:int, modes=str,attributes: Dict[str, str] = None):
        
        self._write_line(f'<link id="{_id}" from="{_from}" to="{to}" length="{length}" freespeed="{freespeed}" capacity="{capacity}" permlanes="{permlanes}" oneway="{oneway}" modes="{modes}">')        
        if isinstance(attributes,dict) and self.write_attrbs:
            self.indent += 1
            self.start_attributes()
            for k, v in attributes.items():                
                self.add_attribute(k, v)
            self.end_attributes()
            self.indent -= 1
        self._write_line('</link>')

    def start_attributes(self):
        self._write_line('<attributes>')
        self.indent += 1

    def end_attributes(self):
        self.indent -= 1
        self._write_line('</attributes>')

    def add_attribute(self, name: str, value: Union[str, int, float, bool], typ: str = None):
        if not typ:
            typ = self.get_java_type(type(value))
        self._write_line(f'<attribute name="{name}" class="{typ}">{value}</attribute>')

    def write_attributes(self, attributes: dict):
        self.start_attributes()
        for name, value in attributes.items():
            self.add_attribute(name, value)
        self.end_attributes()

    def add_nodes(self,_id, x, y):
        self.start_nodes()
        list(map(self.write_node,_id,x,y))
        self.end_nodes()
    
    def add_links(self, _id, _from, to, length, freespeed, capacity, permlanes,
                   oneway, modes, attributes, write_attrbs=True):
        self.write_attrbs = write_attrbs
        self.start_links()
        list(map(self.write_link, _id, _from, to, length, freespeed, capacity, 
                 permlanes, oneway, modes, attributes))
        self.end_links()      
                            