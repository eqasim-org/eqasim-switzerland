import gzip
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

EVENTS_FILE = "output_events.xml.gz"
COUNTS_FILE = "pt_passenger_counts.csv.gz"
EVENTS_COUNTS_FILE = "pt_passenger_counts_from_events.csv.gz"

# PT events only: the person events are restricted to the transit vehicles (veh_...), the cars are not needed
_PT_EVENTS = (
    r'type="(TransitDriverStarts|VehicleArrivesAtFacility)"'
    r'|type="Person(Enters|Leaves)Vehicle" .*vehicle="veh_'
)

_ATTRIBUTES = re.compile(r'(\w+)="([^"]*)"')


def read_schedule_lines(schedule_path):
    """({stop facility id: name}, {transit line id: name}, {(transit line id, transit route id): 'first stop->last stop'})"""
    opener = gzip.open if schedule_path.endswith(".gz") else open

    stop_names, line_names, route_stops = {}, {}, {}
    line_id = route_id = stops = None

    with opener(schedule_path, "rb") as f:
        for event, element in ET.iterparse(f, events = ("start", "end")):
            if event == "start":
                if element.tag == "transitLine":
                    line_id = element.attrib["id"]
                    line_names[line_id] = element.attrib.get("name") or line_id
                elif element.tag == "transitRoute":
                    route_id, stops = element.attrib["id"], []
            elif element.tag == "stopFacility":
                stop_names[element.attrib["id"]] = element.attrib.get("name") or element.attrib["id"]
            elif element.tag == "stop" and stops is not None and "refId" in element.attrib:
                stops.append(element.attrib["refId"])
            elif element.tag == "transitRoute":
                route_stops[(line_id, route_id)] = stops
                route_id = stops = None

    route_directions = {
        key: f"{stop_names.get(stops[0], stops[0])}->{stop_names.get(stops[-1], stops[-1])}"
        for key, stops in route_stops.items() if stops
    }

    return stop_names, line_names, route_directions


def _pt_events(events_path):
    # grep keeps about 1% of the lines: much faster than parsing all the events in Python
    process = subprocess.Popen(
        f"zcat '{events_path}' | grep -a -E '{_PT_EVENTS}'", shell = True, stdout = subprocess.PIPE, text = True,
    )

    for line in process.stdout:
        yield dict(_ATTRIBUTES.findall(line))

    if process.wait() not in (0, 1):  # grep exits with 1 when nothing matches
        raise RuntimeError(f"Could not read the events of {events_path}")


def count_passengers(events_path):
    """Boardings and alightings per (stop facility, transit line, transit route, vehicle, hour of the arrival of the vehicle at the stop)."""
    vehicle_service = {}  # vehicle id -> (transit line id, transit route id)
    vehicle_stop    = {}  # vehicle id -> (stop facility the vehicle is at, time of its arrival)
    drivers         = set()
    counts          = defaultdict(lambda: [0, 0])

    for i, event in enumerate(_pt_events(events_path), start = 1):
        kind    = event["type"]
        vehicle = event.get("vehicle") or event.get("vehicleId")

        if kind == "TransitDriverStarts":
            drivers.add(event["driverId"])
            vehicle_service[vehicle] = (event["transitLineId"], event["transitRouteId"])
            vehicle_stop.pop(vehicle, None)
        elif kind == "VehicleArrivesAtFacility":
            vehicle_stop[vehicle] = (event["facility"], event["time"])
        elif event["person"] not in drivers and vehicle in vehicle_service and vehicle in vehicle_stop:
            line_id, route_id = vehicle_service[vehicle]
            stop_id, arrival = vehicle_stop[vehicle]
            key = (stop_id, line_id, route_id, vehicle, int(float(arrival) // 3600))
            counts[key][0 if kind == "PersonEntersVehicle" else 1] += 1

        if i % 5_000_000 == 0:
            print(f"  -> {i} PT events read", flush = True)

    return counts


def read_counts_from_events(matsim_output_folder, schedule_path, use_cache = True):
    """The content of pt_passenger_counts.csv.gz, rebuilt from output_events.xml.gz and the transit schedule. The
    result is saved next to the events and reused as long as they are not newer."""
    events_path = os.path.join(matsim_output_folder, EVENTS_FILE)
    cache_path  = os.path.join(matsim_output_folder, EVENTS_COUNTS_FILE)

    if not os.path.exists(events_path):
        raise FileNotFoundError(f"{events_path} not found in the MATSim output folder")

    if use_cache and os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(events_path):
        print(f"Passenger counts of the events read from {cache_path}")
        return pd.read_csv(cache_path, sep = ";")

    print(f"Counting the boardings and alightings in {events_path} ...")
    counts = count_passengers(events_path)
    _, line_names, route_directions = read_schedule_lines(schedule_path)

    rows = []
    for (stop_id, line_id, route_id, vehicle_id, hour), (boardings, alightings) in counts.items():
        direction = route_directions.get((line_id, route_id), "")
        rows.append((
            f"{hour:02d}:00", stop_id, line_id, line_names.get(line_id, line_id), route_id, direction, direction,
            vehicle_id, boardings, alightings,
        ))

    columns = ["time_bin", "stop_id", "line_id", "line_name", "route_id", "route_direction", "line_main_direction",
               "vehicle_id", "boardings", "alightings"]
    result = pd.DataFrame(rows, columns = columns)

    try:
        result.to_csv(cache_path, sep = ";", index = False)
    except OSError as error:
        print(f"Could not save the counts of the events ({error}): they will be counted again next time")

    return result
