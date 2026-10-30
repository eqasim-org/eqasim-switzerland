"""
Stage: analysis.counts.cross_border_flow_events

Reads output_events.xml.gz once for analysis.counts.cross_border_flow_cars and
analysis.counts.cross_border_flow_pt: in the same pass, it counts the car link
entries (per link and hour) and the PT riders between consecutive stops (per
stop-to-stop segment and hour), in total and for each rider group (foreign
cross-border, Swiss-resident cross-border and France-resident agents).

The counts are not scaled by the sample size: the two map stages do it.
Writes car_link_entries_hourly.csv.gz, pt_segment_riders_hourly.csv.gz and
pt_segment_lines.csv.gz into the stage cache folder.
"""

import logging
import os
from collections import Counter, defaultdict

import pandas as pd
import xml.etree.ElementTree as ET
import xopen

from .paths import configure_simulation_path, get_simulation_path

logger = logging.getLogger("synpp")

CAR_VEHICLE_SUFFIX          = ":car"
LINK_ENTRY_EVENT_TYPES      = {"entered link"}
PERSON_ENTERS_VEHICLE_EVENT = "PersonEntersVehicle"
PERSON_LEAVES_VEHICLE_EVENT = "PersonLeavesVehicle"
VEHICLE_DEPARTS_EVENT       = "VehicleDepartsAtFacility"
VEHICLE_ARRIVES_EVENT       = "VehicleArrivesAtFacility"
PROGRESS_EVERY              = 2_000_000
CLEAR_EVERY                 = 10_000

# Rider groups, the column names of the counts
GROUPS = ["crossborder", "swiss_crossborder", "french"]


def configure(context):
    configure_simulation_path(context)


def execute(context):

    # Resolve the current simulation files and mutually exclusive traveller groups.

    from .cross_border_flow_cars import (
        _find_output_file, load_cross_border_person_ids, load_french_resident_person_ids,
        load_swiss_resident_cross_border_person_ids,
    )
    from .cross_border_flow_pt import load_transit_schedule

    simulation_path = get_simulation_path(context)
    persons_path  = _find_output_file(simulation_path, "output_persons.csv.gz")
    events_path   = _find_output_file(simulation_path, "output_events.xml.gz")
    schedule_path = _find_output_file(simulation_path, "output_transitSchedule.xml.gz")

    cross_border_ids      = load_cross_border_person_ids(persons_path)
    swiss_crossborder_ids = load_swiss_resident_cross_border_person_ids(persons_path)
    french_resident_ids   = load_french_resident_person_ids(persons_path)
    logger.info(
        "Found %d cross-border persons (%d Swiss-resident) and %d non-cross-border France-resident persons in %s",
        len(cross_border_ids), len(swiss_crossborder_ids), len(french_resident_ids), persons_path,
    )
    group_ids = {
        "crossborder":       cross_border_ids - swiss_crossborder_ids,
        "swiss_crossborder": swiss_crossborder_ids,
        "french":            french_resident_ids,
    }

    line_labels = {}
    _, _, vehicle_lines = load_transit_schedule(schedule_path, line_labels=line_labels)

    # Scan once; expand counts only in the consuming map stages.

    car, pt, lines_by_segment = count_events(events_path, group_ids, vehicle_lines, line_labels)
    logger.info("Counted car entries on %d (link, hour) and PT riders on %d (segment, hour)", len(car[0]), len(pt[0]))

    paths = dict(
        car_hourly   = os.path.join(context.path(), "car_link_entries_hourly.csv.gz"),
        pt_hourly    = os.path.join(context.path(), "pt_segment_riders_hourly.csv.gz"),
        pt_lines     = os.path.join(context.path(), "pt_segment_lines.csv.gz"),
    )
    _counts_frame(car, ["link_id", "hour"]).to_csv(paths["car_hourly"], index = False)
    _counts_frame(pt, ["from_stop", "to_stop", "hour"]).to_csv(paths["pt_hourly"], index = False)
    pd.DataFrame(
        [(a, b, ", ".join(sorted(lines))) for (a, b), lines in lines_by_segment.items()],
        columns = ["from_stop", "to_stop", "lines"],
    ).to_csv(paths["pt_lines"], index = False)

    return paths


def _counts_frame(counts, keys):
    total_counts, group_counts = counts
    rows = [
        (*key, count, *(group_counts[group].get(key, 0) for group in GROUPS))
        for key, count in total_counts.items()
    ]
    return pd.DataFrame(rows, columns = [*keys, "total", *GROUPS])


def count_events(events_path, group_ids, vehicle_lines, line_labels=None):
    """One pass over the events. Returns
    - car: (total Counter, {group: Counter}) of the car link entries keyed by (link id, hour)
    - pt: (total Counter, {group: Counter}) of the PT riders keyed by (from stop, to stop, hour)
    - lines_by_segment: {(from stop, to stop): set of line names}"""
    car_total = Counter()
    car_groups = {group: Counter() for group in GROUPS}
    pt_total = Counter()
    pt_groups = {group: Counter() for group in GROUPS}
    lines_by_segment = defaultdict(set)

    onboard = defaultdict(set)
    last_departure_facility = {}
    drivers = {}
    active_lines = {}
    transit_vehicles = set(vehicle_lines)
    line_labels = line_labels or {}

    processed = 0
    with xopen.xopen(events_path, "rb") as f:
        parser = ET.iterparse(f, events = ["start", "end"])
        _, root = next(parser)  # <events> start tag, kept to periodically release finished events

        for event, elem in parser:
            if event != "end" or elem.tag != "event":
                continue

            processed += 1
            if processed % PROGRESS_EVERY == 0:
                logger.info("Processed %d events...", processed)

            attrib = elem.attrib
            etype = attrib.get("type")

            if etype in LINK_ENTRY_EVENT_TYPES:
                vehicle = attrib.get("vehicle", "")
                if vehicle.endswith(CAR_VEHICLE_SUFFIX):
                    key = (attrib["link"], int(float(attrib["time"]) // 3600) % 24)
                    car_total[key] += 1

                    person_id = vehicle[: -len(CAR_VEHICLE_SUFFIX)]
                    for group, ids in group_ids.items():
                        if person_id in ids:
                            car_groups[group][key] += 1

            elif etype == "TransitDriverStarts":
                # Vehicles can serve several lines during the day. The event identifies
                # both the driver to exclude and the line currently being operated.
                vehicle = attrib["vehicleId"]
                transit_vehicles.add(vehicle)
                drivers[vehicle] = attrib["driverId"]
                onboard.get(vehicle, set()).discard(attrib["driverId"])
                line = attrib.get("transitLineId")
                active_lines[vehicle] = line_labels.get(line, line) or vehicle_lines.get(vehicle)
                last_departure_facility.pop(vehicle, None)
            elif etype == PERSON_ENTERS_VEHICLE_EVENT:
                vehicle, person = attrib["vehicle"], attrib["person"]
                if vehicle in transit_vehicles and person != drivers.get(vehicle):
                    onboard[vehicle].add(person)
            elif etype == PERSON_LEAVES_VEHICLE_EVENT:
                vehicle = attrib["vehicle"]
                if vehicle in onboard:
                    onboard[vehicle].discard(attrib["person"])
                    if not onboard[vehicle]:
                        del onboard[vehicle]
            elif etype == VEHICLE_DEPARTS_EVENT:
                last_departure_facility[attrib["vehicle"]] = attrib["facility"]
            elif etype == VEHICLE_ARRIVES_EVENT:
                vehicle = attrib["vehicle"]
                arrival_facility = attrib["facility"]
                departure_facility = last_departure_facility.get(vehicle)

                if departure_facility is not None and departure_facility != arrival_facility:
                    riders = onboard.get(vehicle, ())
                    if riders:
                        segment = (departure_facility, arrival_facility)
                        key = (*segment, int(float(attrib["time"]) // 3600) % 24)

                        pt_total[key] += len(riders)
                        for group, ids in group_ids.items():
                            n = sum(1 for person in riders if person in ids)
                            if n:
                                pt_groups[group][key] += n

                        line = active_lines.get(vehicle, vehicle_lines.get(vehicle))
                        if line:
                            lines_by_segment[segment].add(line)

            elem.clear()
            if processed % CLEAR_EVERY == 0:
                root.clear()

    logger.info("Finished streaming %d events from %s", processed, events_path)
    return (car_total, car_groups), (pt_total, pt_groups), lines_by_segment
