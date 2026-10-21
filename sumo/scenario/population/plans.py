"""Export person attributes and daily plans together in native SUMO route XML."""

import logging
from pathlib import Path

from sumo.scenario.population.tools import (
    activity_coordinates, identifier, lonlat_transformer, scalar_text, seconds, to_lonlat,
)
from sumo.scenario.population.writer import PopulationWriter
from sumo.scenario import modes

logger = logging.getLogger("synpp")
SUMO_MODES = {"car": "car", "bike": "bicycle", "bicycle": "bicycle", "pt": "public", "taxi": "taxi"}


def configure(context):
    modes.configure(context)
    context.stage("sumo.scenario.population.population")
    context.config("sumo_population_crs", "EPSG:2056")
    context.config("sumo_population_end_time", 30 * 3600)
    context.config("sumo_car_passenger_mode", "ride")
    context.config("population_compresslevel", 1)


def execute(context):

    # Load prepared tables and create a geographic coordinate transformer.

    population = context.stage("sumo.scenario.population.population")
    transformer = lonlat_transformer(context.config("sumo_population_crs"))
    day_end = seconds(context.config("sumo_population_end_time"), "Population end time")
    passenger_mode = context.config("sumo_car_passenger_mode")
    if passenger_mode not in {"ride", "taxi"}:
        raise ValueError("sumo_car_passenger_mode must be 'ride' or 'taxi'")
    output = Path(context.path()).resolve() / "population.rou.xml.gz"

    # Stream complete activity chains, preserving person and trip identifiers.

    activity_groups = iter(population["activities"].groupby("person_id", sort=False))
    count = 0
    with PopulationWriter(output, context.config("population_compresslevel")) as writer:
        for row in population["persons"].itertuples(index=False, name=None):
            person = dict(zip(population["persons"].columns, row))
            person_id, frame = next(activity_groups)
            if person_id != person["person_id"]:
                raise ValueError("SUMO persons and activity chains are not aligned")
            activities = frame.to_dict("records")
            segments = modes.plan_segments(activities, modes.selected(context.config("sumo_modes")))
            for index, segment in enumerate(segments):
                written_person = dict(person)
                if len(segments) > 1 or len(segment) != len(activities):
                    written_person["source_person_id"] = person["person_id"]
                    written_person["person_id"] = f"{person['person_id']}__sumo_part_{index}"
                _write_person(writer, written_person, segment, transformer, day_end, passenger_mode)
                count += 1

    # Return the combined population and plan file for routing or simulation setup.

    logger.info("Wrote %d SUMO persons with their plans to %s", count, output)
    return str(output)


def _write_person(writer, person, activities, transformer, day_end, passenger_mode):
    if not activities:
        raise ValueError(f"Person {person['person_id']} has no activity chain")
    coordinates = [to_lonlat(transformer, activity_coordinates(activity, person)) for activity in activities]
    departure = activities[0].get("start_time")
    departure = 0 if scalar_text(departure) is None else seconds(departure)
    previous_end = departure
    with writer.person(person["person_id"], departure, person):
        for index, activity in enumerate(activities):
            final = index == len(activities) - 1
            end = activity.get("end_time")
            if scalar_text(end) is None:
                if not final:
                    raise ValueError(f"Missing departure for person {person['person_id']}, activity {activity['activity_index']}")
                end = max(day_end, previous_end)
            end = seconds(end, "Activity end time")
            start = activity.get("start_time")
            if scalar_text(start) is not None:
                start = seconds(start, "Activity start time")
                if start < previous_end or end < start:
                    raise ValueError(f"Inconsistent activity/trip times for person {person['person_id']}, activity {activity['activity_index']}")
            elif end < previous_end:
                raise ValueError(f"Decreasing activity times for person {person['person_id']}")
            metadata = {key: value for key, value in activity.items() if key != "geometry"}
            writer.stop(coordinates[index], until=end, activity_type=activity["purpose"], attributes=metadata)
            previous_end = end
            if final:
                continue
            mode = scalar_text(activity.get("following_mode"))
            if mode is None:
                raise ValueError(f"Missing mode for person {person['person_id']}, activity {activity['activity_index']}")
            base_mode = mode.removesuffix("_loop")
            trip_index = activity.get("trip_index")
            if scalar_text(trip_index) is None:
                trip_index = activity["activity_index"]
            leg = {"person_id": person.get("source_person_id", person["person_id"]), "trip_index": trip_index,
                   "trip_id": activity.get("trip_id"), "mode": mode, "departure_time": end,
                   "arrival_time": activities[index + 1].get("start_time")}
            origin, destination = coordinates[index:index + 2]
            if base_mode == "walk":
                writer.walk(origin, destination, attributes=leg)
            elif base_mode == "car_passenger" and passenger_mode == "ride":
                vehicle = activity.get("passenger_vehicle_id")
                writer.ride(origin, destination, lines="ANY" if scalar_text(vehicle) is None else identifier(vehicle), attributes=leg)
            else:
                selected = "taxi" if base_mode == "car_passenger" else SUMO_MODES.get(base_mode)
                if selected is None:
                    raise ValueError(f"Unsupported SUMO population mode {mode!r} for person {person['person_id']}")
                writer.person_trip(origin, destination, selected, attributes=leg)
