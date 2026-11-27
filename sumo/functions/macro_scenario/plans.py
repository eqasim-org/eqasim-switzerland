"""Export person attributes and daily plans together in native SUMO route XML."""

from sumo.functions.scenario.population.tools import (
    activity_coordinates,
    identifier,
    scalar_text,
    seconds,
    to_lonlat,
)

SUMO_MODES = {
    "car": "car",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "pt": "public",
    "taxi": "taxi",
}


def _write_person(writer, person, activities, transformer, day_end, passenger_mode):
    if not activities:
        raise ValueError(f"Person {person['person_id']} has no activity chain")
    coordinates = [
        to_lonlat(transformer, activity_coordinates(activity, person))
        for activity in activities
    ]
    departure = activities[0].get("start_time")
    departure = 0 if scalar_text(departure) is None else seconds(departure)
    previous_end = departure
    with writer.person(person["person_id"], departure, person):
        for index, activity in enumerate(activities):
            final = index == len(activities) - 1
            end = activity.get("end_time")
            if scalar_text(end) is None:
                if not final:
                    raise ValueError(
                        f"Missing departure for person {person['person_id']}, activity {activity['activity_index']}"
                    )
                end = max(day_end, previous_end)
            end = seconds(end, "Activity end time")
            start = activity.get("start_time")
            if scalar_text(start) is not None:
                start = seconds(start, "Activity start time")
                if start < previous_end or end < start:
                    raise ValueError(
                        f"Inconsistent activity/trip times for person {person['person_id']}, activity {activity['activity_index']}"
                    )
            elif end < previous_end:
                raise ValueError(
                    f"Decreasing activity times for person {person['person_id']}"
                )
            metadata = {
                key: value for key, value in activity.items() if key != "geometry"
            }
            writer.stop(
                coordinates[index],
                until=end,
                activity_type=activity["purpose"],
                attributes=metadata,
            )
            previous_end = end
            if final:
                continue
            mode = scalar_text(activity.get("following_mode"))
            if mode is None:
                raise ValueError(
                    f"Missing mode for person {person['person_id']}, activity {activity['activity_index']}"
                )
            base_mode = mode.removesuffix("_loop")
            trip_index = activity.get("trip_index")
            if scalar_text(trip_index) is None:
                trip_index = activity["activity_index"]
            leg = {
                "person_id": person.get("source_person_id", person["person_id"]),
                "trip_index": trip_index,
                "trip_id": activity.get("trip_id"),
                "mode": mode,
                "departure_time": end,
                "arrival_time": activities[index + 1].get("start_time"),
            }
            origin, destination = coordinates[index : index + 2]
            if base_mode == "walk":
                writer.walk(origin, destination, attributes=leg)
            elif base_mode == "car_passenger" and passenger_mode == "ride":
                vehicle = activity.get("passenger_vehicle_id")
                writer.ride(
                    origin,
                    destination,
                    lines="ANY"
                    if scalar_text(vehicle) is None
                    else identifier(vehicle),
                    attributes=leg,
                )
            else:
                selected = (
                    "taxi"
                    if base_mode == "car_passenger"
                    else SUMO_MODES.get(base_mode)
                )
                if selected is None:
                    raise ValueError(
                        f"Unsupported SUMO population mode {mode!r} for person {person['person_id']}"
                    )
                writer.person_trip(origin, destination, selected, attributes=leg)
