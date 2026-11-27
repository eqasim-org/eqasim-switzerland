"""Export person attributes and daily plans together in native SUMO route XML."""

import logging
from itertools import groupby
from pathlib import Path

from sumo.functions.scenario import modes
from sumo.functions.scenario.population.tools import (
    lonlat_transformer,
    seconds,
)
from sumo.functions.scenario.population.writer import PopulationWriter
from sumo.functions.tables import as_polars

logger = logging.getLogger("synpp")

SUMO_MODES = {
    "car": "car",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "pt": "public",
    "taxi": "taxi",
}

from sumo.functions.macro_scenario.plans import _write_person
from sumo.functions.monitoring.runtime import monitored
from sumo.functions.source import source_digest


def configure(context):
    modes.configure(context)
    context.stage("sumo.macro_scenario.population")
    context.config("sumo_population_crs", "EPSG:2056")
    context.config("sumo_population_end_time", 30 * 3600)
    context.config("sumo_car_passenger_mode", "ride")
    context.config("population_compresslevel", 1)


@monitored
def execute(context):

    # Load prepared tables and create a geographic coordinate transformer.

    population = context.stage("sumo.macro_scenario.population")
    transformer = lonlat_transformer(context.config("sumo_population_crs"))
    day_end = seconds(context.config("sumo_population_end_time"), "Population end time")
    passenger_mode = context.config("sumo_car_passenger_mode")
    if passenger_mode not in {"ride", "taxi"}:
        raise ValueError("sumo_car_passenger_mode must be 'ride' or 'taxi'")
    output = Path(context.path()).resolve() / "population.rou.xml.gz"

    # Stream complete activity chains, preserving person and trip identifiers.

    persons, prepared_activities = (
        as_polars(population["persons"]),
        as_polars(population["activities"]),
    )
    activity_groups = iter(
        groupby(
            prepared_activities.iter_rows(named=True), key=lambda row: row["person_id"]
        )
    )
    count = 0
    with PopulationWriter(output, context.config("population_compresslevel")) as writer:
        for person in persons.iter_rows(named=True):
            person_id, frame = next(activity_groups)
            if person_id != person["person_id"]:
                raise ValueError("SUMO persons and activity chains are not aligned")
            activities = list(frame)
            segments = modes.plan_segments(
                activities, modes.selected(context.config("sumo_modes"))
            )
            for index, segment in enumerate(segments):
                written_person = dict(person)
                if len(segments) > 1 or len(segment) != len(activities):
                    written_person["source_person_id"] = person["person_id"]
                    written_person["person_id"] = (
                        f"{person['person_id']}__sumo_part_{index}"
                    )
                _write_person(
                    writer,
                    written_person,
                    segment,
                    transformer,
                    day_end,
                    passenger_mode,
                )
                count += 1

    # Return the combined population and plan file for routing or simulation setup.

    logger.info("Wrote %d SUMO persons with their plans to %s", count, output)
    return str(output)


def validate(context):
    return source_digest("sumo.functions.macro_scenario.plans")
