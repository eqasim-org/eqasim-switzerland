"""Cache selected, untrimmed SUMO road demand before route completion."""

import copy
import logging
from pathlib import Path

from sumo.functions.scenario import modes
from sumo.functions.scenario.population.tools import (
    identifier,
    seconds,
)
from sumo.functions.scenario.population.vehicle_tools import (
    BATCH_SIZE,
    population_trip_batches,
    trip_records,
)
from sumo.functions.scenario.tools import elements
from sumo.functions.tables import read_table

logger = logging.getLogger("synpp")


def _population_batches(context):
    # The producer returns a tiny manifest of IPC paths, never a pickled population table.
    population = context.stage("sumo.macro_scenario.population")
    if "vehicle_trips" in population:
        path = Path(population["vehicle_trips"])
        logger.info(
            "Reading memory-mapped SUMO IPC demand (%.1f MB).",
            path.stat().st_size / 1e6,
        )
        yield from read_table(path).iter_slices(BATCH_SIZE)
    else:
        # Accept in-memory external fixtures and older callers at the same conversion boundary.
        yield from population_trip_batches(
            population,
            context.config("sumo_population_crs"),
            modes.selected(context.config("sumo_modes")),
        )


def _population_trips(context):
    yield from trip_records(
        _population_batches(context), context.config("sumo_assignment_all_as_car")
    )


def _read_vehicle_trips(filename, types, records, selected=None):
    for element in elements(
        filename, ("vType", "trip", "flow", "vehicle", "person", "vTypeDistribution")
    ):
        if element.tag == "vType":
            type_id = element.get("id")
            if type_id in types and _type_definition(
                types[type_id]
            ) != _type_definition(element):
                raise ValueError(f"Conflicting vehicle type definition: {type_id}")
            types[type_id] = copy.deepcopy(element)
        elif element.tag == "trip":
            if len(element) != len(element.findall("param")):
                raise ValueError(
                    "Assignment trip inputs may contain params but not scheduled stops"
                )
            attributes = dict(element.attrib)
            if records.endpoint_resolver is not None and any(
                key in attributes for key in ("via", "viaXY", "viaLonLat")
            ):
                raise ValueError(
                    "Swiss border clipping requires individual OD requests without via constraints; "
                    "set clip_trips_to_border_in_sumo: false for constrained trip files"
                )
            definition = types.get(attributes.get("type"))
            vehicle_class = (
                "passenger"
                if definition is None
                else definition.get("vClass", "passenger")
            )
            if not modes.includes(modes.vehicle_mode(vehicle_class), selected):
                continue
            attributes["id"] = identifier(attributes["id"])
            attributes["depart"] = str(seconds(attributes["depart"]))
            parameters = {
                param.get("key"): param.get("value")
                for param in element.findall("param")
            }
            parameters.setdefault("source_trip_id", attributes["id"])
            parameters.setdefault("mode", modes.vehicle_mode(vehicle_class))
            records.append(
                {"attributes": attributes, "parameters": parameters, "load": True}
            )
        else:
            raise ValueError(
                f"Assignment inputs require individual <trip> elements, found <{element.tag}> in {filename}"
            )


def _type_definition(element):
    """Compare type semantics independently of formatting and attribute order."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(
            _type_definition(child) for child in element if isinstance(child.tag, str)
        ),
    )
