"""Cache selected, untrimmed SUMO road demand before route completion."""

import hashlib
import logging
from pathlib import Path

from lxml import etree

from sumo.functions.config import tools as config_tools
from sumo.functions.scenario import modes
from sumo.functions.scenario.demand_tools import TripSpool

logger = logging.getLogger("synpp")

from sumo.functions.macro_scenario.trips import _population_batches, _read_vehicle_trips
from sumo.functions.macro_scenario.trips import _population_trips as _population_trips
from sumo.functions.macro_scenario.trips import _type_definition as _type_definition
from scripts.monitoring import monitored
from sumo.functions.source import source_digest


def configure(context):
    modes.configure(context)
    context.config("sumo_assignment_trip_files", None)
    if context.config("sumo_assignment_trip_files") is None:
        context.stage("sumo.macro_scenario.population")
        context.stage("sumo.macro_scenario.vehicles")
    modes.require_mode_choice(context)
    context.config("clip_trips_to_border_in_sumo", True)
    if (
        context.config("clip_trips_to_border_in_sumo")
        and context.config("sumo_assignment_trip_files") is not None
    ):
        config_tools.configure_network(context)
    context.config("sumo_assignment_all_as_car", False)
    context.config("sumo_population_crs", "EPSG:2056")


@monitored
def execute(context):

    # Read supplied vehicle trips or construct queries from the synthetic population.

    types = {
        "sumo_query_car": etree.Element(
            "vType", id="sumo_query_car", vClass="passenger", speedDev="0"
        )
    }
    cache = Path(context.path()).resolve()
    from sumo.functions.cutter.demand_tools import EndpointResolver

    resolver = (
        EndpointResolver(lambda: config_tools.network_path(context))
        if context.config("clip_trips_to_border_in_sumo")
        else None
    )
    with TripSpool(cache / "demand.arrow", endpoint_resolver=resolver) as records:
        files = context.config("sumo_assignment_trip_files")
        if files is None:
            for batch in _population_batches(context):
                records.add_population(
                    batch, context.config("sumo_assignment_all_as_car")
                )
            files = [context.stage("sumo.macro_scenario.vehicles")]
        for filename in config_tools.file_list(files):
            _read_vehicle_trips(
                filename, types, records, modes.selected(context.config("sumo_modes"))
            )

        # Sort Arrow-backed trip columns in RAM and write XML in bulk.

        return records.write(cache / "queries.trips.xml.gz", types, write_xml=False)


def validate(context):
    fingerprint = hashlib.sha256()
    from sumo.functions.scenario import demand_tools, tools

    for module in (demand_tools, tools):
        fingerprint.update(Path(module.__file__).read_bytes())
    fingerprint.update(
        (
            Path(__file__).parents[1] / "functions" / "cutter" / "demand_tools.py"
        ).read_bytes()
    )
    return {
        "inputs": (
            [
                config_tools.external_token(context, ("sumo_assignment_trip_files",)),
                fingerprint.hexdigest(),
            ]
        ),
        "helpers": source_digest("sumo.functions.macro_scenario.trips"),
    }
