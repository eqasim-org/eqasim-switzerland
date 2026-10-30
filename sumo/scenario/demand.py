"""Sort expanded Swiss demand and export SUMO queries once with the bulk writer."""

from pathlib import Path

from lxml import etree

from sumo.scenario.demand_tools import TripSpool
from sumo.config.tools import positive
from sumo.scenario import modes
from sumo.scenario.population.vehicle_tools import BATCH_SIZE
from sumo.tables import read_table


def configure(context):
    modes.require_mode_choice(context)
    modes.configure(context)
    context.stage("sumo.scenario.scaled")
    context.config("sumo_native_xml", True)
    if context.config("sumo_native_xml"):
        context.config("threads", volatile=True)
        context.config("sumo_xml_threads", 4)
        context.stage("sumo.native.runtime")


def execute(context):

    # Read expanded trips whose originals have already been filtered and clipped.

    raw = context.stage("sumo.scenario.scaled")
    frame = read_table(raw["table"])
    types = {element.get("id"): element for element in
             etree.fromstring(b"<routes>" + Path(raw["types"]).read_bytes() + b"</routes>")}
    cache = Path(context.path()).resolve()
    native = context.stage("sumo.native.runtime")["writer"] if context.config("sumo_native_xml") else None

    # Sort departures once and write XML without route parsing or border matching.

    xml_threads = (min(256, positive(context.config("threads"), "threads", integer=True),
                       positive(context.config("sumo_xml_threads"), "sumo_xml_threads", integer=True)) if native else 1)
    with TripSpool(cache / "demand.arrow", native_writer=native, native_threads=xml_threads) as records:
        for batch in frame.iter_slices(BATCH_SIZE):
            records.add_frame(batch)
        result = records.write(cache / "queries.trips.xml.gz", types)
    return {**result, **{key: raw[key] for key in ("border_clipping", "scaling_report") if key in raw}}


def validate(context):
    import hashlib
    from sumo.scenario import demand_tools
    return hashlib.sha256(Path(demand_tools.__file__).read_bytes()).hexdigest()
