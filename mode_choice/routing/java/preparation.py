"""Prepare facility endpoints exactly as MATSim does, before starting the server.

The small input population contains freight only: Java relocates its activities
to facilities. Passenger trips are attached to those same facilities in requests.
No trips are routed in this stage and no upstream files are modified.
"""
import gzip
from pathlib import Path

from lxml import etree
import polars as pl
import pyarrow.parquet as pq

import matsim.runtime.eqasim as eqasim
from matsim.writers import PopulationWriter
from mode_choice.routing.java.freight import REQUEST_SCHEMA

FACILITY_SCHEMA = {"facility_id": pl.String, "x": pl.Float64, "y": pl.Float64, "link_id": pl.String}
FREIGHT_SCHEMA = {**REQUEST_SCHEMA, "origin_link_id": pl.String, "destination_link_id": pl.String}
LINK_SCHEMA = {"link_id": pl.String, "allowed_modes": pl.List(pl.String)}


def configure(context):
    context.stage("matsim.runtime.eqasim")
    context.stage("matsim.runtime.java")
    context.stage("matsim.scenario.network.pt")
    context.stage("matsim.scenario.facilities")
    context.stage("mode_choice.routing.java.freight")
    context.config("threads", volatile=True)
    context.config("java_memory", "50G", volatile=True)


def execute(context):

    # Write freight endpoints for Java's normal facility assignment.

    directory = Path(context.path()).resolve()
    population = directory / "freight_input.xml.gz"
    write_freight(context.stage("mode_choice.routing.java.freight"), population)
    network = directory / "network.xml.gz"
    facilities = directory / "facilities.xml.gz"
    prepared = directory / "freight_prepared.xml.gz"

    # Use the same facility placement and network preparation as the simulation.

    eqasim.run(context, "org.eqasim.core.scenario.preparation.RunPreparation", [
        "--input-facilities-path", context.stage("matsim.scenario.facilities"),
        "--output-facilities-path", str(facilities),
        "--input-population-path", str(population), "--output-population-path", str(prepared),
        "--input-network-path", context.stage("matsim.scenario.network.pt")["network"],
        "--output-network-path", str(network), "--threads", context.config("threads"),
    ])

    # Keep compact, streaming-readable endpoint tables for request construction.

    facility_table = directory / "facilities.parquet"
    freight_table = directory / "freight.parquet"
    link_table = directory / "links.parquet"
    write_table(facility_rows(facilities), FACILITY_SCHEMA, facility_table)
    write_table(freight_rows(prepared), FREIGHT_SCHEMA, freight_table)
    write_table(link_rows(network), LINK_SCHEMA, link_table)
    population.unlink()
    prepared.unlink()
    return {"network": str(network), "facilities": str(facility_table), "freight": str(freight_table), "links": str(link_table)}


def write_freight(trips, path):
    with gzip.open(path, "wb", compresslevel=1) as stream:
        writer = PopulationWriter(stream)
        writer.start_population()
        for row in trips.iter_rows(named=True):
            departure = int(row["departure_time"])
            writer.start_person(row["person_id"])
            writer.start_attributes()
            writer.add_attribute("isFreight", "java.lang.Boolean", "true")
            writer.end_attributes()
            writer.start_plan(True)
            writer.add_activity("freight_loading", (row["origin_x"], row["origin_y"], None), 0, departure)
            writer.add_leg(row["mode"], departure, 3600)
            writer.add_activity("freight_unloading", (row["destination_x"], row["destination_y"], None), departure + 3600)
            writer.end_plan()
            writer.end_person()
        writer.end_population()


def elements(path, name):
    with gzip.open(path, "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",), tag=name):
            yield element
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]


def facility_rows(path):
    for facility in elements(path, "facility"):
        yield {"facility_id": facility.get("id"), "x": float(facility.get("x")),
               "y": float(facility.get("y")), "link_id": facility.get("linkId")}


def freight_rows(path):
    for person in elements(path, "person"):
        origin, destination = person.findall("plan/activity")
        hours, minutes, seconds = map(float, origin.get("end_time").split(":"))
        yield {"identifier": person.get("id"), "person_id": person.get("id"), "trip_index": 0,
               "origin_x": float(origin.get("x")), "origin_y": float(origin.get("y")),
               "destination_x": float(destination.get("x")), "destination_y": float(destination.get("y")),
               "departure_time": hours * 3600 + minutes * 60 + seconds,
               "origin_link_id": origin.get("link"), "destination_link_id": destination.get("link"),
               "mode": "truck"}


def link_rows(path):
    for link in elements(path, "link"):
        yield {"link_id": link.get("id"), "allowed_modes": link.get("modes", "car").split(",")}


def write_table(rows, schema, path):
    arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema
    with pq.ParquetWriter(path, arrow_schema, compression="zstd") as writer:
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) == 65536:
                writer.write_table(pl.DataFrame(batch, schema=schema).to_arrow())
                batch.clear()
        if batch:
            writer.write_table(pl.DataFrame(batch, schema=schema).to_arrow())
