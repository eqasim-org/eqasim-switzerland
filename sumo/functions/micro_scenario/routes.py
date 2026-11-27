"""Stream national routes into bounded, independently processed regional Arrow parts."""

import json
import logging
import tempfile
from pathlib import Path

import polars as pl
from lxml import etree

from sumo.functions.cutter.tools import cut_vehicle
from sumo.functions.native import processing

SCHEMA = {"id": pl.String, "depart": pl.Float64, "xml": pl.Binary}

logger = logging.getLogger("synpp")


def _native_parts(context, loaded, region):
    import pyarrow as pa

    from sumo.functions.config.tools import positive
    from sumo.functions.scenario.route_io import write_batches

    cache = Path(context.path()).resolve()
    schema = pa.schema(
        [
            ("source", pa.string()),
            ("target", pa.string()),
            ("start", pa.float64()),
            ("end", pa.float64()),
            ("length", pa.float64()),
        ]
    )
    mapping = json.loads(Path(region["mapping"]).read_text())
    rows = (
        (source, piece["id"], piece["start"], piece["end"], piece["length"])
        for source, pieces in mapping.items()
        for piece in pieces
    )
    target = cache / "mapping.arrow"
    write_batches(
        target,
        schema,
        rows,
        positive(
            context.config("sumo_route_batch_size"),
            "sumo_route_batch_size",
            integer=True,
        ),
        positive(
            context.config("sumo_route_batch_bytes"),
            "sumo_route_batch_bytes",
            integer=True,
        ),
    )
    output, edges, report = (
        cache / "routes.arrow",
        cache / "edges.arrow",
        cache / "parts.json",
    )
    with tempfile.TemporaryDirectory(prefix="cut_", dir=cache) as temporary:
        temporary = Path(temporary)
        processing.run(
            context,
            "cut",
            routes=loaded["routes"],
            edges=loaded["edges"],
            types=loaded["types"],
            network=region["network"],
            mapping=target,
            output=temporary / output.name,
            output_edges=temporary / edges.name,
            metadata_edges=edges,
            report=temporary / report.name,
        )
        result = json.loads((temporary / report.name).read_text())
        for path in (output, edges):
            (temporary / path.name).replace(path)
    result.update(
        {
            "parts": [str(output)],
            "types": loaded["types"],
            "network": region["network"],
            "route_format": "arrow",
            "edges": str(edges),
        }
    )
    report.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _process(batch, mapping, source, cut, transitions, kernel):
    rows = []
    counts = {
        "count": 0,
        "outside_or_discarded": 0,
        "repeated_visits": 0,
        "interpolated_visits": 0,
    }
    for vehicle in batch:
        visits = cut_vehicle(vehicle, mapping, source, cut, transitions, kernel=kernel)
        counts["outside_or_discarded"] += not visits
        counts["repeated_visits"] += len(visits) > 1
        counts["count"] += len(visits)
        for visit in visits:
            counts["interpolated_visits"] += (
                visit["parameters"]["sumo_boundary_timing"] == "interpolated"
            )
            element = etree.Element("vehicle", attrib=visit["attributes"])
            etree.SubElement(
                element,
                "route",
                attrib={
                    **visit["route_attributes"],
                    "edges": " ".join(visit["edges"]),
                    "exitTimes": " ".join(
                        format(t, ".12g") for t in visit["exit_times"]
                    ),
                },
            )
            for key, value in visit["parameters"].items():
                etree.SubElement(element, "param", key=key, value=str(value))
            rows.append(
                {
                    "id": visit["attributes"]["id"],
                    "depart": float(visit["attributes"]["depart"]),
                    "xml": etree.tostring(element),
                }
            )
    return rows, counts
