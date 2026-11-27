"""Bounded Arrow I/O and compatibility readers for structured SUMO routes."""

import json
from pathlib import Path

import pyarrow as pa
from lxml import etree


def batches(path):
    """Memory-map one IPC record batch at a time without reading a national table."""
    with pa.memory_map(str(path), "r") as stream:
        reader = pa.ipc.open_file(stream)
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)


def write_batches(path, schema, rows, batch_size, batch_bytes):
    """Bound buffering by row count and approximate payload size; retain input order."""
    columns = [[] for _ in schema]
    size = count = 0
    with (
        pa.OSFile(str(path), "wb") as raw,
        pa.BufferedOutputStream(raw, 1024 * 1024) as stream,
        pa.ipc.new_file(stream, schema) as writer,
    ):

        def flush():
            nonlocal size, count
            if count:
                writer.write_batch(
                    pa.RecordBatch.from_arrays(
                        [
                            pa.array(values, type=field.type)
                            for values, field in zip(columns, schema)
                        ],
                        schema=schema,
                    )
                )
                for values in columns:
                    values.clear()
                size = count = 0

        for row in rows:
            for column, value in zip(columns, row):
                column.append(value)
                size += len(value.encode("utf8")) if isinstance(value, str) else 8
            count += 1
            if count >= batch_size or size >= batch_bytes:
                flush()
        flush()


def metadata(path):
    with pa.memory_map(str(path), "r") as stream:
        values = pa.ipc.open_file(stream).schema.metadata or {}
    if values.get(b"sumo_format") != b"sumo-routes-v1":
        raise ValueError(f"Unsupported SUMO route IPC schema: {path}")
    return {key.decode(): value.decode() for key, value in values.items()}


def iter_arrow_routes(path, selected_edges=None, route_filter=None):
    """Compatibility conversion; the default regional cutter consumes IPC in C++."""
    from sumo.functions.tables import read_table

    info = metadata(path)
    edge_ids = read_table(info["sumo_edges"])["id"].to_list()
    root = etree.fromstring(
        b"<routes>" + Path(info["sumo_types"]).read_bytes() + b"</routes>"
    )
    for definition in root:
        yield "vType", definition
    for batch in batches(path):
        for row in batch.to_pylist():
            edges = [edge_ids[index] for index in row["edges"]]
            if selected_edges is not None and not any(
                edge in selected_edges for edge in edges
            ):
                yield "outside", None
                continue
            yield (
                "vehicle",
                {
                    "attributes": json.loads(row["attributes"]),
                    "parameters": json.loads(row["parameters"]),
                    "edges": edges,
                    "exit_times": row["exit_times"],
                    "route_attributes": {
                        "cost": format(row["cost"], ".12g"),
                        "routeLength": format(row["distance"], ".12g"),
                    },
                },
            )
