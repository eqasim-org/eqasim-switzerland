"""Sort and export cached regional route parts without repeating national cutting."""

import json
import tempfile
from pathlib import Path

import polars as pl

from sumo.functions.native import processing
from sumo.functions.scenario.rerouting import retention_settings


def _native_export(context, source):
    from sumo.functions.config.tools import positive

    fraction, seed = retention_settings(context)
    cache = Path(context.path()).resolve()
    sorted_routes = cache / "sorted.arrow"
    pl.concat([pl.scan_ipc(path) for path in source["parts"]]).sort(
        "depart", "id"
    ).sink_ipc(
        sorted_routes,
        record_batch_size=positive(
            context.config("sumo_route_batch_size"),
            "sumo_route_batch_size",
            integer=True,
        ),
    )
    counts = (
        pl.scan_ipc(sorted_routes)
        .select(pl.len().alias("rows"), pl.col("id").n_unique().alias("ids"))
        .collect()
    )
    if counts["rows"][0] != counts["ids"][0]:
        raise ValueError("Duplicate regional vehicle IDs")
    if counts["rows"][0] != source["count"]:
        raise ValueError("Regional Arrow row count does not match the cutting report")
    output = cache / "region.routes.xml.gz"
    with tempfile.TemporaryDirectory(prefix="export_", dir=cache) as temporary:
        temporary = Path(temporary) / output.name
        processing.run(
            context,
            "export",
            routes=sorted_routes,
            edges=source["edges"],
            types=source["types"],
            network=source["network"],
            output=temporary,
            initial_route_retention=fraction,
            retention_seed=seed,
        )
        temporary.replace(output)
    result = {
        key: value
        for key, value in source.items()
        if key not in {"parts", "types", "route_format", "edges"}
    }
    result["routes"] = str(output)
    (cache / "cut_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
