"""Expand selected SUMO demand to the full sample after Swiss-border clipping."""

import hashlib
import json
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import polars as pl

from sumo.config import tools
from sumo.scenario import scaling
from sumo.tables import read_table, write_table

logger = logging.getLogger("synpp")


def configure(context):
    context.stage("sumo.scenario.clipped")
    context.config("input_downsampling")
    context.config("threads", volatile=True)
    context.config("sumo_spatial_upsampling_window", 500)
    context.config("sumo_temporal_upsampling_window", 600)
    context.config("sumo_assignment_trip_files", None)
    factor, spatial, _ = scaling.settings(context.config("input_downsampling"),
        context.config("sumo_spatial_upsampling_window"), context.config("sumo_temporal_upsampling_window"))
    if factor > 1:
        context.config("random_seed")
        if context.config("sumo_assignment_trip_files") is None:
            context.stage("sumo.scenario.population.population")
        else:
            tools.configure_network(context)
        if spatial > 0:
            context.stage("data.statpop.persons")
            context.stage("synthesis.population.destinations")


def execute(context):

    # Load only selected, Swiss-clipped trips and resolve the requested expansion.

    started = time.monotonic()
    source = context.stage("sumo.scenario.clipped")
    factor, spatial, temporal = scaling.settings(context.config("input_downsampling"),
        context.config("sumo_spatial_upsampling_window"), context.config("sumo_temporal_upsampling_window"))
    cache = Path(context.path()).resolve()
    report = {"factor": factor, "spatial_window_m": spatial, "temporal_window_s": temporal,
              "source_trips": source["count"], "source_units": 0, "expanded_units": 0}
    if factor == 1 or not source["count"]:
        report.update({"trips": source["count"], "seconds": time.monotonic()-started})
        target = cache / "scaling_report.json"
        target.write_text(json.dumps(report, indent=2)+"\n")
        return {**source, "scaling_report": str(target), "scale": factor}
    frame = read_table(source["table"])
    schema = frame.to_arrow().schema

    # Identify activity anchors and choose one expansion count per complete person chain.

    activities, resolver = None, None
    if context.config("sumo_assignment_trip_files") is None:
        activities = read_table(context.stage("sumo.scenario.population.population")["activities"])
    else:
        from sumo.cutter.demand_tools import EndpointResolver
        resolver = EndpointResolver(lambda: tools.network_path(context))
    frame, units, anchors = scaling.prepare(frame, activities, resolver)
    streams = [np.random.default_rng(seed) for seed in np.random.SeedSequence(context.config("random_seed")).spawn(3)]
    counts = scaling.replication_counts(units.height, factor, streams[0])

    # Index actual homes and eligible destinations once; sample clone vectors in native code.

    pools = (scaling.candidate_pools(context.stage("data.statpop.persons"),
                                    context.stage("synthesis.population.destinations")) if spatial > 0 else {})
    threads = tools.positive(context.config("threads"), "threads", integer=True)
    if source.get("border"):
        from sumo.scenario.selection import domestic_pools, lock_border_anchors
        pools = domestic_pools(pools, source["border"])
        anchors = lock_border_anchors(anchors, source.get("fixed_endpoints"))
    windows, border_units = scaling.time_windows(units, anchors, temporal)
    shifts = scaling.offsets(units, counts, windows, streams[1])
    report.update({"border_temporal_window_s": max(1800., 2 * temporal),
                   "border_units": int(border_units.sum())})
    coordinates, spatial_report = scaling.sample_anchors(anchors, counts, pools, spatial, streams[2], threads)
    report.update(spatial_report)
    report.update({"source_units": units.height, "expanded_units": int(counts.sum()),
                   "minimum_copies": int(counts.min()), "maximum_copies": int(counts.max())})

    # Stream expanded Arrow batches; final demand preparation performs the global time sort.

    result = scaling.write_expanded(scaling.expanded_batches(frame, counts, coordinates, shifts, schema),
                                     cache / "demand.arrow", schema)
    lineage = units.with_columns(pl.Series("copies", counts), pl.Series("time_window_s", windows))
    write_table(lineage, cache / "replication.arrow")
    types = cache / "demand.types.xml"
    shutil.copyfile(source["types"], types)
    report.update({"trips": result["count"], "seconds": time.monotonic()-started})
    target = cache / "scaling_report.json"
    target.write_text(json.dumps(report, indent=2)+"\n")
    logger.info("SUMO demand upsampling: %s", report)
    return {**result, "queries": None, "types": str(types), "scaling_report": str(target),
            "replication": str(cache / "replication.arrow"), "scale": factor,
            **{key: source[key] for key in ("border_clipping", "border") if key in source}}


def validate(context):
    return hashlib.sha256(Path(scaling.__file__).read_bytes()).hexdigest()
