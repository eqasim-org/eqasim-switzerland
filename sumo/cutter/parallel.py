"""Process-parallel road-path clipping with one shared parent endpoint index."""

from concurrent.futures import ProcessPoolExecutor
import json
import logging
import math
import multiprocessing
from pathlib import Path
import tempfile
import time

from lxml import etree
import polars as pl

from sumo.cutter.route_tools import RoutedBorderClipper, SumoEndpointMatcher
from sumo.cutter.tools import read_region
from sumo.scenario.demand_tools import DEMAND_SCHEMA
from sumo.tables import read_table, write_table

logger = logging.getLogger("synpp")


def worker_count(requested, threads, count):
    """Bound workers by the allocation and avoid tiny automatic partitions."""
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise ValueError("sumo_demand_workers must be a positive integer or null")
    automatic = max(1, math.ceil(count / 10000))
    return min(threads, automatic if requested is None else requested, max(1, count))


def clip_batches(frame, settings, cache, workers):
    """Yield clipped Arrow batches and additive reports in partition order.

    Spawn workers pass only paths. Each reads its own trip partition and selects
    only referenced road shapes. SUMO XML and its spatial index remain solely in
    the parent. Workers never receive a Synpp context or a pickled network.
    """
    if workers == 1:
        clipper = _make_clipper(settings, paths=frame.select("id", "route", "travel_time", "travel_distance"))
        result = clipper(frame)
        yield result, clipper.report
        return
    started = time.monotonic()
    logger.info("SUMO Swiss clipping: %d trips across %d processes", frame.height, workers)
    matcher = object.__new__(RoutedBorderClipper)
    matcher.types = _read_types(settings["types"])
    matcher.matcher = SumoEndpointMatcher(lambda: settings["network"], read_region(settings["border"]), settings["distance"])
    # Temporary IPC parts belong to this consuming stage and are always cleaned.
    with tempfile.TemporaryDirectory(prefix="clipping_", dir=cache) as directory:
        size = math.ceil(frame.height / workers)
        inputs = []
        for number, batch in enumerate(frame.iter_slices(size)):
            source = Path(directory) / f"input_{number:04d}.arrow"
            write_table(batch, source)
            inputs.append(source)
        # Spawn avoids inheriting Polars' already running native thread pool.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = [pool.submit(_clip_partition, str(source), settings) for source in inputs]
            completed = 0
            for future in futures:
                output, report = future.result()
                batch = _match_visits(read_table(output), matcher)
                completed += report["input"]
                logger.info("SUMO parallel clipping: %d/%d trips, %.1f seconds", completed, frame.height,
                            time.monotonic() - started)
                yield batch, report


def _make_clipper(settings, link_ids=None, paths=None):
    return RoutedBorderClipper(settings["border"], settings["links"], settings["routes"],
        lambda: settings["network"], settings["distance"], _read_types(settings["types"]), link_ids=link_ids, paths=paths)


def _read_types(path):
    return {element.get("id"): element for element in
            etree.fromstring(b"<routes>" + Path(path).read_bytes() + b"</routes>")}


def _clip_partition(source, settings):
    # Prevent each spawned process from making a full allocation-sized Polars
    # pool. This is set before this child's first Polars operation.
    import os
    os.environ["POLARS_MAX_THREADS"] = "1"
    frame = read_table(source)
    identifiers = set()
    for row in frame.select("id", "route").iter_rows(named=True):
        try:
            itinerary = json.loads(row["route"])
            for part in itinerary["parts"]:
                if part.get("kind") != "leg":
                    continue
                route = part["route"]
                identifiers.update(route["links"])
                identifiers.update(route[key] for key in ("startLinkId", "endLinkId") if route.get(key))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Cannot clip SUMO trip {row['id']} on its road path: {error}") from error
    clipper = _make_clipper(settings, identifiers, frame.select("id", "route", "travel_time", "travel_distance"))
    result = clipper(frame, defer_matching=True).select(*DEMAND_SCHEMA, "_sumo_clip_visit")
    output = Path(source).with_name(Path(source).stem.replace("input_", "output_") + ".arrow")
    write_table(result, output)
    return str(output), clipper.report


def _match_visits(frame, matcher):
    changed = frame.filter(pl.col("_sumo_clip_visit").is_not_null())
    unchanged = frame.filter(pl.col("_sumo_clip_visit").is_null()).drop("_sumo_clip_visit")
    if not changed.height:
        return unchanged
    rows = []
    for row in changed.iter_rows(named=True):
        visit = json.loads(row.pop("_sumo_clip_visit"))
        try:
            rows.append(matcher.matched_trip(row, visit))
        except ValueError as error:
            raise ValueError(f"Cannot match clipped SUMO trip {row['id']}: {error}") from error
    return pl.concat([unchanged, pl.DataFrame(rows, schema=unchanged.schema)])


def merge_reports(total, report):
    if total is None:
        return dict(report)
    for key, value in report.items():
        if isinstance(value, int):
            total[key] += value
        elif total[key] != value:
            raise ValueError(f"Inconsistent parallel clipping report: {key}")
    return total
