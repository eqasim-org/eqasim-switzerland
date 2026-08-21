"""Memory-efficient storage helpers for TomTom routing responses.

Raw API responses are kept in immutable ``routed_trips_*.json.gz`` batches.
The fields used by the analysis are kept separately in a small CSV index, so a
normal pipeline run never has to deserialize the raw route geometries.

Legacy ``routed_trips*.json`` files can be indexed with bounded memory.  The
streaming reader below expects the existing top-level ``{identifier: route}``
shape but only retains one trip at a time.
"""

import csv
import gzip
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


logger = logging.getLogger("synpp")

SUMMARY_FILENAME = "routed_trips.csv"
MANIFEST_FILENAME = "routed_trips_indexed_files.txt"
SUMMARY_COLUMNS = [
    "identifier",
    "distance_km",
    "travel_time_min",
    "departure_time",
    "origin_x",
    "origin_y",
    "destination_x",
    "destination_y",
]


class _StreamingJSONReader:
    """Incrementally decode values from a text stream."""

    def __init__(self, stream, chunk_size=1024 * 1024):
        self.stream = stream
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _read_more(self):
        if self.eof:
            return False

        # Discard text belonging to values that have already been yielded.
        if self.position:
            self.buffer = self.buffer[self.position :]
            self.position = 0

        chunk = self.stream.read(self.chunk_size)
        if chunk == "":
            self.eof = True
            return False
        self.buffer += chunk
        return True

    def skip_whitespace(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer) or not self._read_more():
                return

    def peek(self):
        self.skip_whitespace()
        if self.position == len(self.buffer) and not self._read_more():
            return ""
        self.skip_whitespace()
        return self.buffer[self.position] if self.position < len(self.buffer) else ""

    def consume(self, expected):
        actual = self.peek()
        if actual != expected:
            raise ValueError(f"Expected {expected!r} in JSON input, found {actual!r}")
        self.position += 1

    def decode(self):
        self.skip_whitespace()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
                self.position = end
                return value
            except json.JSONDecodeError:
                if not self._read_more():
                    raise


def _open_json(path):
    path = Path(path)
    if path.name.endswith(".json.gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_raw_routes(path):
    """Yield ``(identifier, route_data)`` without loading a raw file at once."""

    with _open_json(path) as stream:
        reader = _StreamingJSONReader(stream)
        reader.consume("{")

        if reader.peek() == "}":
            reader.consume("}")
            return

        while True:
            identifier = reader.decode()
            if not isinstance(identifier, str):
                raise ValueError(f"TomTom route identifier must be a string in {path}")
            reader.consume(":")
            yield identifier, reader.decode()

            separator = reader.peek()
            if separator == ",":
                reader.consume(",")
            elif separator == "}":
                reader.consume("}")
                break
            else:
                raise ValueError(f"Expected ',' or '}}' in {path}, found {separator!r}")


def raw_route_files(directory):
    """Return all legacy and compressed raw TomTom files in stable order."""

    directory = Path(directory)
    paths = set(directory.glob("routed_trips*.json"))
    paths.update(directory.glob("routed_trips*.json.gz"))
    return sorted(path for path in paths if path.is_file())


def summary_record(identifier, data):
    """Extract the analysis fields from one stored TomTom response."""

    if not data or data.get("route_info") is None:
        return None

    summary = data["route_info"].get("summary")
    if not summary:
        return None

    travel_time_seconds = summary.get("historicTrafficTravelTimeInSeconds")
    if travel_time_seconds is None:
        # Some TomTom responses only expose the general travel time.
        travel_time_seconds = summary.get("travelTimeInSeconds")

    length_meters = summary.get("lengthInMeters")
    if length_meters is None or travel_time_seconds is None:
        return None

    return {
        "identifier": str(identifier),
        "distance_km": length_meters / 1000,
        "travel_time_min": travel_time_seconds / 60,
        "departure_time": data["departure_time"],
        "origin_x": data["origin_x"],
        "origin_y": data["origin_y"],
        "destination_x": data["destination_x"],
        "destination_y": data["destination_y"],
    }


def _read_manifest(path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as stream:
        return {line.strip() for line in stream if line.strip()}


def _append_manifest(path, filenames):
    filenames = list(filenames)
    if not filenames:
        return
    with path.open("a", encoding="utf-8") as stream:
        for filename in filenames:
            stream.write(f"{filename}\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_routed_ids(summary_path):
    """Read only the identifier column from the compact index."""

    summary_path = Path(summary_path)
    if not summary_path.exists():
        return set()
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        return {row["identifier"] for row in csv.DictReader(stream)}


def append_summary_records(summary_path, records):
    """Append summary rows and durably flush them before returning."""

    records = iter(records)
    try:
        first_record = next(records)
    except StopIteration:
        return 0

    summary_path = Path(summary_path)
    needs_header = not summary_path.exists() or summary_path.stat().st_size == 0
    count = 1
    with summary_path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(first_record)
        for record in records:
            writer.writerow(record)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    return count


def _records_from_files(paths, known_ids=None):
    known_ids = set() if known_ids is None else known_ids
    for path in paths:
        logger.info("Indexing raw TomTom routes from %s", path)
        file_count = 0
        for identifier, data in iter_raw_routes(path):
            identifier = str(identifier)
            if identifier in known_ids:
                continue
            record = summary_record(identifier, data)
            if record is not None:
                known_ids.add(identifier)
                file_count += 1
                yield record
                if file_count % 10_000 == 0:
                    logger.info("Indexed %s trips from %s", file_count, path)


def rebuild_summary_csv(directory):
    """Stream all raw route files into a new CSV and replace the old index."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / SUMMARY_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    paths = raw_route_files(directory)
    temporary_path = summary_path.with_name(f".{summary_path.name}.{uuid4().hex}.tmp")

    try:
        append_summary_records(temporary_path, _records_from_files(paths))
        if not temporary_path.exists():
            with temporary_path.open("w", encoding="utf-8", newline="") as stream:
                csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS).writeheader()
        os.replace(temporary_path, summary_path)

        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
        with temporary_manifest.open("w", encoding="utf-8") as stream:
            for path in paths:
                stream.write(f"{path.name}\n")
        os.replace(temporary_manifest, manifest_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    logger.info("TomTom summary index is available at %s", summary_path)
    return summary_path


def ensure_summary_csv(directory):
    """Create the CSV once, then index only raw files not in the manifest."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / SUMMARY_FILENAME
    manifest_path = directory / MANIFEST_FILENAME

    if not summary_path.exists():
        return rebuild_summary_csv(directory)

    indexed_files = _read_manifest(manifest_path)
    pending_files = [path for path in raw_route_files(directory) if path.name not in indexed_files]
    if pending_files:
        known_ids = read_routed_ids(summary_path)
        append_summary_records(summary_path, _records_from_files(pending_files, known_ids))
        _append_manifest(manifest_path, (path.name for path in pending_files))

    return summary_path


def save_raw_batch(directory, routed_data):
    """Atomically save one complete, compact, gzip-compressed JSON batch."""

    if not routed_data:
        return None

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_path = directory / f"routed_trips_{timestamp}.json.gz"
    temporary_path = directory / f".{output_path.name}.{uuid4().hex}.tmp"

    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(routed_data, stream, separators=(",", ":"))
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return output_path


def persist_routed_batch(directory, routed_data):
    """Save raw responses, append their summaries, and mark the batch indexed."""

    if not routed_data:
        return None

    directory = Path(directory)
    summary_path = directory / SUMMARY_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    raw_path = save_raw_batch(directory, routed_data)
    records = (
        record
        for identifier, data in routed_data.items()
        if (record := summary_record(identifier, data)) is not None
    )
    append_summary_records(summary_path, records)
    _append_manifest(manifest_path, [raw_path.name])
    logger.info("Saved %s new TomTom routes to %s", len(routed_data), raw_path)
    return raw_path
