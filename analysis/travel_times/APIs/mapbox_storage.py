"""Compact CSV indexing and raw batch storage for Mapbox routes."""

from analysis.travel_times.APIs.tomtom_storage import (
    ensure_summary_csv,
    persist_routed_batch,
)


RAW_PATTERNS = ("*.json", "*.json.gz")


def summary_record(identifier, data):
    if not data or data.get("route_info") is None:
        return None

    route = data["route_info"].get("route")
    if not route:
        return None

    duration = route.get("duration")
    if duration is None:
        duration = route.get("duration_typical")
    if route.get("distance") is None or duration is None:
        return None

    return {
        "identifier": str(identifier),
        "distance_km": route["distance"] / 1000,
        "travel_time_min": duration / 60,
        "departure_time": data["departure_time"],
        "origin_x": data["origin_x"],
        "origin_y": data["origin_y"],
        "destination_x": data["destination_x"],
        "destination_y": data["destination_y"],
    }


def ensure_mapbox_summary_csv(summary_directory, raw_directories, metadata_directory):
    return ensure_summary_csv(
        summary_directory,
        raw_directories=raw_directories,
        raw_patterns=RAW_PATTERNS,
        record_factory=summary_record,
        metadata_directory=metadata_directory,
    )


def persist_mapbox_batch(summary_directory, backup_directory, routed_data):
    return persist_routed_batch(
        summary_directory,
        routed_data,
        raw_directory=backup_directory,
        metadata_directory=backup_directory,
        record_factory=summary_record,
        filename_prefix="routed_trips_mapbox",
    )
